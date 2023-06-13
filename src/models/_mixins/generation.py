import gc
import json
import torch

from typing import Union, List, Iterable

from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

class LLMGenerationMixin:
    @torch.inference_mode()
    def generate(
        self, 
        input_texts: Union[List[str], str] = "",
        generation_config: dict = {},
        encode_config: dict = {},
        decode_config: dict = {},
    ):
        input_seq = self.tokenizer(
                text=input_texts, 
                return_tensors="pt",
                **encode_config,
            ) \
            .to(self.first_device)

        ### generation
        input_seq.update({"generation_config": GenerationConfig(**generation_config),}) 
        
        if encode_config["padding"]:
            input_seq.update({"pad_token_id": self._tokenizer.pad_token_id,}) 

        generation_output = self.model.generate(**input_seq)
        torch.cuda.empty_cache()
        
        ### decode
        input_len = len(input_seq["input_ids"]) if decode_config.get("new_tokens_only", False) else 0
        if "new_tokens_only" in decode_config.keys():
            decode_config.pop("new_tokens_only")
        
        output_texts = self.tokenizer.batch_decode(
            generation_output.sequences[:, input_len:],
            **decode_config,
        )
        return output_texts

    @torch.inference_mode()
    def generate_stream(
        self,
        input_texts: Union[List[str], str] = "",
        generation_config: dict = {},
        encode_config: dict = {},
        decode_config: dict = {},
        stream_config: dict = {},
    ):
        params = generation_config
        tokenizer = self.tokenizer
        model = self.model
        
        # inherited from fastchat.serve.inference:
        prompt = input_texts
        len_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))
        
        context_len = encode_config.get("max_length", 2048)
        
        echo = decode_config.get("new_tokens_only", False) # TODO
        stream_interval: int = stream_config.get("max_length", 2)
        stop_str = stream_config.get("stop_str", None)
        if isinstance(stop_str, str):
            stop_str = [e for e in stop_str.split(";") if e]
        elif isinstance(stop_str, Iterable):
            stop_str = [e for e in stop_str if e]
            
        print(stop_str)
        
        stop_token_ids = stream_config.get("stop_token_ids", None) or []
        stop_token_ids.append(self.tokenizer.eos_token_id)
        
        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )
        
        if model.config.is_encoder_decoder:
            max_src_len = context_len
        else:
            max_src_len = context_len - max_new_tokens - 8

        input_ids = input_ids[-max_src_len:]

        if model.config.is_encoder_decoder:
            encoder_output = model.encoder(
                input_ids=torch.as_tensor([input_ids], device=self._first_device)
            )[0]
            start_ids = torch.as_tensor(
                [[model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=self._first_device,
            )

        past_key_values = out = None
        for i in range(max_new_tokens):
            if i == 0:
                if model.config.is_encoder_decoder:
                    out = model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = model.lm_head(out[0])
                else:
                    out = model(torch.as_tensor([input_ids], device=self._first_device), use_cache=True)
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                if model.config.is_encoder_decoder:
                    out = model.decoder(
                        input_ids=torch.as_tensor([[token]], device=self._first_device),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )

                    logits = model.lm_head(out[0])
                else:
                    out = model(
                        input_ids=torch.as_tensor([[token]], device=self._first_device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values

            if logits_processor:
                if repetition_penalty > 1.0:
                    tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
                else:
                    tmp_output_ids = None
                last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
            else:
                last_token_logits = logits[0, -1, :]

            if self._first_device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-5 or top_p < 1e-8:  # greedy
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0
                    # modity to make sure it only outputs new tokens every time
                    input_echo_len = len(output_ids) 

                output = tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=decode_config.get("skip_special_tokens", False),
                    spaces_between_special_tokens=decode_config.get("spaces_between_special_tokens", False),
                )

                partially_stopped = False
                if stop_str:
                    if isinstance(stop_str, str):
                        pos = output.rfind(stop_str, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                        else:
                            partially_stopped = partial_stop(output, stop_str)
                    elif isinstance(stop_str, Iterable):
                        for each_stop in stop_str:
                            pos = output.rfind(each_stop, rfind_start)
                            if pos != -1:
                                output = output[:pos]
                                stopped = True
                                break
                            else:
                                partially_stopped = partial_stop(output, each_stop)
                                if partially_stopped:
                                    break
                    else:
                        raise ValueError("Invalid stop field type.")

                # prevent yielding partial stop sequence
                if not partially_stopped:
                    # finish stream event, which contains finish reason
                    if i == max_new_tokens - 1:
                        finish_reason = "length"
                    elif stopped:
                        finish_reason = "stop"
                    else:
                        finish_reason = None
                        
                    res = {
                        "generted_text": output,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": i,
                            "total_tokens": input_echo_len + i,
                        },
                        "finish_reason": finish_reason,
                    }
                    yield json.dumps(res)
                    # yield output
                    
            if stopped:
                break

        # finish stream event, which contains finish reason
        if i == max_new_tokens - 1:
            finish_reason = "length"
        elif stopped:
            finish_reason = "stop"
        else:
            finish_reason = None

        # clean
        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()
        
def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list