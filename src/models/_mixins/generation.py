import gc
import json
import torch

from typing import Union, List, Iterable

from abc import abstractmethod

from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

class _GenerationMixin:
    @abstractmethod
    def generate():
        """To generate something.
        """
    
    @abstractmethod
    def generate_stream():
        """To generate something in a stream.
        """

class DummyGenerationMixin(_GenerationMixin):
    def generate(self, 
        input_texts: Union[List[str], str] = "",
    ):
        return input_texts
    
    def generate_stream(self, 
        input_texts: Union[List[str], str] = "",
    ):
        return input_texts
    

class LLMGenerationMixin(_GenerationMixin):
    @torch.inference_mode()
    def generate(
        self, 
        input_texts: Union[List[str], str] = "",
        generation_config: dict = {},
        encode_config: dict = {},
        decode_config: dict = {},
        stream_config: dict = {},
    ):
        # TODO
        self.generation_config.update_values(generation_config)
        self.encode_config.update_values(encode_config)
        self.decode_config.update_values(decode_config)
        self.stream_config.update_values(stream_config)
        
        generation_config = self.generation_config
        encode_config = self.encode_config
        decode_config = self.decode_config
        stream_config = self.stream_config
        
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
        # TODO cqju
        self.generation_config.update_values(generation_config)
        self.encode_config.update_values(encode_config)
        self.decode_config.update_values(decode_config)
        self.stream_config.update_values(stream_config)
        
        generation_config = self.generation_config
        encode_config = self.encode_config
        decode_config = self.decode_config
        stream_config = self.stream_config

        # inherited from fastchat.serve.inference
        temperature = float(generation_config.get("temperature", 1.0))
        repetition_penalty = float(generation_config.get("repetition_penalty", 1.0))
        top_p = float(generation_config.get("top_p", 1.0))
        top_k = int(generation_config.get("top_k", -1))  # -1 means disable
        max_new_tokens = int(generation_config.get("max_new_tokens", 256))
        context_len = generation_config.get("max_length", 2048)
        
        # include_input = stream_config.get("include_input", False)
        stream_new_tokens = stream_config.get("stream_new_tokens", False)
        stream_interval: int = stream_config.get("max_length", 3)
        stop_str = stream_config.get("stop_str", None)
        if isinstance(stop_str, str):
            stop_str = [e for e in stop_str.split(";") if e]
        elif isinstance(stop_str, Iterable):
            stop_str = [e for e in stop_str if e]
        
        stop_token_ids = stream_config.get("stop_token_ids", None) or []
        stop_token_ids.append(self.tokenizer.eos_token_id)
        
        input_ids = self.tokenizer(input_texts).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)

        logits_processor = prepare_logits_processor(
            temperature, repetition_penalty, top_p, top_k
        )
        
        if self.model.config.is_encoder_decoder:
            max_src_len = context_len
        else:
            # max_src_len = context_len - max_new_tokens - 8
            max_src_len = max(context_len - max_new_tokens, 0)

        input_ids = input_ids[-max_src_len:]

        if self.model.config.is_encoder_decoder:
            encoder_output = self.model.encoder(
                input_ids=torch.as_tensor([input_ids], device=self.first_device)
            )[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=self.first_device,
            )

        output = ""
        past_key_values = out = None
        for i in range(max_new_tokens):
            if i == 0:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=start_ids,
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                    )
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([input_ids], device=self.first_device), 
                        use_cache=True,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(
                        input_ids=torch.as_tensor([[token]], device=self.first_device),
                        encoder_hidden_states=encoder_output,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )

                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([[token]], device=self.first_device),
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

            if self.first_device == "mps":
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
                # if include_input:
                #     tmp_output_ids = output_ids
                #     rfind_start = len_prompt
                #     last_output_len = 0
                # else:
                if stream_new_tokens:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0
                    last_output_len = len(output)
                else:    
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0
                    last_output_len = 0
                    
                output = self.tokenizer.decode(
                    token_ids=tmp_output_ids,
                    skip_special_tokens=stream_config.get("skip_special_tokens", False),
                    spaces_between_special_tokens=stream_config.get("spaces_between_special_tokens", True),
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
                        "generated_text": output[last_output_len:],
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

class STGenerationMixin(_GenerationMixin):
    @torch.inference_mode()
    def generate(
        self, 
        input_texts: Union[List[str], str] = "",
        generation_config: dict = {},
        encode_config: dict = {},
        decode_config: dict = {},
        stream_config: dict = {},
    ):
        # TODO
        self.generation_config.update_values(generation_config)
        self.encode_config.update_values(encode_config)
        self.decode_config.update_values(decode_config)
        self.stream_config.update_values(stream_config)
        
        embeddings = self.model.encode(
            sentences=input_texts,
            batch_size=self.generation_config.get("batch_size", 32), # TODO
            show_progress_bar=False
        )

        return embeddings.tolist()
    
    def generate_stream(self):
        raise NotADirectoryError("Method not implemented.")

      
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