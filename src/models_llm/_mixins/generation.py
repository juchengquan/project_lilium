import torch
from transformers import GenerationConfig
from typing import Union, List


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
