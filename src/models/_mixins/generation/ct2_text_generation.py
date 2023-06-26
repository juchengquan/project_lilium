import torch
from typing import Union, List, Iterable

from transformers import GenerationConfig

from ._base import _GenerationMixin
from ....logging import logger # TODO

class CT2TextGenerationMixin(_GenerationMixin):
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
        
        # TODO
        logger.info(self.tokenizer(input_texts[0]).input_ids)
        
        input_ids = self.tokenizer(input_texts).input_ids
        input_seq = [
            self.tokenizer.convert_ids_to_tokens(ele)
            for ele in input_ids
        ]
        logger.info(input_seq)
        
        generation_output = self.model.generate_batch(
            input_seq, 
            max_length=generation_config["max_new_tokens"], 
            sampling_topk=generation_config["top_k"],
            sampling_topp=generation_config["top_p"],
            sampling_temperature=generation_config["temperature"],
            repetition_penalty=generation_config["repetition_penalty"],
            include_prompt_in_result=not decode_config.get("new_tokens_only", False),
            
        )

        logger.info(generation_output)
        # input_len = len(input_ids["input_ids"]) if decode_config.get("new_tokens_only", False) else 0
        input_len = 0
        if "new_tokens_only" in decode_config.keys():
            decode_config.pop("new_tokens_only")
            
        output_texts = [self.tokenizer.decode(
            ele.sequences_ids[0][input_len:], 
            # **decode_config,
        ) for ele in generation_output]
        # ououtput_texts = self.tokenizer.batch_decode(
        #     generation_output.sequences[:, input_len:],
        #     **decode_config,
        # )
        logger.info(output_texts)
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
        # TODO
        yield "Not Implemented"
        # raise NotImplementedError("Not implemented.")
