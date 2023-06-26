import torch
from typing import Union, List, Iterable

from ._base import _GenerationMixin

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
