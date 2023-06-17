from abc import abstractmethod
from typing import Union, List

from ...utils.logging import logger
from ...utils.funcs import get_first_device

class Base(object):
    def __init__(self) -> None:
        self._first_device = get_first_device(_print=True)
        self._model = None
        self._tokenizer = None
        self._generation_config = None
        
        # Load model and tokenizer and generation_config
        # self._model = load_hf_model()
        # self._tokenizer = load_tokenizer()
        # self._generation_config, self._encode_config, self._decode_config, self._stream_config = load_all_configs()

    @property
    def model(self):
        return self._model
    
    # @model.setter
    # def model(self, value):
    #     self._model = value
        
    @model.deleter
    def model(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    # @tokenizer.setter
    # def tokenizer(self, value):
    #     self._tokenizer = value
    
    @tokenizer.deleter
    def tokenizer(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def generation_config(self):
        return self._generation_config
    
    # @generation_config.setter
    # def generation_config(self, value):
    #     self._generation_config = value
    
    @generation_config.deleter
    def generation_config(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def encode_config(self):
        return self._encode_config
    
    # @encode_config.setter
    # def encode_config(self, value):
    #     self._encode_config = value
    
    @encode_config.deleter
    def encode_config(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def decode_config(self):
        return self._decode_config
    
    # @decode_config.setter
    # def decode_config(self, value):
    #     self.decode_config = value
    
    @decode_config.deleter
    def decode_config(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def stream_config(self):
        return self._stream_config
    
    # @stream_config.setter
    # def stream_config(self, value):
    #     self.stream_config = value
    
    @stream_config.deleter
    def stream_config(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def first_device(self):
        return self._first_device
    
    # @first_device.setter
    # def first_device(self, value):
    #     self._first_device = value
    
    @first_device.deleter
    def first_device(self):
        raise AttributeError("Can't delete attribute")

    def api_generate_response(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            # TODO
            # if not generation_config and self._generation_config:
            #     generation_config = self._generation_config
            if hasattr(self, "generate"):
                return self.generate(
                    input_texts=input_texts, 
                    generation_config=self.generation_config,
                    encode_config=self.encode_config,
                    decode_config=self.decode_config,
                )
            else:
                err = NotImplementedError("Method is not implemented.")
                raise err
            
    def api_generate_response_stream(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            if hasattr(self, "generate_stream"):
                return self.generate_stream(
                    input_texts=input_texts, 
                    generation_config=self.generation_config,
                    stream_config=self.stream_config,
                )
            else:
                err = NotImplementedError("Method is not implemented.")
                raise err