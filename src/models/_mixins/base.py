# from abc import abstractmethod
from typing import Union, List

from ..utils import load_config_from_yaml, get_first_device

class Base:
    def __init__(self) -> None:
        self._first_device = get_first_device(_print=True)
        self._model = None
        self._tokenizer = None
        self._generation_config = None
        
        for ele in [
            "generation_config", "encode_config", "decode_config", "stream_config"
        ]:
            self.__setattr__(ele, load_config_from_yaml(ele))
        
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
    
    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value
    
    @generation_config.deleter
    def generation_config(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def encode_config(self):
        return self._encode_config
    
    @encode_config.setter
    def encode_config(self, value):
        self._encode_config = value
    
    @encode_config.deleter
    def encode_config(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def decode_config(self):
        return self._decode_config
    
    @decode_config.setter
    def decode_config(self, value):
        self._decode_config = value
    
    @decode_config.deleter
    def decode_config(self):
        raise AttributeError("Can't delete attribute")
    
    @property
    def stream_config(self):
        return self._stream_config
    
    @stream_config.setter
    def stream_config(self, value):
        self._stream_config = value
    
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

    def func_generate_response(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            # TODO
            # if not generation_config and self._generation_config:
            #     generation_config = self._generation_config
            if hasattr(self, "generate"):
                return self.generate(
                    input_texts=input_texts, 
                )
            else:
                err = NotImplementedError("Method is not implemented.")
                raise err
            
    def func_generate_response_stream(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            if hasattr(self, "generate_stream"):
                return self.generate_stream(
                    input_texts=input_texts, 
                )
            else:
                err = NotImplementedError("Method is not implemented.")
                raise err
