# from abc import ABCMeta, abstractmethod
from abc import abstractmethod
from typing import Union, List

from .model_loading import load_model, load_tokenizer, load_all_configs

class BaseLM(object):
    def __init__(self) -> None:
        # Load model and tokenizer and generation_config
        self._model, self._first_device = load_model()
        self._tokenizer = load_tokenizer()
        self._generation_config, self._encode_config, self._decode_config, self._stream_config = load_all_configs()

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

    @abstractmethod
    def generate_response(self, 
            input_texts: Union[List[str], str] = "",
            generation_config: dict = {},
        ):
        pass
