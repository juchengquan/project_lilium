
def createClass(cls_list):
    class ABCLM(*cls_list):
        def __init__(self):
            super().__init__()
            self._initialize()
        
        def _initialize(self):
            if hasattr(self, "load_model"):
                self._model = self.load_model()
            else:
                raise NotImplemented('Function "load_model" not implemented')
            if hasattr(self, "load_tokenizer"):
                self._tokenizer = self.load_tokenizer()
            else:
                raise NotImplemented('Function "load_tokenizer" not implemented')
            return self
    
    return ABCLM

from .generation import DummyGenerationMixin

from .generation import LLMGenerationMixin
from .post_processing import PostProcessorMixin
from .pre_processing import PreProcessorMixin

from .generation import STGenerationMixin

__all__ = [
    "DummyGenerationMixin", 
    
    "LLMGenerationMixin",
    
    "STGenerationMixin",
    
    "PreProcessorMixin",
    "PostProcessorMixin"
]