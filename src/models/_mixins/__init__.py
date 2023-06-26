
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

from .generation import DummyGenerationMixin, HFTextGenerationMixin, STGenerationMixin, CT2TextGenerationMixin
# TODO: to be more robust
from .post_processing import PostProcessorMixin
from .pre_processing import PreProcessorMixin



__all__ = [
    "DummyGenerationMixin", 
    
    "HFTextGenerationMixin",
    "CT2TextGenerationMixin"
    
    "STGenerationMixin",
    
    "PreProcessorMixin",
    "PostProcessorMixin"
]