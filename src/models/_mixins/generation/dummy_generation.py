
from typing import Union, List, Iterable

from ._base import _GenerationMixin

class DummyGenerationMixin(_GenerationMixin):
    def generate(self, 
        input_texts: Union[List[str], str] = "",
    ):
        return input_texts
    
    def generate_stream(self, 
        input_texts: Union[List[str], str] = "",
    ):
        return input_texts
    