from typing import Union, List

def createClass(cls_list):
    class HuggingFaceLM(*cls_list):
        def __init__(self):
            self._model = None
            self._tokenizer = None
            self._generation_config = None

        def generate_response(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            return f"Dummy result: {input_texts}"
    
    return HuggingFaceLM
