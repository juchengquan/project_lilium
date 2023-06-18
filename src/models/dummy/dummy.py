from typing import Union, List

def createClass(cls_list):
    class ABCLM(*cls_list):
        def __init__(self):
            # super().__init__()
            self._model = None
            self._tokenizer = None
            self._generation_config = None

        def func_generate_response(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            return f"Dummy result: {input_texts}"
        
        def func_generate_response_stream(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            import time
            for _ in range(5):
                yield f"Dummy result: {input_texts}".encode()
                time.sleep(0.25)
    
    return ABCLM
