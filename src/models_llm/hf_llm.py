from typing import Union, List

def createClass(cls_list):
    class HuggingFaceLM(*cls_list):
        def __init__(self):
            super().__init__()
        
        def generate_response(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            # TODO
            # if not generation_config and self._generation_config:
            #     generation_config = self._generation_config
            
            output_texts = self.generate(
                input_texts=input_texts, 
                generation_config=self.generation_config,
                encode_config=self.encode_config,
                decode_config=self.decode_config,
            )

            return output_texts
        
    return HuggingFaceLM
