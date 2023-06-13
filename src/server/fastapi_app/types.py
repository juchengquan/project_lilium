from pydantic import BaseModel
from typing import Union, List
 
class BatchRequest(BaseModel):
    inputs: Union[List[str], str, None] = None

class BatchResponse(BaseModel):
    generated_text: Union[List[str], str, None] = None
    elasped_time: Union[float, None] = 0.0