from pydantic import BaseModel
from typing import Union, List
 
class BatchRequest(BaseModel):
    inputs: Union[List[str], str, None] = None

class BatchResponse(BaseModel):
    trace_id: str = None
    generated_text: Union[List[str], str, None] = None
    embeddings: Union[List[List[float]], List[float]] = None
    elasped_time: Union[float, None] = 0.0