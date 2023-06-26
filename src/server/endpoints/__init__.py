import time
from fastapi.responses import StreamingResponse

from ...logging import logger

from ..types import BatchRequest, BatchResponse
from ..utils import gen_sha
from ...models import ModelLM

# TODO: cqju: how to move this out of skeleton?
modelLM = ModelLM()

async def api_probe():
    """ Placeholder endpoint """
    trace_id = gen_sha()
    return BatchResponse(
        trace_id=trace_id,
        generated_text="OK",
    )
    
async def api_inference(payload: BatchRequest): 
    t_s = time.time()
    trace_id = gen_sha()
    
    payload = payload.dict()
    
    if isinstance(payload["inputs"], str):
        payload["inputs"] = [payload["inputs"]]
    
    logger.info({
        "trace_id": trace_id,
        "inputs": payload["inputs"],
    })

    if isinstance(payload["inputs"], str):
        payload["inputs"] = [payload["inputs"]]

    result = modelLM.func_generate_response(
        input_texts=payload["inputs"],
    )
    
    logger.info({
        "trace_id": trace_id,
        "outputs": result,
    })

    t_e = time.time() - t_s

    if len(result) == 1:
        result = result[0]
        
    return [
        BatchResponse(
            # trace_id=trace_id,
            generated_text=result,
            elasped_time=round(t_e, 3),
        ).dict(exclude_none=True)
    ]
    
async def api_inference_stream(payload: BatchRequest):
    payload = payload.dict()
    # TODO: change input type
    # if isinstance(payload["inputs"], str):
    #     payload["inputs"] = [payload["inputs"]]
    
    result = modelLM.func_generate_response_stream(
        input_texts=payload["inputs"],
    )
    
    return StreamingResponse(result)

async def api_embedding(payload: BatchRequest):
    t_s = time.time()
    trace_id = gen_sha()
    payload = payload.dict()

    if isinstance(payload["inputs"], str):
        payload["inputs"] = [payload["inputs"]]

    logger.info({
        "trace_id": trace_id,
        "inputs": payload["inputs"],
    })
    
    result = modelLM.func_generate_response(
        input_texts=payload["inputs"],
    )

    logger.info({
        "trace_id": trace_id,
        "embedding": result,
    })

    t_e = time.time() - t_s

    return BatchResponse(
        trace_id=trace_id,
        embeddings=result,
        elasped_time=round(t_e, 3),
    ).dict(exclude_none=True)
