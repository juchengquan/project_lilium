import os, time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .types import BatchRequest, BatchResponse

from ..utils.logging import logger

from ..models import ModelLM
modelLM = ModelLM()

app = FastAPI()

@app.get("/")
async def probe():
    """ Placeholder endpoint """
    return BatchResponse(
        generated_text="OK",
    )

# @app.post("/embed")
# async def api_embed(payload: BatchRequest):
#     t_s = time.time()
#     payload = payload.dict()

#     if isinstance(payload["inputs"], str):
#         payload["inputs"] = [payload["inputs"]]

#     result = modelLM.generate_response(
#         input_texts=payload["inputs"],
#     )

#     logger.info({
#         "inputs": payload,
#         "outputs": result,
#     })

#     t_e = time.time() - t_s

#     return BatchResponse(
#         # generated_text=result,
#         embeddings=result,
#         elasped_time=round(t_e, 3),
    # )

@app.post("/infer")
async def api_inference(payload: BatchRequest):
    t_s = time.time()
    payload = payload.dict()

    if isinstance(payload["inputs"], str):
        payload["inputs"] = [payload["inputs"]]

    result = modelLM.generate_response(
        input_texts=payload["inputs"],
    )

    logger.info({
        "inputs": payload["inputs"],
        "outputs": result,
    })

    t_e = time.time() - t_s

    if len(result) == 1:
        result = result[0]

    return [BatchResponse(
        generated_text=result,
        elasped_time=round(t_e, 3),
    )]

@app.post("/infer_stream")
async def api_inference_stream(payload: BatchRequest):
    payload = payload.dict()
    # TODO: change input type
    # if isinstance(payload["inputs"], str):
    #     payload["inputs"] = [payload["inputs"]]
    
    result = modelLM.generate_response_stream(
        input_texts=payload["inputs"],
    )
    
    return StreamingResponse(result)