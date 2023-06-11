import os, time

from fastapi import FastAPI
from .types import BatchRequest, BatchResponse

from ...utils.logging import logger
from ...models_llm import HuggingFaceLM

app = FastAPI()

@app.get("/")
async def probe():
    """ Placeholder endpoint """
    return BatchResponse(
        generated_text="OK",
    )

@app.post("/infer")
async def inference(payload: BatchRequest):
    t_s = time.time()
    payload = payload.dict()

    if isinstance(payload["inputs"], str):
        payload["inputs"] = [payload["inputs"]]

    result = HuggingFaceLM.generate_response(
        input_texts=payload["inputs"],
    )

    logger.info({
        "inputs": payload,
        "outputs": result,
    })

    t_e = time.time() - t_s

    if len(result) == 1:
        result = result[0]

    return [BatchResponse(
        generated_text=result,
        elasped_time=round(t_e, 3),
    )]
