import pathlib
import yaml
import importlib
from fastapi import FastAPI, APIRouter
from fastapi.responses import UJSONResponse

from ..logging import logger
from ..model_configuration import MODEL_CONFIG

def get_app():
    try:
        app = FastAPI(default_response_class=UJSONResponse)
        app_router = APIRouter(default_response_class=UJSONResponse)
        
        cwd = pathlib.Path(__file__).parent.resolve()
        with open(str(cwd) + "/service_endpoints.yaml", "r") as file:
            endpoint_setting = yaml.load(file, Loader=yaml.FullLoader)

        from .endpoints import api_probe
        app.add_api_route(path="/", endpoint=api_probe, methods=["GET", "POST"])
        
        
        from functools import partial
        for _services in endpoint_setting.get(MODEL_CONFIG.get("type"), []):
            _endpoints = importlib.import_module("..endpoints", package=__name__)
            
            if hasattr(_endpoints, _services["endpoint"]):
                pkg_endpoint = getattr(_endpoints, _services["endpoint"])
                
                # TODO: cqju: Seems wierd for post method working
                # @app.post(path=_services["path"],)
                # async def shell_func(payload: BatchRequest):
                #     logger.info("WTF")
                #     return await pkg_endpoint(payload=payload, modelLM=modelLM)
                
                # TODO
                # partial(pkg_endpoint, modelLM=modelLM)
                app.add_api_route(
                    path=_services["path"],
                    endpoint=pkg_endpoint, # partial(pkg_endpoint, modelLM=modelLM),
                    methods=["POST"],
                )
                logger.info(f'Binded function {pkg_endpoint.__name__} at API endpoint: {_services["path"]}')
                
            else:
                raise ValueError(f'No such endpoint: {_services["endpoint"]}')
        
        app.include_router(app_router) #, prefix=f'/{MODEL_CONFIG.get("type")}')
        logger.info("Application has been started.")
        return app
    except (ImportError) as err:
        logger.error(err.msg if err.msg else str(err))
        raise(err)
    except Exception as err:
        logger.error(err.msg if err.msg else str(err))
        raise(err)

app = get_app()
