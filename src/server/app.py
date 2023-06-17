import pathlib
import yaml
from fastapi import FastAPI
from fastapi.responses import UJSONResponse

from ..utils.logging import logger
from ..model_configuration import MODEL_CONFIG
from ..models import ModelLM

def get_app():
    try:
        app = FastAPI(default_response_class=UJSONResponse)
        
        cwd = pathlib.Path(__file__).parent.resolve()
        with open(str(cwd) + "/endpoints.yaml", "r") as file:
            endpoint_setting = yaml.load(file, Loader=yaml.FullLoader)

        from .commons import api_probe
        app.add_api_route(path="/", endpoint=api_probe, methods=["GET", "POST"])
        
        modelLM = ModelLM()
        from functools import partial
        for _services in endpoint_setting.get(MODEL_CONFIG.get("type"), []):
            exec(f'from .commons import {_services["endpoint"]}')
            
            app.add_api_route(
                path=_services["path"],
                endpoint=eval(f'partial({_services["endpoint"]}, modelLM=modelLM)'),
                methods=["POST"],
            )
        
        logger.info("Application has been started.")
        return app
    except (ImportError) as err:
        logger.error(err.msg if err.msg else str(err))
        raise(err)
    except Exception as err:
        logger.error(err.msg if err.msg else str(err))
        raise(err)

app = get_app()
