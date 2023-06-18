import os
import importlib
from ..model_configuration import MODEL_CONFIG
from ..utils.logging import logger

try:
    from ._mixins.base import Base
    mixin_list = [Base]
    
    mixin_class = importlib.import_module(f'._mixins', package=__name__)
    for _m in MODEL_CONFIG["mixins"]:
        logger.info(f"Registering mixin: {_m}.")
        if hasattr(mixin_class, _m):
            mixin_list.append(getattr(mixin_class, _m))
        else:
            raise NotImplementedError("Mixin in the list nor implemented or wrongly set.")
        
    _model = importlib.import_module(f'.{MODEL_CONFIG["type"]}.{MODEL_CONFIG["model_template"]}', package=__name__)
    logger.info(f"Creating the model object with type: {_model.__name__}.")
    
    ModelLM = _model.createClass(mixin_list)
    logger.info("All mixins are registered successfully.")
except Exception as err:
    print(err)
    logger.error(err)
    raise ImportError(f'Cannot import module: {MODEL_CONFIG["model_template"]}')

__all__ = [
    "ModelLM"
]