import os
import importlib
from ..model_configuration import MODEL_CONFIG
from ..utils.logging import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid deadlock warning

__all__ = []
# try:
exec("from ._mixins.base import Base")
if MODEL_CONFIG['mixins']:
    exec(f"from ._mixins import {','.join(MODEL_CONFIG['mixins'])}") 

_m = importlib.import_module(f".{MODEL_CONFIG['type']}.{MODEL_CONFIG['template_name']}", __name__)
ModelLM = _m.createClass(
    eval("Base, " + f"{','.join(MODEL_CONFIG['mixins'])}") if MODEL_CONFIG['mixins']
        else [eval("Base")]
)
__all__ += ["ModelLM"]
# except Exception as err:
#     print(err)
#     logger.info(err)
#     raise ImportError(f"Cannot import module: {MODEL_CONFIG['template_name']}")
