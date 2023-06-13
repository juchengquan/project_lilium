import os
import importlib
from .model_configuration import MODEL_CONFIG
from ..utils.logging import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid deadlock warning

__all__ = []
try:
    exec("from .base import BaseLM")
    exec(f"from ._mixins import {','.join(MODEL_CONFIG['mixins'])}")

    _m = importlib.import_module("." + MODEL_CONFIG["template_name"], __name__)
    HuggingFaceLM = _m.createClass(
        eval("BaseLM, " + f"{','.join(MODEL_CONFIG['mixins'])}")
    )()
    __all__ += ["HuggingFaceLM"]
except Exception as err:
    print(err)
    logger.info(err)
    raise ImportError(f"Cannot import module: {MODEL_CONFIG['template_name']}")
