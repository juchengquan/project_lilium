import torch
from ...logging import logger
from ...model_configuration import MODEL_CONFIG

from ...utils.datatypes import ConfigDict

def get_first_device(_print: bool = True):
    if torch.cuda.is_available():
        _DEVICE = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     _DEVICE = "mps"
    else:
        _DEVICE = "cpu" 
    
    if _print:
        logger.info(f"First device is {_DEVICE}.")
    return _DEVICE

def load_config_from_yaml(ele):
    return ConfigDict(
        MODEL_CONFIG.get(ele, {})
    )
