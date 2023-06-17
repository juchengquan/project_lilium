import torch
import hashlib
from .logging import logger
from ..model_configuration import MODEL_CONFIG

def gen_sha():
    return hashlib.sha1(b"Nobody inspects the spammish repetition").hexdigest()

def get_first_device(_print: bool = True):
    if torch.cuda.is_available():
        _DEVICE = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     _DEVICE = "mps"
    else:
        _DEVICE = "cpu" 
    
    if _print:
        logger.info(f"Device is {_DEVICE}")
    return _DEVICE

def load_all_configs():
    # TODO
    return ( MODEL_CONFIG["generation_config"], 
        MODEL_CONFIG.get("encode_config", {}),
        MODEL_CONFIG.get("decode_config", {}),
        MODEL_CONFIG.get("stream_config", {}))