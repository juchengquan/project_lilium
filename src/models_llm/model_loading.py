import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.logging import logger
from .model_configuration import MODEL_CONFIG

def load_model():
    try:
        if torch.cuda.is_available():
            _DEVICE = "cuda:0"
        # elif torch.backends.mps.is_available():
        #     _DEVICE = "mps"
        else:
            _DEVICE = "cpu"        
        logger.info(f'Device is {_DEVICE}')
        
        _model_config = MODEL_CONFIG["model"]
        
        _model = AutoModelForCausalLM.from_pretrained(
            _model_config["path"],
            load_in_8bit=_model_config["load_in_8bit"],
            torch_dtype=eval(_model_config["torch_dtype"]),
            device_map=_model_config["device_map"],
        )
        
        model_name = MODEL_CONFIG["name"]
        logger.info(f"{model_name} model loaded ready")
    except Exception as e:
        m = f"{model_name} model loaded failed. Exception: {e}"
        logger.error(m)
        raise Exception(m)

    return _model, _DEVICE

def load_tokenizer():
    """ Load tokenizer
    """
    try:
        model_name = MODEL_CONFIG["name"]
        _tokenizer_config = MODEL_CONFIG["tokenizer"]
        
        _tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=_tokenizer_config["path"], 
            use_fast=True,
            padding_side=_tokenizer_config["padding_side"],
            truncation_side=_tokenizer_config["truncation_side"],
        )
        
        if _tokenizer_config["set_pad_token"] == "bos":
            _tokenizer.pad_token = _tokenizer.bos_token
        elif _tokenizer_config["set_pad_token"] == "eos":
            _tokenizer.pad_token = _tokenizer.eos_token
            
    except Exception as err:
        m = "{model_name} tokenizer load failed. Exception: {err}".format(model_name=model_name, err=err)
        logger.error(m)
        raise Exception(m)

    return _tokenizer

def load_all_configs():
    # TODO
    return ( MODEL_CONFIG["generation_config"], 
        MODEL_CONFIG.get("encode_config", {}),
        MODEL_CONFIG.get("decode_config", {}) )
    