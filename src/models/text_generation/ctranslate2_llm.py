try:
    import ctranslate2 
    from transformers import AutoTokenizer
except ModuleNotFoundError as err:
    raise(err) 

from ...logging import logger
from ...model_configuration import MODEL_CONFIG

from ..utils import get_first_device

def load_model():
    try:
        _model_config = MODEL_CONFIG["model_config"]
        
        # modify the default name and type
        import torch # need to keep for the following line
        # _model_config["torch_dtype"] = eval(_model_config["torch_dtype"])
        
        _model = ctranslate2.Generator(
            model_path=_model_config["pretrained_model_name_or_path"],
            device=get_first_device(_print=False),
            compute_type="auto", # TODO
        )
        
        model_name = MODEL_CONFIG["name"]
        logger.info(f"Model loaded ready: {model_name}.")
    except Exception as e:
        model_name = MODEL_CONFIG["name"]
        m = f"Model loaded failed: {model_name}; Exception: {e}"
        logger.error(m)
        raise Exception(m)

    return _model

def load_tokenizer():
    """ Load tokenizer
    """
    try:
        model_name = MODEL_CONFIG["name"]
        _tokenizer_config = MODEL_CONFIG["tokenizer_config"]
        
        _tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=_tokenizer_config["pretrained_model_name_or_path"], 
            use_fast=_tokenizer_config.get("use_fast", True),
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

# __name__ = "huggingface_llm"