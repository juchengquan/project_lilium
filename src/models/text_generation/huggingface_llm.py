try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as err:
    raise(err) 

from ...utils.logging import logger
from ...utils.funcs import load_config_from_yaml
from ...model_configuration import MODEL_CONFIG

def createClass(cls_list):
    class ABCLM(*cls_list):
        def __init__(self):
            super().__init__()
            for ele in [
                "generation_config", "encode_config", "decode_config", "stream_config"
            ]:
                self.__setattr__(ele, load_config_from_yaml(ele))
            
            self._model = _load_model()
            self._tokenizer = _load_tokenizer()
        
    return ABCLM

def _load_model():
    try:
        _model_config = MODEL_CONFIG["tokenizer_config"]
        
        # modify the default name and type
        import torch # need to keep for the following line
        _model_config["torch_dtype"] = eval(_model_config["torch_dtype"])
        
        _model = AutoModelForCausalLM.from_pretrained(
            **_model_config,
        )
        
        model_name = MODEL_CONFIG["name"]
        logger.info(f"Model loaded ready: {model_name}.")
    except Exception as e:
        model_name = MODEL_CONFIG["name"]
        m = f"Model loaded failed: {model_name}; Exception: {e}"
        logger.error(m)
        raise Exception(m)

    return _model

def _load_tokenizer():
    """ Load tokenizer
    """
    try:
        model_name = MODEL_CONFIG["name"]
        _tokenizer_config = MODEL_CONFIG["tokenizer"]
        
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

__name__ = "huggingface_llm"