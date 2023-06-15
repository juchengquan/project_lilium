from typing import Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.logging import logger
from ...utils.funcs import load_all_configs
from ...model_configuration import MODEL_CONFIG

def createClass(cls_list):
    class ABCLM(*cls_list):
        def __init__(self):
            super().__init__()
            self._model = _load_model()
            self._tokenizer = _load_tokenizer()
            self._generation_config, self._encode_config, self._decode_config, self._stream_config = load_all_configs()

        def generate_response_stream(self,
            input_texts: Union[List[str], str] = "",
        ) -> Union[List[str], str]:
            output_texts = self.generate_stream(
                input_texts=input_texts, 
                generation_config=self.generation_config,
                stream_config=self.stream_config,
            )

            return output_texts
        
    return ABCLM

def _load_model():
    try:
        _model_config = MODEL_CONFIG["model"]
        
        # modify the default name and type
        exec("import torch")
        _model_config["torch_dtype"] = eval(_model_config["torch_dtype"])
        _model_config["pretrained_model_name_or_path"] = _model_config["path"]
        _model_config.pop("path")
        
        _model = AutoModelForCausalLM.from_pretrained(
            **_model_config,
        )
        
        model_name = MODEL_CONFIG["name"]
        logger.info(f"{model_name} model loaded ready")
    except Exception as e:
        model_name = MODEL_CONFIG["name"]
        m = f"{model_name} model loaded failed. Exception: {e}"
        logger.error(m)
        raise Exception(m)

    return _model

def _load_tokenizer():
    """ Load tokenizer
    """
    try:
        model_name = MODEL_CONFIG["name"]
        _tokenizer_config = MODEL_CONFIG["tokenizer"]
        
        _tokenizer_config["pretrained_model_name_or_path"] = _tokenizer_config["path"]
        _tokenizer_config.pop("path")
        
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
