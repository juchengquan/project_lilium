try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as err:
    raise(err) 

from ...utils.logging import logger
from ...utils.funcs import load_config_from_yaml, get_first_device
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
    
    return ABCLM

def _load_model():
    try:
        _model_config = MODEL_CONFIG["tokenizer_config"]
        # modify the default name and type
        
        _model = SentenceTransformer(
            model_name_or_path=_model_config["model_name_or_path"],
            device=get_first_device(_print=False),
        )
        
        model_name = MODEL_CONFIG["name"]
        logger.info(f"{model_name} model loaded ready")
    except Exception as e:
        model_name = MODEL_CONFIG["name"]
        m = f"{model_name} model loaded failed. Exception: {e}"
        logger.error(m)
        raise Exception(m)

    return _model

__name__ = "sentence_transformer"