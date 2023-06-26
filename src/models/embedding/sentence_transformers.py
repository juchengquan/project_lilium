try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as err:
    raise(err) 

from ...logging import logger
from ...model_configuration import MODEL_CONFIG

from ..utils import get_first_device

def load_model():
    try:
        _model_config = MODEL_CONFIG["model_config"]
        # modify the default name and type
        
        _model = SentenceTransformer(
            model_name_or_path=_model_config["model_name_or_path"],
            device=get_first_device(_print=False),
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
    return None

__name__ = "sentence_transformer"