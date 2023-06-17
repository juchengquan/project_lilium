from sentence_transformers import SentenceTransformer

from ...utils.logging import logger
from ...utils.funcs import load_all_configs, get_first_device
from ...model_configuration import MODEL_CONFIG

def createClass(cls_list):
    class ABCLM(*cls_list):
        def __init__(self):
            super().__init__()
            self._model = _load_model()
            self._generation_config, self._encode_config, self._decode_config, self._stream_config = load_all_configs()
    
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

    