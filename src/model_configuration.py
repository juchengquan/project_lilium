import os
import yaml
import pathlib
from .logging import logger

def get_model_config():
    try:
        if not os.environ["MODEL_CONFIG_FILE"]:
            cwd = pathlib.Path(__file__).parent.parent.resolve()

            model_config_file = str(cwd) + "/configs.yaml"
        else:
            model_config_file = os.environ["MODEL_CONFIG_FILE"]

        with open(model_config_file, "r") as f:
            _model_config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Reading model setting: " + model_config_file)
        
        if not _model_config.get("mixins", None):
            _model_config["mixins"] = []
        for ele in [
            "generation_config", "encode_config", "decode_config", "stream_config"
        ]:
            if not _model_config.get(ele, None):
                _model_config[ele] = {}
            
            
        return _model_config
    except Exception as err:
        raise(err)
    
MODEL_CONFIG = get_model_config()
