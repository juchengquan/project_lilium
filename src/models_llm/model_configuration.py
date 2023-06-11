import os
import yaml
import pathlib
from ..utils.logging import logger

if not os.environ["MODEL_CONFIG_FILE"]:
    cwd = pathlib.Path(__file__).parent.parent.resolve()

    model_config_file = str(cwd) + "/configs.yaml"
else:
    model_config_file = os.environ["MODEL_CONFIG_FILE"]

with open(model_config_file, "r") as f:
    MODEL_CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    logger.info("Reading model setting: " + model_config_file)
    
    
    