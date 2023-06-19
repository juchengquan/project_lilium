import pathlib
import logging 

curr_path = pathlib.Path(__file__).parent.resolve()
logging.config.fileConfig(str(curr_path) + "/logging_config.ini")

logger = logging.getLogger("project_lilium")

__all__ = [
    "logger"
]