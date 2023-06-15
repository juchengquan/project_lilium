
from ..model_configuration import MODEL_CONFIG

print(MODEL_CONFIG)

# TODO: conbine this logic
try:
    if MODEL_CONFIG.get("type", "text_generation") == "text_generation":
        from .app_llm import app
    elif MODEL_CONFIG.get("type") == "embedding":
        from .app_st import app
    else:
        from .app_dummy import app
except (ImportError, Exception) as e:
    print(e)
    raise(e)

__all__ = [
    "app"
]