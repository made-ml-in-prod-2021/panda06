import os
import logging

from ml_project.models import Model

logger = logging.getLogger(__name__)


class ModelSingleton:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            model_path = os.getenv("PATH_TO_MODEL")
            if model_path is None:
                err = f"PATH_TO_MODEL {model_path} is None"
                logger.error(err)
                raise RuntimeError(err)
            cls.instance = Model.deserialize(model_path)

        return cls.instance
