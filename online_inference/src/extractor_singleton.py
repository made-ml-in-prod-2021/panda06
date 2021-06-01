import os
import logging

from ml_project.features import FeatureExtractor
logger = logging.getLogger(__name__)


class FeatureExtractorSingleton:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            extractor_path = os.getenv("PATH_TO_EXTRACTOR")
            if extractor_path is None:
                err = f"PATH_TO_EXTRACTOR {extractor_path} is None"
                logger.error(err)
                raise RuntimeError(err)
            cls.instance = FeatureExtractor.deserialize(extractor_path)
        return cls.instance
