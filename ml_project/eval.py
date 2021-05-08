import logging
import sys

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore

from ml_project.data import read_data
from ml_project.features import FeatureExtractor
from ml_project.entities import EvalParams
from ml_project.models import Model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(config_path="configs", config_name="eval_config")
def eval_model(cfg: EvalParams):
    model = Model.deserialize(cfg.model_path)
    logger.info(f"Loaded model")

    feature_extractor = FeatureExtractor.deserialize(cfg.extractor_path)
    logger.info(f"Loaded feature extractor")

    data = read_data(cfg.data_path)
    logger.info(f"Data size is {data.shape[0]}")
    X = feature_extractor.transform(data)
    logger.info(f"Created dataset")

    predicts = model.predict(X)
    logger.info("Predicted")

    df_predicts = pd.DataFrame(predicts)
    df_predicts.to_csv(cfg.output_data_path, header=False, index=False)
    logger.info(f"Predications saved to {cfg.output_data_path}")


if __name__ == '__main__':
    eval_model()
