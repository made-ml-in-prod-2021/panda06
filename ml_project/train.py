import json
import logging
import sys

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from ml_project.data import split_train_val_data, read_data
from ml_project.features import FeatureExtractor
from ml_project.models import Model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(config_path="configs", config_name="config")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.model)
    model = Model(model)
    data = read_data(to_absolute_path(cfg.general.data_path))
    logger.info(f"Data size is {data.shape}")
    train_data, test_data = split_train_val_data(data, cfg.splitting_params)
    logger.info(f"Train data size is {train_data.shape[0]}")
    logger.info(f"Val data size is {test_data.shape[0]}")

    feature_extractor = FeatureExtractor(cfg.feature_params)
    logger.info("Created feature extractor")

    y_train = feature_extractor.extract_target(train_data)
    train_data = feature_extractor.drop_target(train_data)
    X_train = feature_extractor.fit_transform(train_data)
    logger.info("Created train dataset")

    y_test = feature_extractor.extract_target(test_data)
    test_data = feature_extractor.drop_target(test_data)
    X_test = feature_extractor.transform(test_data)
    logger.info("Created val dataset")

    model.train_model(X_train, y_train)
    predicts = model.predict(X_test)
    metrics = Model.evaluate_model(predicts, y_test)
    logger.info(f"Metrics is {metrics}")

    model.serialize(cfg.general.model_output_path)
    feature_extractor.serialize(cfg.general.extractor_output_path)

    with open(cfg.general.metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    train_model()
