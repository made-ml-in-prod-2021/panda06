import logging
from typing import List, Union

import click
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

from extractor_singleton import FeatureExtractorSingleton
from model_singleton import ModelSingleton

app = FastAPI()
logger = logging.getLogger(__name__)


class HeartDisease(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=1, max_items=80)]
    features: List[str]


class HeartDiseaseResponse(BaseModel):
    id: str
    heart_disease: int


@app.get("/")
def root():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    logger.info("Load model")
    ModelSingleton()
    logger.info("Model loaded")
    logger.info("Load extractor")
    FeatureExtractorSingleton()
    logger.info("Extractor loaded")


@app.get("/health")
def health() -> bool:
    logger.info("Health check")
    return hasattr(ModelSingleton, 'instance') and hasattr(FeatureExtractorSingleton, 'instance')


@app.get("/predict", response_model=List[HeartDiseaseResponse])
def predict(request: HeartDisease):
    logger.info("Start predict")
    model = ModelSingleton()
    extractor = FeatureExtractorSingleton()
    df = pd.DataFrame(request.data, columns=request.features)
    ids = df["Id"]
    data = df.drop(["Id"], axis=1)
    logger.info("Data transform")
    try:
        data = extractor.transform(data)
    except Exception as err:
        logger.error(f"Extractor error: {err}")
        raise HTTPException(status_code=500, detail="Extractor error")
    logger.info("Data transformed")
    logger.info("Model predict")
    try:
        predicts = model.predict(data)
    except Exception as err:
        logger.error(f"Model error: {err}")
        raise HTTPException(status_code=500, detail="Model error")
    logger.info("Model predicted")
    return [
        HeartDiseaseResponse(id=idx,
                             heart_disease=hd) for idx, hd in zip(ids,
                                                                  predicts)]


@click.command()
@click.option('-h', '--host', default='0.0.0.0')
@click.option('-p', '--port', default=8080)
def main(host, port):
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    main()
