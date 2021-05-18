import logging

import click
import uvicorn

from typing import Optional

from fastapi import FastAPI
from sklearn.pipeline import Pipeline

app = FastAPI()
model: Optional[Pipeline] = None
logger = logging.getLogger(__name__)


@app.get("/")
def root():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    pass


@app.get("/health")
def health() -> bool:
    return model is not None


@app.post("/predict")
def predict():
    pass


@click.command()
@click.option('-h', '--host', default='0.0.0.0')
@click.option('-p', '--port', default=8080)
def main(host, port):
    uvicorn.run("app:app", host=host, port=port)


if __name__ == "__main__":
    main()
