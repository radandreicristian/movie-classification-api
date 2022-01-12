"""API endpoints."""
import json
import pathlib
from io import BytesIO

import nltk
import numpy as np
import pandas as pd
import torch
from fastapi.applications import FastAPI
from fastapi.param_functions import File
from fastapi.responses import PlainTextResponse

from .service.prediction_service import PredictionServiceFactory
from .service.training_service import TrainingServiceFactory

nltk.download('stopwords')

seed = 21
np.random.seed(seed)
torch.manual_seed(seed)
config_path = pathlib.Path(__file__).parent.absolute() / 'service' / 'service_config.json'
config = json.load(open(config_path))
training_service = TrainingServiceFactory().get_service(**config)
testing_service = PredictionServiceFactory().get_service(**config)
app = FastAPI()


@app.post("/genres/train")
async def train(file: bytes = File(...)) -> None:
    """Train a predictive model to rank movie genres based on their synopsis."""
    df = pd.read_csv(BytesIO(file))
    training_service.prepare_data(df)
    training_service.create_model()
    training_service.train_and_save_model()


@app.post("/genres/predict")
async def test(file: bytes = File(...)) -> PlainTextResponse:
    """Evaluate a previously trained model that predicts movie genres based on their synopsis."""
    df = pd.read_csv(BytesIO(file))
    testing_service.prepare_data(df)
    testing_service.load_model()
    testing_service.evaluate_model()
    testing_service.decode_predictions()
    csv = testing_service.make_csv()
    return PlainTextResponse(csv)
