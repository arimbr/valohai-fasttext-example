import json
import logging
import uuid
from typing import List, Dict

import fasttext
from fastapi import FastAPI
from pydantic import BaseModel

from models.classification.commands import process_text, format_label

MODEL_PATH = 'model.bin'

logger = logging.getLogger(__name__)


class FeaturesModel(BaseModel):
    text: str


class PredictionModel(BaseModel):
    label: str
    probability: float


class PredictionsModel(BaseModel):
    predictions: List[PredictionModel]


# Initialize model
model = fasttext.load_model(MODEL_PATH)

# Initialize API
app = FastAPI()


@app.get(".*/predict")
def hello():
    return 'OK'


@app.post(".*/predict", response_model=PredictionsModel)
def predict(features: FeaturesModel, k: int = 10, decimals: int = 2):
    request_id = uuid.uuid4().hex

    # Log features
    logger.info(json.dumps(
        {'request_id': request_id, 'features': features.dict()}))

    # Preprocess data
    data = process_text(features.text)

    # Get predictions
    labels, probas = model.predict(data, k=k)

    # Format predictions
    predictions = [{
        'label': format_label(label),
        'probability': round(proba, decimals)
    } for label, proba in zip(labels, probas)]

    # Log predictions
    logger.info(json.dumps(
        {'request_id': request_id, 'predictions': predictions}))

    return {'predictions': predictions}
