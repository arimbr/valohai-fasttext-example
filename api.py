from typing import List, Dict

import fasttext
from fastapi import FastAPI
from pydantic import BaseModel

from models.classification.commands import process_text, format_label

MODEL_PATH = 'model.bin'


class InputModel(BaseModel):
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
def predict(input: InputModel, k: int = 10):
    # Preprocess data
    print(f'input: {input}')
    data = process_text(input.text)

    # Get predictions
    labels, probas = model.predict(data, k=k)

    # Format predictions
    predictions = [{
        'label': format_label(label),
        'probability': proba
    } for label, proba in zip(labels, probas)]

    # Log input and output
    print(f'{{input: {input.text}, predictions: {predictions}}}')

    return {'predictions': predictions}
