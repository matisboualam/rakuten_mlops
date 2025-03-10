import pandas as pd
import os
from fastapi import FastAPI

from src.modeling.models import Model

# Définition de l'application FastAPI
model_api = FastAPI()

model_weights = "models:/ImageClassificationModel/1"

print(f"Loading model ...")
model = Model(
    img_model_weights=model_weights
    )
print("Model loaded successfully.")

@model_api.post("/split_data")
def train_model():
    command = "python /workspace/src/preprocessing/split.py"
    os.system(command)
    return {"message": "Splitting started successfully."}

@model_api.post("/train")
def train_model():
    command = "python /workspace/src/modeling/train.py"
    os.system(command)
    return {"message": "Training started successfully."}

@model_api.post("/evaluate")
def evaluate_model():
    command = "python /workspace/src/modeling/evaluate.py"
    os.system(command)
    return {"message": "Evaluation started successfully."}