import pandas as pd
import os
from fastapi import FastAPI

# Définition de l'application FastAPI
model_api = FastAPI()

@model_api.post("/split_data")
def train_model():
    command = "python /workspace/src/preprocessing/split.py"
    os.system(command)
    return {"message": "Splitting started successfully."}

@model_api.post("/train")
def train_model():
    command = "python /workspace/src/modeling/train_model.py"
    os.system(command)
    return {"message": "Training started successfully."}