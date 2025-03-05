from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
import numpy as np
from src.modeling.dataloaders import ImagePreprocessor
import pandas as pd
import os
import json
import mlflow
import tensorflow as tf

class Model:
    def __init__(
            self, 
            txt_model_weights=None, 
            img_model_weights=None,
            mlflow_img_model_weights=None,
            train_data_path=None, 
            val_data_path=None,
            ):
        if txt_model_weights is not None:
            self.txt_model = tf.keras.models.load_model(txt_model_weights)
        if mlflow_img_model_weights is not None:
            self.img_model = mlflow.keras.load_model(img_model_weights)
        elif img_model_weights is not None:
            self.img_model = tf.keras.models.load_model(img_model_weights)
        with open('/workspace/models/catalog.json', 'r') as f:
            self.catalog = json.load(f)
        self.train_data = train_data_path
        self.val_data = val_data_path
        self.img_dataloader = ImagePreprocessor()

    def preprocess_image(self, image_path, target_size=(224, 224)):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_img(self, image_path):
        image_input = self.preprocess_image(image_path)
        pred_im = self.img_model.predict(image_input)
        predicted_class_index_im = np.argmax(pred_im[0])
        return self.catalog[predicted_class_index_im]