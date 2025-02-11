import tensorflow as tf
import numpy as np
from dataloaders import ImagePreprocessor
import pandas as pd
import os
import json


class Model:
    def __init__(
            self, 
            txt_model_weights=None, 
            img_model_weights=None, 
            train_data_path=None, 
            val_data_path=None,
            ):
        if txt_model_weights is not None:
            self.txt_model = tf.keras.models.load_model(txt_model_weights)
        if img_model_weights is not None:
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
    
    # def train_img_model(self):
    #     if not os.path.exists(self.train_data):
    #         print(f"Error: File not found at {self.train_data}")
    #         return
    #     if not os.path.exists(self.val_data):
    #         print(f"Error: File not found at {self.val_data}")
    #         return

    #     train_data_preprocessor = ImagePreprocessor(self.train_data, image_size=(224, 224), batch_size=16)
    #     val_data_preprocessor = ImagePreprocessor(self.val_data, image_size=(224, 224), batch_size=16)
    #     batch = self.img_preprocessor.batch_size
    #     steps_per_epoch = train_data_preprocessor.samples // batch
    #     validation_steps = val_data_preprocessor.samples // batch
        
    #     # Check if GPU is available and set the device accordingly
    #     if tf.config.list_physical_devices('GPU'):
    #         device = '/GPU:0'
    #     else:
    #         device = '/CPU:0'
        
    #     with tf.device(device):
    #         self.img_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #         self.img_model.fit(train_data_generator, 
    #                            steps_per_epoch=steps_per_epoch, 
    #                            validation_data=val_data_generator,
    #                            validation_steps=validation_steps,
    #                            epochs=1)