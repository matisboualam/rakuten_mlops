import tensorflow as tf
import numpy as np
from dataloaders import ImagePreprocessor
import os

catalog = [
    "articles for newborns and babies",
    "children's games",
    "children's toys",
    "downloadable video games",
    "figurines",
    "figurines to paint and assemble",
    "food",
    "foreign literature",
    "gaming accessories",
    "garden accessories and decorations",
    "gardening accessories and tools",
    "historical literature",
    "home accessories and decorations",
    "home furnishings and decoration",
    "literature series",
    "model making",
    "non-fiction books",
    "nursery products",
    "outdoor accessories",
    "pet accessories",
    "pool accessories",
    "sets of gaming or video game accessories",
    "stationery",
    "textile accessories and decorations",
    "trading card games",
    "video game consoles",
    "video games"
]


class Model:
    def __init__(self, txt_model_weights=None, img_model_weights=None):
        if txt_model_weights is not None:
            self.txt_model = tf.keras.models.load_model(txt_model_weights)
        if img_model_weights is not None:
            self.img_model = tf.keras.models.load_model(img_model_weights)
        self.catalog = catalog
        self.train_data = '/workspace/data/processed/train_data.csv'
        self.validation_data = '/workspace/data/processed/val_data.csv'
        if os.path.exists(self.train_data) and os.path.exists(self.validation_data):
            self.img_preprocessor = ImagePreprocessor(self.train_data, self.validation_data)
        
    def preprocess_text(self, text):
        pass

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
    
    def train(self):
        train_data_generator = self.img_preprocessor.get_train_generator()
        val_data_generator = self.img_preprocessor.get_val_generator()
        batch = self.img_preprocessor.batch_size
        steps_per_epoch = train_data_generator.samples // batch
        validation_steps = val_data_generator.samples // batch
        
        # Check if GPU is available and set the device accordingly
        if tf.config.list_physical_devices('GPU'):
            device = '/GPU:0'
        else:
            device = '/CPU:0'
        
        with tf.device(device):
            self.img_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.img_model.fit(train_data_generator, 
                               steps_per_epoch=steps_per_epoch, 
                               validation_data=val_data_generator,
                               validation_steps=validation_steps,
                               epochs=1)