import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImagePreprocessor:
    def __init__(self, image_size=(224, 224), batch_size=4):
        """
        Initialize the ImagePreprocessor class.
        
        :param csv: Path to the CSV file for data to preprocess.
        :param image_size: The target size for resizing images.
        :param batch_size: The number of images per batch for data generation.
        """

        self.image_size = image_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1.0 / 255)

    def _load_csv(self, csv_path):
        """Charge le CSV et vérifie la présence des colonnes nécessaires."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Le fichier {csv_path} est introuvable.")
        
        df = pd.read_csv(csv_path)
        required_columns = {"image_path", "prdtypecode"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Le fichier {csv_path} doit contenir les colonnes {required_columns}.")
        return df
    
    def get_generator(self, csv):
        """
        Creates and returns the data generator for training data.
        
        :return: A Keras data generator for training data.
        """
        df = self._load_csv(csv) if csv else None
        return self.datagen.flow_from_dataframe(
            dataframe=df,
            x_col="image_path",  # Column containing the image file paths
            y_col="prdtypecode", # Column containing the labels
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True  # Shuffle training data
        )

class TextPreprocessor:
    pass

