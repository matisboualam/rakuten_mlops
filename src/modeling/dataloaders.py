import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImagePreprocessor:
    def __init__(self, train_csv, val_csv, image_size=(224, 224), batch_size=16):
        """
        Initialize the ImagePreprocessor class.
        
        :param train_csv: Path to the CSV file for training data.
        :param val_csv: Path to the CSV file for validation data.
        :param image_size: The target size for resizing images.
        :param batch_size: The number of images per batch for data generation.
        """
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.image_size = image_size
        self.batch_size = batch_size

        # Load the CSV files
        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)

        # Create an ImageDataGenerator object for rescaling and augmentations
        self.train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    def get_train_generator(self):
        """
        Creates and returns the data generator for training data.
        
        :return: A Keras data generator for training data.
        """
        return self.train_datagen.flow_from_dataframe(
            dataframe=self.train_df,
            x_col="image_path",  # Column containing the image file paths
            y_col="prdtypecode", # Column containing the labels
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True  # Shuffle training data
        )

    def get_val_generator(self):
        """
        Creates and returns the data generator for validation data.
        
        :return: A Keras data generator for validation data.
        """
        return self.val_datagen.flow_from_dataframe(
            dataframe=self.val_df,
            x_col="image_path",  # Column containing the image file paths
            y_col="prdtypecode", # Column containing the labels
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False  # Do not shuffle validation data
        )

class TextPreprocessor:
    pass


# Example usage
if __name__ == "__main__":
    # Paths to the training and validation CSV files
    train_csv = "data/processed/train_data.csv"
    val_csv = "data/processed/val_data.csv"
    
    # Parameters
    image_size = (224, 224)
    batch_size = 64

    # Initialize the ImagePreprocessor
    preprocessor = ImagePreprocessor(train_csv, val_csv, image_size=image_size, batch_size=batch_size)

    # Create training and validation generators
    train_generator = preprocessor.get_train_generator()
    val_generator = preprocessor.get_val_generator()

    # Print to verify the generators
    print(f"Training generator: {train_generator.samples} samples")
    print(f"Validation generator: {val_generator.samples} samples")
