import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
import string
import unicodedata

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
    def __init__(self,train_csv,val_csv,test_csv):
        """
        Constructeur de la classe.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('french'))

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv

        # Load the CSV files
        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)
        self.test_df = pd.read_csv(test_csv)


    def remove_isolated_letter_apostrophe(self, text):
        """Retirer les lettres isolées suivies d'une apostrophe."""
        return re.sub(r"\b[a-zA-Z]'", "", text)

    def remove_malencoded_chars(self, text):
        """Nettoyer les caractères mal encodés et autres caractères non souhaités."""
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = ''.join([c for c in text if c.isprintable()])
        return text

    def clean_html(self, text):
        """Enlever les balises HTML."""
        return BeautifulSoup(text, "html.parser").get_text()

    def remove_punctuation(self, text):
        """Supprimer la ponctuation."""
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize_and_lemmatize(self, text):
        """Tokenisation et lemmatisation des mots, en retirant les mots vides (stopwords)."""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def clean_text(self, text):
        """Appliquer toutes les étapes de nettoyage sur un texte."""
        text = text.lower()
        text = self.remove_isolated_letter_apostrophe(text)
        text = self.remove_malencoded_chars(text)
        text = self.clean_html(text)
        text = self.remove_punctuation(text)
        text = self.tokenize_and_lemmatize(text)
        return text

    def preprocess_text_for_dataset(self, df, column, image_path=None):
        """Appliquer le nettoyage sur un texte ou une colonne entière du DataFrame.
        
        - Si `image_path` est spécifié, retourne le texte nettoyé correspondant à cette image.
        - Sinon, nettoie toute la colonne spécifiée dans le DataFrame.
        """
        if image_path:
            row = df[df["image_path"] == image_path]
            if row.empty:
                raise ValueError(f"Aucune correspondance trouvée pour {image_path}.")
            text = row[column].values[0]
            return self.clean_text(text)
        
        df[column] = df[column].astype(str).apply(self.clean_text)
        return df


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
