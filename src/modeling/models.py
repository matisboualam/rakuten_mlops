import tensorflow as tf
import numpy as np
from modeling.dataloaders import ImagePreprocessor
from modeling.dataloaders import TextPreprocessor
import os
# Imports nécessaires
import tensorflow as tf
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model  # Changement ici
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, Dropout, BatchNormalization, Attention, Reshape, Conv1D, MaxPooling1D  # Changement ici
from tensorflow.keras.utils import to_categorical  # Changement ici
from tensorflow.keras.optimizers import Adam  # Changement ici
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint  # Changement ici
import gensim.downloader as api
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # Changement ici
import unicodedata
import mlflow
import mlflow.keras
import json
from sklearn.metrics import classification_report



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


# Télécharger les ressources nécessaires pour NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

mlflow.set_tracking_uri("http://localhost:5000")

# Charger le modèle d'embeddings pré-entraîné FastText
word2vec_model = api.load('fasttext-wiki-news-subwords-300')
    
class CustomModel:
    def __init__(self, txt_model_weights=None, img_model_weights=None):
        if txt_model_weights is not None:
            self.txt_model = tf.keras.models.load_model(txt_model_weights)
        if img_model_weights is not None:
            self.img_model = tf.keras.models.load_model(img_model_weights)
        self.catalog = catalog
        self.train_data = '/workspace/data/processed/train.csv'
        self.validation_data = '/workspace/data/processed/val.csv'
        self.test_data = '/workspace/data/processed/test.csv'
        if os.path.exists(self.train_data) and os.path.exists(self.validation_data):
            self.img_preprocessor = ImagePreprocessor(self.train_data, self.validation_data,self.test_data)
            self.txt_preprocessor = TextPreprocessor(self.train_data, self.validation_data,self.test_data)
        self.model_vect = word2vec_model

    def get_mean_vector(self, model, text):
        words = text.split()
        word_vectors = []
        for word in words:
            try:
                word_vectors.append(model[word])  # Utiliser get_word_vector pour FastText
            except KeyError:
                continue  # Si le mot n'est pas trouvé, on l'ignore

        print(f"Texte: {text}")
        print(f"Nombre de mots trouvés dans le modèle: {len(word_vectors)}")

        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(300)  # Si aucun mot n'a été trouvé, retourner un vecteur de zéros

    def predict_text(self, df, image_path,model):
        """Prédire une classe à partir d'une image et récupérer sa description nettoyée."""
        text_input = self.txt_preprocessor.preprocess_text_for_dataset(df, "description", image_path=image_path)
        text_embedded_input = np.array([self.get_mean_vector(model, text) for text in text_input])
        
        # Chargement et prédiction avec le modèle d'image

        pred_txt = self.txt_model.predict(text_embedded_input)
        predicted_class_index_txt = np.argmax(pred_txt[0])
        
        return self.catalog[predicted_class_index_txt], text_input  # Renvoi de la classe prédite et du texte nettoyé


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
    
    def text_encoding(self,text_input):
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(text_input)
        Y_categorical = to_categorical(Y_encoded)
        return Y_categorical
    
    def create_X(self, df):
        print(f"Nombre d'échantillons dans df: {len(df)}")  # Vérifier la taille de df

        text_input = self.txt_preprocessor.preprocess_text_for_dataset(df, "description")
        text_input = text_input['description']
        print(f"Nombre de descriptions après préprocessing: {len(text_input)}")  # Vérifier que le texte est bien extrait

        text_embedded_input = np.array([self.get_mean_vector(self.model_vect, text) for text in text_input])
        print(f"Shape du tableau final: {text_embedded_input.shape}")  # Vérifier que la sortie correspond bien à la taille attendue

        return text_embedded_input

        
    def create_Y(self,df):
        text_input = self.txt_preprocessor.preprocess_text_for_dataset(df, "prdtypecode")
        text_input = text_input["prdtypecode"]
        print(text_input)
        text_categorical_input = self.text_encoding(text_input)
        return text_categorical_input
    
    def train_text(self):
        """Entraîne le modèle et enregistre les hyperparamètres, les métriques, et le rapport de classification dans MLflow."""
        
        # Charger les données
        data_train = pd.read_csv(self.train_data)
        X_train_split = self.create_X(data_train)
        X_val_split = self.create_X(self.txt_preprocessor.val_df)
        Y_train_split = self.create_Y(self.txt_preprocessor.train_df)
        Y_val_split = self.create_Y(self.txt_preprocessor.val_df)

        # Afficher les formes des données
        print(f"Shape of X_train_split: {X_train_split.shape}")
        print(f"Shape of Y_train_split: {Y_train_split.shape}")
        print(f"Shape of X_val_split: {X_val_split.shape}")
        print(f"Shape of Y_val_split: {Y_val_split.shape}")
        
        # Démarrer un nouveau run MLflow
        with mlflow.start_run():
            # Définir les hyperparamètres
            learning_rate = 0.001
            batch_size = 32
            epochs = 1
            
            # 🔹 Log des hyperparamètres
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("dataset_size", len(data_train))  # Log la taille du dataset

            # Compilation du modèle
            self.txt_model.compile(optimizer=Adam(learning_rate=learning_rate),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

            # Entraînement du modèle
            history = self.txt_model.fit(X_train_split, Y_train_split,
                                        validation_data=(X_val_split, Y_val_split),
                                        epochs=epochs, batch_size=batch_size)

            # Prédiction et évaluation
            X_test_split = self.create_X(self.txt_preprocessor.test_df)
            Y_test_split = self.create_Y(self.txt_preprocessor.test_df)
            Y_pred = self.txt_model.predict(X_test_split)
            Y_pred_classes = np.argmax(Y_pred, axis=1)
            Y_true_classes = np.argmax(Y_test_split, axis=1)

            # Rapport de classification
            class_report = classification_report(Y_true_classes, Y_pred_classes, output_dict=True)
            accuracy = class_report["accuracy"]
            class_report_json = json.dumps(class_report)


            # 🔹 Log des métriques (accuracy, loss)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("val_loss", history.history["val_loss"][-1])
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

            # 🔹 Log du modèle
            mlflow.keras.log_model(self.txt_model, "text_model")

            # 🔹 Log du rapport de classification en texte
            mlflow.log_text(class_report_json, "classification_report.json")

            print("Entraînement terminé et enregistré dans MLflow.")


    def train_img(self):
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

# Définition des cheinms des modèles
TEXT_MODEL_PATH = "models/textmodel_27_classes.keras"

# Chargement du DataFrame (exemple : fichier CSV)
DATA_PATH = "data/processed/train.csv"
df = pd.read_csv(DATA_PATH)

# Création de l'instance de prétraitement
model = CustomModel(TEXT_MODEL_PATH)

model.train_text()