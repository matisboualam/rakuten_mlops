import tensorflow as tf
import numpy as np
from dataloaders import ImagePreprocessor
from dataloaders import TextPreprocessor
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
from keras.models import Model
from keras.layers import Input, Dense, GRU, Bidirectional, Dropout, BatchNormalization, Attention, Reshape, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import gensim.downloader as api
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import unicodedata


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

# Charger le modèle d'embeddings pré-entraîné FastText
word2vec_model = api.load('fasttext-wiki-news-subwords-300')

class PreprocessText:
    def __init__(self):
        """
        Constructeur de la classe.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('french'))
        self.catalog = catalog

    def remove_isolated_letter_apostrophe(self, text):
        """Retirer les lettres isolées suivies d'une apostrophe."""
        return re.sub(r"\b[a-zA-Z]'", "", text)

    def remove_malencoded_chars(self, text):
        """Nettoyer les caractères mal encodés (ex. : â, â) et tous autres caractères non souhaités."""
        
        # 1. Normalisation des caractères Unicode (pour résoudre les problèmes d'encodage)
        text = unicodedata.normalize('NFKD', text)
        
        # 2. Remplacer les caractères non-ASCII par des espaces
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # 3. Nettoyage des caractères non imprimables (ex. : caractères invisibles)
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

    def preprocess_text_for_dataset(self, df, column, image_path=None):
        """Appliquer toutes les étapes de nettoyage sur un texte ou une colonne entière du DataFrame."""
        if image_path:
            # Filtrer le DataFrame pour trouver la ligne correspondant à l'image
            row = df[df["image_path"] == image_path]
            if row.empty:
                raise ValueError(f"Aucune correspondance trouvée pour {image_path}.")
            text = row[column].values[0]  # Récupérer la description associée
            text = self.clean_text(text)  # Appliquer le nettoyage
            return text
        
        # Si image_path n'est pas fourni, traiter toute la colonne
        df[column] = df[column].apply(self.clean_text)
        return df

    def clean_text(self, text):
        """Nettoyage du texte."""
        text = text.lower()  # Conversion en minuscules
        text = self.remove_isolated_letter_apostrophe(text)  # Retirer les lettres isolées suivies d'une apostrophe
        text = self.remove_malencoded_chars(text)  # Nettoyer les caractères mal encodés
        text = self.clean_html(text)  # Enlever les balises HTML
        text = self.remove_punctuation(text)  # Retirer la ponctuation
        text = self.tokenize_and_lemmatize(text)  # Tokenisation et lemmatisation
        return text


    
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
            self.txt_preprocessor = TextPreprocessor(self.train_data, self.validation_data)
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

    def predict_text(self, df, image_path, text_model_weights,model):
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
        self.txt_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        # Entraîner le modèle
        data_train = pd.read_csv(self.train_data)
        X_train_split = self.create_X(data_train)

        print(f"X_train_split vision {X_train_split}")

        X_val_split = self.create_X(self.txt_preprocessor.val_df)
        Y_train_split = self.create_Y(self.txt_preprocessor.train_df)
        Y_val_split = self.create_Y(self.txt_preprocessor.val_df)

        print(f"Shape of X_train_split: {X_train_split.shape}")
        print(f"Shape of Y_train_split: {Y_train_split.shape}")

        print(f"Shape of X_val_split: {X_val_split.shape}")
        print(f"Shape of Y_val_split: {Y_val_split.shape}")

        history = self.txt_model.fit(X_train_split, Y_train_split, validation_data=(X_val_split, Y_val_split), epochs=1, batch_size=32)
        print("rentrainment worked")
        
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
DATA_PATH = "data/processed/train_data.csv"
df = pd.read_csv(DATA_PATH)

# Création de l'instance de prétraitement
model = Model(TEXT_MODEL_PATH)

# Demande de l'image à traiter (exemple : chemin d'image existant dans df['image_path'])
image_path_input = "/workspace/data/raw/img/image_1193221506_product_3139415323.jpg"  # Remplacez par un chemin réel présent dans df['image_path']

# Vérification si l'image existe dans le dataset
if image_path_input not in df["image_path"].values:
    raise ValueError(f"L'image '{image_path_input}' n'existe pas dans le dataset.")

# Prédiction avec le modèle de texte
#predicted_category, cleaned_text = model.predict_text(df, image_path_input, TEXT_MODEL_PATH,word2vec_model)
model.train_text()
# Affichage des résultats
#print(f" Texte nettoyé : {cleaned_text}")
#print(f" Catégorie prédite : {predicted_category}")