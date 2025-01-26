import os
import logging
import pandas as pd
from PIL import Image
from collections import Counter

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/check_data.log"),
        logging.StreamHandler()
    ]
)

def check_data(x_data_path, y_data_path, images_dir):
    """
    Vérifie la cohérence des données entre x_data.csv, y_data.csv et les images.
    :param x_data_path: Chemin vers x_data.csv
    :param y_data_path: Chemin vers y_data.csv
    :param images_dir: Chemin vers le dossier contenant les images
    """
    errors = []

    # Chargement des fichiers CSV
    try:
        x_data = pd.read_csv(x_data_path)
        y_data = pd.read_csv(y_data_path)
        logging.info("Chargement des fichiers CSV réussi.")
    except Exception as e:
        errors.append(f"Erreur lors du chargement des fichiers CSV : {e}")
        logging.error(f"Erreur lors du chargement des fichiers CSV : {e}")
        return errors

    # Vérification des colonnes obligatoires
    required_x_columns = {"productid", "imageid", "description", "designation"}
    required_y_columns = {"prdtypecode"}

    if not required_x_columns.issubset(x_data.columns):
        missing_columns = required_x_columns - set(x_data.columns)
        errors.append(f"Colonnes manquantes dans x_data.csv : {missing_columns}")
        logging.error(f"Colonnes manquantes dans x_data.csv : {missing_columns}")

    if not required_y_columns.issubset(y_data.columns):
        missing_columns = required_y_columns - set(y_data.columns)
        errors.append(f"Colonnes manquantes dans y_data.csv : {missing_columns}")
        logging.error(f"Colonnes manquantes dans y_data.csv : {missing_columns}")

    # Vérification des valeurs manquantes
    if x_data.isnull().any().any():
        errors.append("Valeurs manquantes détectées dans x_data.csv")
        logging.warning("Valeurs manquantes détectées dans x_data.csv")

    if y_data.isnull().any().any():
        errors.append("Valeurs manquantes détectées dans y_data.csv")
        logging.warning("Valeurs manquantes détectées dans y_data.csv")

    # Vérification des doublons
    if x_data.duplicated(subset=["productid", "imageid"]).any():
        errors.append("Doublons détectés dans x_data.csv (productid, imageid)")
        logging.warning("Doublons détectés dans x_data.csv (productid, imageid)")

    # Vérification de la correspondance entre productid dans x_data et y_data
    x_productids = len(x_data)
    y_productids = len(y_data)

    if missing_in_y := x_productids - y_productids:
        errors.append(f"productid présents dans x_data.csv mais absents dans y_data.csv : {len(missing_in_y)}")
        logging.warning(f"productid présents dans x_data.csv mais absents dans y_data.csv : {len(missing_in_y)}")

    if missing_in_x := y_productids - x_productids:
        errors.append(f"productid présents dans y_data.csv mais absents dans x_data.csv : {len(missing_in_x)}")
        logging.warning(f"productid présents dans y_data.csv mais absents dans x_data.csv : {len(missing_in_x)}")

    # Vérification de la correspondance avec les images
    missing_images = []
    unused_images = []
    corrupted_images = []

    for _, row in x_data.iterrows():
        image_name = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            missing_images.append(image_name)
        else:
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Vérifie que l'image n'est pas corrompue
            except Exception:
                corrupted_images.append(image_name)

    # Vérification des images non utilisées
    all_images = {f for f in os.listdir(images_dir) if f.endswith(".jpg")}
    used_images = {f"image_{row['imageid']}_product_{row['productid']}.jpg" for _, row in x_data.iterrows()}

    unused_images = all_images - used_images

    # Rapports sur les images
    if missing_images:
        errors.append(f"Images manquantes : {len(missing_images)}")
        logging.warning(f"Images manquantes : {len(missing_images)}")

    if unused_images:
        errors.append(f"Images inutilisées : {len(unused_images)}")
        logging.info(f"Images inutilisées : {len(unused_images)}")

    if corrupted_images:
        errors.append(f"Images corrompues : {len(corrupted_images)}")
        logging.error(f"Images corrompues : {len(corrupted_images)}")

    # Statistiques supplémentaires
    code_distribution = Counter(y_data["prdtypecode"])
    if code_distribution:
        logging.info("Distribution des codes produits (top 5) :")
        for code, count in code_distribution.most_common(5):
            logging.info(f"  {code}: {count}")

    # Résultat final
    if not errors:
        logging.info("Toutes les vérifications sont passées avec succès !")
    else:
        logging.error("Erreurs détectées :")
        for error in errors:
            logging.error(f"- {error}")

    return errors

# Exemple d'utilisation
if __name__ == "__main__":
    x_data_path = "data/raw/x_data.csv"
    y_data_path = "data/raw/y_data.csv"
    images_dir = "data/raw/img"
    dir = "raw"

    logging.info(f"Vérification des données issues du dossier {dir}...")

    errors = check_data(x_data_path, y_data_path, images_dir)
    if errors:
        logging.error("\nRésumé des erreurs :")
        for error in errors:
            logging.error(f"- {error}")
