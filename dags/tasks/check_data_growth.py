import pandas as pd
from airflow.models import Variable
from airflow.utils.dates import days_ago

DATA_PATH = '/opt/airflow/data/processed/data.csv'
THRESHOLD = 4  # Nombre de nouvelles lignes avant déclenchement

def check_data_growth(**context):
    """
    Vérifie si `data.csv` a grandi de 15 lignes depuis la dernière exécution.
    Stocke la taille du fichier dans une Variable Airflow.
    """
    ti = context['ti']

    # Récupérer la dernière taille stockée
    previous_size = Variable.get("data_size", default_var=0)  # Airflow Variables

    # Lire le fichier CSV
    try:
        df = pd.read_csv(DATA_PATH)
        new_size = len(df)
    except Exception as e:
        context['task_instance'].log.error(f"Erreur lors de la lecture du fichier : {e}")
        return "skip_training"
    interval = new_size - int(previous_size)
    print(f"previous size {previous_size}")
    print(f"new size {new_size}")
    print(f"interval{interval}")
    # Vérifier si le seuil est atteint
    if new_size - int(previous_size) >= THRESHOLD:
        # Stocker la nouvelle taille
        Variable.set("data_size", new_size)

        return "train_model"

    return "skip_training"

