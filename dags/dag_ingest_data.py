from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
from airflow.models import Variable

# Configuration des arguments par défaut
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 2, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'dag_ingest_data',
    default_args=default_args,
    schedule_interval="*/30 * * * * *",  # Toutes les 30 secondes
    catchup=False
)

DATA_PATH_DATA = '/opt/airflow/data/processed/data.csv'
DATA_PATH_UNSEEN = '/opt/airflow/data/processed/unseen.csv'

def add_new_data(**context):
    """Ajoute une nouvelle ligne à data.csv et stocke la nouvelle taille du fichier."""
    # Configuration du logger
    logger = logging.getLogger('airflow.task')
    
    try:
        # Lire les deux fichiers CSV avec gestion des erreurs de formatage
        df = pd.read_csv(DATA_PATH_DATA, sep=",", quoting=1)  # quoting=1 pour gérer les guillemets
        df_unseen = pd.read_csv(DATA_PATH_UNSEEN, sep=",", quoting=1)
        logger.info(f"Colonnes de df : {df.columns}")
        logger.info(f"Colonnes de df_unseen : {df_unseen.columns}")

        if not df_unseen.empty:
            last_input = df_unseen.iloc[[-1]]  # Dernière ligne de unseen
            logger.info(f"Ajout de la ligne suivante à data :\n{last_input}")
            
            # Ajouter cette ligne à data.csv
            df = pd.concat([df, last_input], ignore_index=True)
            
            # Retirer la dernière ligne de unseen
            df_unseen = df_unseen.drop(df_unseen.index[-1])
            
            # Sauvegarder les fichiers mis à jour
            df.to_csv(DATA_PATH_DATA, index=False)
            df_unseen.to_csv(DATA_PATH_UNSEEN, index=False)
            
            # Enregistrer la taille actuelle du fichier dans Airflow Variables
            current_size = len(df)
            
            # Si c'est la première exécution, on stocke la taille initiale
            if Variable.get("data_size", default_var=None) is None:
                Variable.set("data_size", current_size)
                logger.info(f"Stockage de la taille initiale : {current_size}")
            
        else:
            logger.info("Aucune donnée à ajouter, le fichier unseen est vide.")
    
    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de l'ajout de données : {e}")
        raise e  # Optionnel, pour remonter l'erreur dans Airflow

add_data_task = PythonOperator(
    task_id='add_new_data',
    python_callable=add_new_data,
    provide_context=True,
    dag=dag
)
