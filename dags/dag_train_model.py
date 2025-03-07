from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import requests
from tasks.check_data_growth import check_data_growth

# Définir les arguments par défaut
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 2, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Création du DAG
dag = DAG(
    'dag_train_model',
    default_args=default_args,
    schedule_interval="*/30 * * * *",
    catchup=False
)

# Nom du service Docker (conteneur Dev)
API_URL = "http://tensorflow:8001"

# Fonction pour appeler l'API
def call_api(endpoint: str):
    url = f"{API_URL}/{endpoint}"
    try:
        response = requests.post(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# Tâche pour vérifier la croissance des données
check_data_task = BranchPythonOperator(
    task_id='check_data_growth',
    python_callable=check_data_growth,
    provide_context=True,
    dag=dag
)

# Tâche "skip_training" pour indiquer que l'entraînement ne doit pas être effectué
skip_training_task = DummyOperator(
    task_id='skip_training',  
    dag=dag
)

# Tâche pour diviser les données (appelle `/split_data`)
split_data_task = PythonOperator(
    task_id="split_data",
    python_callable=call_api,
    op_kwargs={"endpoint": "split_data"},
    dag=dag,
)

# Tâche pour entraîner le modèle (appelle `/train`)
train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=call_api,
    op_kwargs={"endpoint": "train"},
    dag=dag,
)

# Définition du flow des tâches
check_data_task >> [skip_training_task, split_data_task]
split_data_task >> train_model_task
