from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from tasks.check_data_growth import check_data_growth

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 2, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'dag_train_model',
    default_args=default_args,
    schedule_interval="*/30 * * * *",
    catchup=False
)

# Tâche pour vérifier la croissance des données
check_data_task = BranchPythonOperator(
    task_id='check_data_growth',
    python_callable=check_data_growth,
    provide_context=True,
    dag=dag
)

# Tâche "skip_training" pour indiquer que l'entraînement ne doit pas être effectué
skip_training_task = DummyOperator(
    task_id='skip_training',  # Ceci est maintenant une tâche valide
    dag=dag
)

# Tâche "process_data" pour l'entraînement des données
process_data_task = DummyOperator(
    task_id='process_data',  # Remplacer par votre tâche d'entraînement réelle
    dag=dag
)

# Définir les dépendances entre les tâches
check_data_task >> [process_data_task, skip_training_task]
