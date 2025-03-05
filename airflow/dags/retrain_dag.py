import os
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from datetime import datetime, timedelta

API_URL = "http://deployment:5001"  # Replace with the correct URL

def split_data():
    response = requests.post(os.path.join(API_URL, 'split_data'))
    if response.status_code != 200:
        raise Exception(f"Failed to split: {response.status_code}, {response.text}")
    else:
        print("Splitting started successfully.")


def launch_train():
    response = requests.post(os.path.join(API_URL, 'train'))
    if response.status_code == 200:
        print("train launched:")
    else:
        raise Exception(f"Failed to send feedback: {response.status_code}, {response.text}")

# Airflow DAG definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 14),
    'retries': 0
}

dag = DAG(
    'retrain_dag',
    default_args=default_args,
    schedule_interval="0 * * * *",  # Runs every hour
    catchup=False
)

task_0 = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag
)

task_1 = PythonOperator(
    task_id='launch_train',
    python_callable=launch_train,
    dag=dag
)

task_0 >> task_1