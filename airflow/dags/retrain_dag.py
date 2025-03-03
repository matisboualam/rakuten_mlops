import os
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import requests
from datetime import datetime, timedelta

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
    schedule_interval="0 * * * *",  # Runs every minute
    catchup=False
)

task_0 = BashOperator(
        task_id="run_dvc_split",
        bash_command="python ../src/preprocessing/split.py",
        dag=dag
    )

task_1 = BashOperator(
        task_id="train_model",
        bash_command="python ../src/modeling/train.py",
        dag=dag
    )

task_0 >> task_1