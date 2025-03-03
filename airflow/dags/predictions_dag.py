import os
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from datetime import datetime, timedelta

API_URL = "http://deployment:8000"  # Replace with the correct URL

def generate_indices(csv, **kwargs):
    df = pd.read_csv(csv)
    indices = np.random.randint(0, len(df), size=50).tolist()
    print('indices generated : ', indices)
    kwargs['ti'].xcom_push(key='indices', value=indices)
    return indices

def prediction_api(**kwargs):
    ti = kwargs['ti']
    indices = ti.xcom_pull(task_ids='generate_indices', key='indices')
    predictions = []
    
    for indice in indices:
        params = {"indice": indice}
        response = requests.post(os.path.join(API_URL, 'predict'), params=params)
        
        if response.status_code == 200:
            prediction_output = response.json()
            predictions.append(prediction_output)
            print("Prediction received:", prediction_output)
        else:
            raise Exception(f"Failed to get prediction: {response.status_code}, {response.text}")
    
    ti.xcom_push(key='predictions', value=predictions)

def feedback_api(**kwargs):
    ti = kwargs['ti']
    predictions = ti.xcom_pull(task_ids='call_prediction', key='predictions')
    
    for prediction_output in predictions:
        params = {
            "indice": prediction_output['indice'],
            "predicted_class": prediction_output['predicted_class']
        }
        
        response = requests.post(os.path.join(API_URL, 'predict', 'feedback'), params=params)
        if response.status_code == 200:
            print("Feedback received:", response.json())
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
    'prediction_dag',
    default_args=default_args,
    schedule_interval="* * * * *",  # Runs every minute
    catchup=False
)

task_0 = PythonOperator(
    task_id='generate_indices',
    python_callable=generate_indices,
    op_args=['../data/processed/unseen.csv'],  # Set the path of the CSV file as an argument
    provide_context=True,
    dag=dag
)

task_1 = PythonOperator(
    task_id='call_prediction',
    python_callable=prediction_api,
    provide_context=True,
    dag=dag
)

task_2 = PythonOperator(
    task_id='call_feedback',
    python_callable=feedback_api,
    provide_context=True,
    dag=dag
)

task_0 >> task_1 >> task_2