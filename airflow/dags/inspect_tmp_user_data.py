from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
import pandas as pd
import os
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin
import requests

API_URL = "http://deployment:8080"  # Replace with the correct URL

# Define DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 8),
    'retries': 1,
}

# Paths to the CSV files
TMP_CSV_FILE_PATH = "../data/processed/tmp/tmp_predicted_data.csv"
DATA_CSV_FILE_PATH = "../data/processed/data.csv"
TARGET_DAG_ID = "retrain_dag"  # DAG to be triggered if condition met

# Function to check CSV length
def check_csv_length(**kwargs):

    log = LoggingMixin().log

    try:
        if not os.path.exists(TMP_CSV_FILE_PATH):
            log.info(f"File does not exist: {TMP_CSV_FILE_PATH}")
            return "skip_trigger"
        
        df = pd.read_csv(TMP_CSV_FILE_PATH)  # Read CSV
        row_count = len(df)  # Count rows
        unique_prdtypecode_count = df['prdtypecode'].nunique()  # Count unique values in 'prdtypecode' column
        
        log.info(f"Row count: {row_count}")
        log.info(f"Unique 'prdtypecode' count: {unique_prdtypecode_count}")
        
        result = row_count >= 100 and unique_prdtypecode_count == 27  # Check conditions
        log.info(f"Condition met: {result}")
        return "merge_csv_files" if result else "skip_trigger"
    except Exception as e:
        log.error(f"Error reading CSV: {e}")
        return "skip_trigger"

# Function to merge CSV files without duplicates and empty tmp predicted data
def merge_csv_files():
    log = LoggingMixin().log
    # Send merged data to API endpoint
    response = requests.post(f"{API_URL}/merge")
    if response.status_code == 200:
        log.info(f"Successfully sent merged data to API: {API_URL}/merge")
    else:
        log.error(f"Failed to send merged data to API: {response.status_code} - {response.text}")

# Define DAG
with DAG(
    'check_csv_and_trigger_dag',
    default_args=default_args,
    schedule_interval="*/5 * * * *",  # Run every 15 minutes
    catchup=False,
) as dag:

    check_csv_task = BranchPythonOperator(
        task_id="check_csv_length",
        python_callable=check_csv_length,
        provide_context=True,
    )

    merge_csv_task = PythonOperator(
        task_id="merge_csv_files",
        python_callable=merge_csv_files,
        provide_context=True,
    )

    trigger_dag_task = TriggerDagRunOperator(
        task_id="trigger_target_dag",
        trigger_dag_id=TARGET_DAG_ID,
        wait_for_completion=False,
        execution_date="{{ ds }}",
        reset_dag_run=True,
    )

    skip_trigger = DummyOperator(
        task_id="skip_trigger"
    )

    # Set up dependencies
    check_csv_task >> [merge_csv_task, skip_trigger]
    merge_csv_task >> trigger_dag_task
