from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

def run_pipeline():
    subprocess.run(["python", "pipeline/train_pipeline.py"])

with DAG("multimodal_forecasting", start_date=datetime(2025, 1, 1), schedule_interval="@daily", catchup=False) as dag:
    task = PythonOperator(
        task_id="run_training_pipeline",
        python_callable=run_pipeline
    )
