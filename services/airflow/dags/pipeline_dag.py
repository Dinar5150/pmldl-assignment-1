from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

with DAG(
    dag_id='minimal_ml_pipeline',
    start_date=datetime(2025, 9, 24),
    schedule='*/5 * * * *',  # every 5 minutes
    catchup=False,
    max_active_runs=1,
) as dag:

    prepare = BashOperator(
        task_id='prepare_data',
        bash_command=f'python3 {os.path.join(ROOT, "code", "datasets", "prepare_data.py")}'
    )

    train = BashOperator(
        task_id='train_model',
        bash_command=f'python3 {os.path.join(ROOT, "code", "models", "train.py")}'
    )

    deploy = BashOperator(
        task_id='deploy_services',
        bash_command=f'cd {os.path.join(ROOT, "code", "deployment")} && docker-compose up -d --build'
    )

    prepare >> train >> deploy
