from datetime import datetime
from pathlib import Path
import subprocess
import sys

from airflow import DAG
from airflow.decorators import task

from data_prep_dag import load_data, clean_data, split_and_save

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / 'models'
TRAIN_SCRIPT = REPO_ROOT / 'code' / 'models' / 'train.py'

with DAG(
    'complete_pipeline_dag',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    @task()
    def train_model():
        result = subprocess.run([
            sys.executable, str(TRAIN_SCRIPT)
        ], capture_output=True, text=True, cwd=str(REPO_ROOT))
        
        return f"Model training completed successfully. Output: {result.stdout}"

    @task()
    def deploy_services():
        compose_file = REPO_ROOT / 'code' / 'deployment' / 'docker-compose.yml'
        result = subprocess.run([
            'docker-compose', '-f', str(compose_file), 'up', '-d', '--build'
        ], capture_output=True, text=True, cwd=str(compose_file.parent))

        return f"Services deployed successfully. Output: {result.stdout}"


    df = load_data()
    df_clean = clean_data(df)
    saved = split_and_save(df_clean)
    model_result = train_model()
    deployment_result = deploy_services()
