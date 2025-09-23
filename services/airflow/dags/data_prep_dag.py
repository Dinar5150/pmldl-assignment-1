from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from airflow import DAG
from airflow.decorators import task

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / 'data' / 'raw' / 'diabetes.csv'
PROCESSED_DIR = REPO_ROOT / 'data' / 'processed'


with DAG(
    'data_prep_dag',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False
) as dag:

    @task()
    def load_data():
        return pd.read_csv(RAW_PATH)

    @task()
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().dropna()

    @task()
    def split_and_save(df: pd.DataFrame) -> dict:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train_path = PROCESSED_DIR / 'train.csv'
        test_path = PROCESSED_DIR / 'test.csv'
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        return {'train': str(train_path), 'test': str(test_path)}

    df = load_data()
    df_clean = clean_data(df)
    saved = split_and_save(df_clean)
