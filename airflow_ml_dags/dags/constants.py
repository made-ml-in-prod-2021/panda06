DEFAULT_ARGS = {
    "owner": "vgaparkhoev",
    "retry": 1
}

VOLUME_DIR = "/home/panda/projects/made/mlops/panda06/airflow_ml_dags/data"
RAW_DATA_DIR = "/data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODEL_DIR = "/data/models/{{ ds }}"
PREDICT_DIR = "/data/predictions/{{ ds }}"

