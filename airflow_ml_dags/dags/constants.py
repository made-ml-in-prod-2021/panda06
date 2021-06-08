from airflow.models import Variable


DEFAULT_ARGS = {
    "owner": "vgaparkhoev",
    "retries": 1,
    "email": [Variable.get("SEND_TO")],
    'email_on_failure': True
}

VOLUME_DIR = Variable.get("VOLUME_DIR")
RAW_DATA_DIR = "/data/raw/{{ ds }}"
SENSOR_RAW_DATA_DIR = "data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODEL_DIR = "/data/models/{{ ds }}"
PREDICT_DIR = "/data/predictions/{{ ds }}"
