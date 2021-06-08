from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator

from constants import DEFAULT_ARGS, PREDICT_DIR, RAW_DATA_DIR, VOLUME_DIR

with DAG(
        dag_id="predict",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=datetime.now()) as dag:

    t1 = DockerOperator(
        task_id="predict",
        image="airflow-predict",
        command=f"--input_dir {RAW_DATA_DIR} --output_dir {PREDICT_DIR} --model_path {Variable.get('MODEL_PATH')}",
        do_xcom_push=False,
        volumes=[f"{VOLUME_DIR}:/data"]
    )
