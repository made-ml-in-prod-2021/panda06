from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from constants import DEFAULT_ARGS, RAW_DATA_DIR, VOLUME_DIR

with DAG(
        dag_id="dataloader",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=datetime.now()) as dag:

    t1 = DockerOperator(
        task_id="download-data",
        image="airflow-download",
        command=f"--output_dir {RAW_DATA_DIR}",
        do_xcom_push=False,
        volumes=[f"{VOLUME_DIR}:/data"]
    )
