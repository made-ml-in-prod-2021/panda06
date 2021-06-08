import os
from datetime import datetime

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from constants import DEFAULT_ARGS, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, SENSOR_RAW_DATA_DIR, VOLUME_DIR

with DAG(
        dag_id="train",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=datetime.now()) as dag:

    t0 = DummyOperator(task_id='begin')

    data_wait = FileSensor(
        task_id='data_wait',
        poke_interval=10,
        fs_conn_id='fs',
        retries=100,
        filepath=os.path.join(SENSOR_RAW_DATA_DIR, 'data.csv')
    )
    preprocess = DockerOperator(
        image="airflow-train",
        command=f"--input_dir {RAW_DATA_DIR} "
                f"--output_dir {PROCESSED_DATA_DIR}",
        do_xcom_push=False,
        task_id="data-preprocess",
        volumes=[f"{VOLUME_DIR}:/data"],
        entrypoint="python preprocess.py"
    )

    data_split = DockerOperator(
        image="airflow-train",
        command=f"--input_dir {PROCESSED_DATA_DIR} "
                f"--train_size 0.7",
        do_xcom_push=False,
        task_id="data-split",
        volumes=[f"{VOLUME_DIR}:/data"],
        entrypoint="python split.py"
    )

    train_model = DockerOperator(
        image="airflow-train",
        command=f"--input_dir {PROCESSED_DATA_DIR} "
                f"--model_path {MODEL_DIR}",
        network_mode="bridge",
        do_xcom_push=False,
        task_id="train_model",
        volumes=[f"{VOLUME_DIR}:/data"],
        entrypoint="python train.py"
    )

    validate_model = DockerOperator(
        image="airflow-train",
        command=f"--input_dir {PROCESSED_DATA_DIR} "
                f"--model_dir {MODEL_DIR}",
        do_xcom_push=False,
        task_id="validate_model",
        volumes=[f"{VOLUME_DIR}:/data"],
        entrypoint="python validate.py"
    )
    t0 >> data_wait >> preprocess >> data_split >> train_model >> validate_model
