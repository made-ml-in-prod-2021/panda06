import os
from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from constants import DEFAULT_ARGS, PREDICT_DIR, RAW_DATA_DIR, VOLUME_DIR, SENSOR_RAW_DATA_DIR

with DAG(
        dag_id="predict",
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

    model_wait = FileSensor(
        task_id='model_wait',
        fs_conn_id='fs',
        poke_interval=10,
        retries=100,
        filepath=Variable.get('MODEL_PATH')[1:]
    )

    t1 = DockerOperator(
        task_id="predict",
        image="airflow-predict",
        command=f"--input_dir {RAW_DATA_DIR} --output_dir {PREDICT_DIR} --model_path {Variable.get('MODEL_PATH')}",
        do_xcom_push=False,
        volumes=[f"{VOLUME_DIR}:/data"]
    )

    t0 >> [data_wait, model_wait] >> t1
