from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import pandas as pd
import subprocess
from edit_yaml import update_yaml
import uuid

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 3, 4),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def create_serving_data(**context):
    batch_id = str(uuid.uuid4())
    df = pd.read_csv("data/serve.csv")
    sampled = df.sample(frac=0.2)
    sampled_path = f"data/serve_{batch_id}.csv"
    sampled.to_csv(sampled_path)
    context["ti"].xcom_push(key="batch_id", value=batch_id)
    return sampled


def update_predict_config(**context):
    batch_id = context.get("ti").xcom_pull(key="batch_id")
    update_data = {
        "data.raw_data": f"data/serve_{batch_id}.csv",
        "data.test_data": f"data/serve_{batch_id}.csv",
        "data.batch_id": "new_batch_id",
    }
    update_yaml("pipelines/predict/params.yaml", update_data)


def run_prediction(**context):
    bash_command = "dvc repro pipelines/predict/dvc.yaml"
    subprocess.run(bash_command, shell=True)


def get_report_file(**context):
    batch_id = context.get("ti").xcom_pull(key="batch_id")
    bash_command = f"cp data/predictions/prediction_{batch_id}.csv ."
    subprocess.run(bash_command, shell=True)


with DAG(
    "simulate_serve_process",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:
    # Define a PythonOperator that uses my_task function
    create_data = PythonOperator(
        task_id="create_data",
        python_callable=create_serving_data,
        provide_context=True,
    )

    update_config = PythonOperator(
        task_id="update_config",
        python_callable=update_predict_config,
        provide_context=True,
    )

    predict = PythonOperator(
        task_id="run_prediction",
        python_callable=run_prediction,
        provide_context=True,
    )

    export_report = PythonOperator(
        task_id="export_report",
        python_callable=get_report_file,
        provide_context=True,
    )


# Set task dependencies
create_data >> update_config >> predict >> export_report
