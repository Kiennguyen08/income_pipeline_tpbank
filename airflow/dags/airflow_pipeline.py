from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models.param import Param
import pandas as pd
import subprocess

import ruamel.yaml

# import yaml
import uuid

default_args = {
    "owner": "super",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def update_yaml(yaml_file, keys_to_update):
    yaml = ruamel.yaml.YAML()
    # Load the YAML file
    with open(yaml_file, "r") as file:
        yaml_data = yaml.load(file)

    # Update the specified keys with new values
    for key, value in keys_to_update.items():
        nested_keys = key.split(".")
        nested_data = yaml_data
        for nested_key in nested_keys[:-1]:
            nested_data = nested_data[nested_key]
        nested_data[nested_keys[-1]] = value

    # Write the modified YAML back to the file
    with open(yaml_file, "w") as file:
        yaml.dump(yaml_data, file)


def create_serving_data(**context):
    print("HELLLO")
    batch_id = str(uuid.uuid4())
    df = pd.read_csv(f"{context['params']['base_path']}/data/serve.csv")
    sampled = df.sample(frac=0.2)
    sampled_path = f"{context['params']['base_path']}/data/serve_{batch_id}.csv"
    sampled.to_csv(sampled_path)
    context["ti"].xcom_push(key="batch_id", value=batch_id)
    return sampled


def update_predict_config(**context):
    batch_id = context.get("ti").xcom_pull(key="batch_id")
    base_path = f"{context['params']['base_path']}"
    update_data = {
        "data.raw_data": f"{base_path}/data/serve_{batch_id}.csv",
        "data.test_data": f"{base_path}/data/serve_{batch_id}_processed.csv",
        "data.batch_id": f"{batch_id}",
        "predict.model_path": f"{base_path}/models/model.joblib",
        "predict.predictions_dir": f"{base_path}/data/predictions",
    }
    update_yaml(
        f"{context['params']['base_path']}/pipelines/predict/params.yaml", update_data
    )


def run_prediction(**context):
    bash_command = (
        f"dvc repro {context['params']['base_path']}/pipelines/predict/dvc.yaml"
    )
    subprocess.run(bash_command, shell=True)


def get_report_file(**context):
    batch_id = context.get("ti").xcom_pull(key="batch_id")
    bash_command = f"cp {context['params']['base_path']}/data/predictions/prediction_{batch_id}.csv ."
    subprocess.run(bash_command, shell=True)


with DAG(
    "simulate_serve_process",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    params = {
        "base_path": Param(os.getcwd(), type="string")
    }
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

    # predict = PythonOperator(
    #     task_id="run_prediction",
    #     python_callable=run_prediction,
    #     provide_context=True,
    # )
    
    predict = BashOperator(
        task_id="run_prediction",
        bash_command="dvc repro {{ params.base_path }}/pipelines/predict/dvc.yaml",
        params={"base_path": dag.params["base_path"]}
    )
    
    export_report = BashOperator(
        task_id="export_report",
        bash_command="cp {{ params.base_path }}/data/predictions/prediction_{{ ti.xcom_pull(key=\'batch_id\') }}.csv .",
        params={"base_path": dag.params["base_path"]}
    )

    # export_report = PythonOperator(
    #     task_id="export_report",
    #     python_callable=get_report_file,
    #     provide_context=True,
    # )

    # Set task dependencies
    create_data >> update_config >> predict >> export_report
