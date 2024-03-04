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


def update_predict_config(**context):
    batch_id = f"{context['params']['batch_id']}"
    base_path = f"{context['params']['base_path']}"
    depth = f"{context['params']['depth']}"
    iterations = f"{context['params']['iterations']}"
    learning_rate = f"{context['params']['learning_rate']}"
    train_data_path = f"{context['params']['train_path']}"

    update_data = {
        "base.reports_dir": f"{base_path}/reports/train_{batch_id}",
        "data.raw_data": f"{train_data_path}",
        "data.train_data": f"{base_path}/data/train.csv",
        "data.val_data": f"{base_path}/data/val.csv",
        "data.reference_data": f"{base_path}/data/reference_data.csv",
        "train.depth": f"{depth}",
        "train.iterations": f"{iterations}",
        "train.learning_rate": f"{learning_rate}",
    }
    update_yaml(
        f"{context['params']['base_path']}/pipelines/train/params.yaml", update_data
    )


with DAG(
    "train_model",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    params={
        "base_path": Param(os.getcwd(), type="string"),
        "batch_id": Param("", type="string"),
        "depth": Param(4, type="number"),
        "iterations": Param(200, type="number"),
        "learning_rate": Param(0.01, type="number"),
        "train_path": Param(f"{os.getcwd()}/data/train.txt", type="string"),
    },
) as dag:
    # Define a PythonOperator that uses my_task function
    update_config = PythonOperator(
        task_id="update_config",
        python_callable=update_predict_config,
        provide_context=True,
    )

    train = BashOperator(
        task_id="run_trainning",
        bash_command="cd {{ params.base_path }} && dvc exp run {{ params.base_path }}/pipelines/train/dvc.yaml",
        params={"base_path": dag.params["base_path"]},
    )

    export_report = BashOperator(
        task_id="export_report",
        bash_command="cp {{ params.base_path }}/data/reports/train_{{ params.batch_id }}/*.html .",
        params={
            "base_path": dag.params["base_path"],
            "batch_id": dag.params["batch_id"],
        },
    )

    # Set task dependencies
    update_config >> train >> export_report
