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
    update_data = {
        "data.predict_data": f"{base_path}/data/serve_{batch_id}.csv",
        "data.batch_id": f"{batch_id}",
        "predict.model_path": f"{base_path}/models/model.joblib",
        "predict.predictions_dir": f"{base_path}/data/predictions",
        "monitoring.report_dir": f"{base_path}/reports",
        "monitoring.reference_data": f"{base_path}/data/reference_data.csv",
    }
    update_yaml(
        f"{context['params']['base_path']}/pipelines/monitor/params.yaml", update_data
    )


with DAG(
    "monitor_model_and_data",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    params={
        "base_path": Param(os.getcwd(), type="string"),
        "batch_id": Param("", type="string"),
    },
) as dag:

    update_config = PythonOperator(
        task_id="update_config",
        python_callable=update_predict_config,
        provide_context=True,
    )

    trigger_export_report = BashOperator(
        task_id="run_prediction",
        bash_command="cd {{ params.base_path }} && dvc repro {{ params.base_path }}/pipelines/monitor/dvc.yaml",
        params={"base_path": dag.params["base_path"]},
    )

    export_report = BashOperator(
        task_id="export_report",
        bash_command="cp {{ params.base_path }}/report/{{ params.batch_id }}/*.html {{ params.batch_id }}/",
        params={
            "base_path": dag.params["base_path"],
            "batch_id": dag.params["batch_id"],
        },
    )

    # Set task dependencies
    update_config >> trigger_export_report >> export_report
