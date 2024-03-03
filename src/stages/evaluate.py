import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Text

from dvclive import Live
import joblib
import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report

from utils import _preprocessing


def evaluate(config_path: Text, pdir: Text) -> None:
    """Train model.

    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="TRAIN: %(message)s"
    )

    REPORTS_DIR: Path = Path(config["base"]["reports_dir"])
    REPORTS_DIR.mkdir(exist_ok=True)

    target_col: Text = config["data"]["target_col"]
    prediction_col: Text = config["data"]["prediction_col"]

    logging.info("Load data")
    train_data_path: Path = Path(config["data"]["train_data"])
    val_data_path: Path = Path(config["data"]["val_data"])
    numerical_features: List[Text] = config["data"]["numerical_features"]
    categorical_features: List[Text] = config["data"]["categorical_features"]
    train_data: pd.DataFrame = pd.read_csv(train_data_path).reset_index(drop=True)
    val_data: pd.DataFrame = pd.read_csv(val_data_path).reset_index(drop=True)
    train_data = _preprocessing(train_data, numerical_features, categorical_features)
    val_data = _preprocessing(val_data, numerical_features, categorical_features)

    logging.info("Load model")
    model_path: Path = Path(config["train"]["model_path"])
    model = joblib.load(model_path)

    logging.info("Get predictions to VALIDATION data")
    train_prediction = model.predict(train_data[train_data.columns[:-1]])
    val_prediction = model.predict(val_data[val_data.columns[:-1]])
    train_prediction = [int(x) for x in train_prediction]
    val_prediction = [int(x) for x in val_prediction]

    logging.info("Prepare datasets for monitoring")
    val_data["prediction"] = val_prediction
    train_data["prediction"] = train_prediction

    reference_data = train_data.sample(frac=0.3)

    logging.info("Prepare column_mapping object for Evidently reports")
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col
    column_mapping.target_names = {1: "High", 0: "Low"}
    column_mapping.pos_label = 0

    logging.info("Create a model performance report")
    model_performance_report = Report(metrics=[ClassificationPreset()])
    model_performance_report.run(
        reference_data=reference_data,
        current_data=val_data,
        column_mapping=column_mapping,
    )

    # Save reports in HTML format
    model_performance_report_path = REPORTS_DIR / "model_performance.html"
    model_performance_report.save_html(str(model_performance_report_path))

    logging.info("Extract metrics")
    classification_metrics: Dict = model_performance_report.as_dict()["metrics"][0][
        "result"
    ]["current"]

    metric_names = ["accuracy", "f1"]
    selected_metrics = {k: classification_metrics.get(k) for k in metric_names}

    logging.info("Save evaluation metrics and model report")
    with Live(
        dir=str(REPORTS_DIR),
        dvcyaml=f"{pdir}/dvc.yaml",
    ) as live:
        [live.log_metric(k, v, plot=False) for k, v in selected_metrics.items()]

    logging.info("Save reference data")
    ref_data_path = Path(config["data"]["reference_data"])
    reference_data.to_csv(ref_data_path, index=None)
    logging.info(f"Saved reference data to to {ref_data_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args_parser.add_argument("--pdir", dest="pdir", required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config, pdir=args.pdir)
