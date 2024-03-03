import argparse
import logging
from pathlib import Path
from typing import Dict, Text

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import TargetDriftPreset, RegressionPreset
from evidently.report import Report


def monitoring(config_path: Text) -> None:
    """Build and save monitoring reports.

    Args:
        batch_id: ID of batch Inference
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="MONITORING MODEL: %(message)s"
    )

    # WEEK_START = config["predict"]["week_start"]
    # WEEK_END = config["predict"]["week_end"]
    BATCH_ID = config["data"]["batch_id"]
    logging.info(f"Predict for batch: {BATCH_ID}")

    # Load Data
    PREDICTIONS_DIR: Path = Path(config["predict"]["predictions_dir"])
    REPORTS_DIR: Path = Path(config["monitoring"]["reports_dir"]) / f"{BATCH_ID}"
    REPORTS_DIR.mkdir(exist_ok=True)

    reference_data_path: Path = Path(config["monitoring"]["reference_data"])
    reference: pd.DataFrame = pd.read_csv(reference_data_path)
    current_data_path: Path = PREDICTIONS_DIR / f"prediction_{BATCH_ID}.csv"
    current: pd.DataFrame = pd.read_csv(current_data_path)

    logging.info("Prepare column_mapping object for Evidently reports")
    column_mapping = ColumnMapping()
    column_mapping.target = config["data"]["target_col"]
    column_mapping.prediction = config["data"]["prediction_col"]
    column_mapping.numerical_features = config["data"]["numerical_features"]
    column_mapping.categorical_features = config["data"]["categorical_features"]
    print(f"column mapping: {column_mapping}")

    logging.info("Build Model Reports...")

    # Model performance report
    model_performance_report = Report(metrics=[RegressionPreset()])
    model_performance_report.run(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    model_performance_report_path = (
        REPORTS_DIR / config["monitoring"]["model_performance_path"]
    )
    model_performance_report.save_html(str(model_performance_report_path))
    logging.info(f"Model Performance report saved to {model_performance_report_path}")

    # Target drift report
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    target_drift_report_path = REPORTS_DIR / config["monitoring"]["target_drift_path"]
    target_drift_report.save_html(str(target_drift_report_path))
    logging.info(f"Target Drift report saved to: {target_drift_report_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    monitoring(config_path=args.config)
