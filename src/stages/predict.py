import argparse
import logging
from pathlib import Path
from typing import Dict, List, Text

import joblib
import pandas as pd
import yaml


def predict(config_path: Text) -> None:
    """Make and save predictions on reference and predict data.
    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="PREDICT: %(message)s"
    )

    predictions_dir: Path = Path(config["predict"]["predictions_dir"])
    predictions_dir.mkdir(exist_ok=True)

    logging.info("Load metadata")
    prediction_col: Text = config["data"]["prediction_col"]
    batch_id = config["data"]["batch_id"]

    logging.info("Load data")
    predict_data_path: Path = config["data"]["test_data"]
    # predict_data_path: Path = workdir / config["data"]["predict_data"]
    predict_data: pd.DataFrame = pd.read_csv(predict_data_path)

    logging.info("Load model")
    model_path: Path = config["predict"]["model_path"]
    # model_path: Path = workdir / config["predict"]["model_path"]
    model = joblib.load(model_path)
    logging.info(model)

    logging.info("Make predictions")
    predictions = model.predict(predict_data[predict_data.columns[:-1]])

    logging.info("Save predictions")
    raw_test_path: Path = config["data"]["raw_data"]
    raw_test_df: pd.DataFrame = pd.read_csv(raw_test_path)
    raw_test_df[prediction_col] = [int(pred) for pred in predictions]
    predictions_path: Path = predictions_dir / f"prediction_{batch_id}.csv"
    raw_test_df.to_csv(predictions_path, index=False)
    logging.info(f"Save data with predictions to {predictions_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    predict(config_path=args.config)
