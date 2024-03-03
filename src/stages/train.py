import argparse
import logging
from pathlib import Path
from typing import Dict, List, Text
from catboost import CatBoostClassifier

import joblib
import pandas as pd
import yaml
from sklearn import ensemble


def train(config_path: Text) -> None:
    """Train model.

    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="TRAIN: %(message)s"
    )

    target_col: Text = config["data"]["target_col"]

    logging.info("Load train data")
    train_data_path: Path = Path(config["data"]["train_data"])
    train_data: pd.DataFrame = pd.read_csv(train_data_path)
    logging.info("Train model")
    classifier = CatBoostClassifier(
        depth=config["train"]["depth"],
        iterations=config["train"]["iterations"],
        learning_rate=config["train"]["learning_rate"],
    )
    logging.info(f"Cols: {train_data.columns}")

    classifier.fit(
        X=train_data[train_data.columns[:-1]],
        y=train_data[target_col],
    )

    logging.info("Save the model")
    model_path: Path = Path(config["train"]["model_path"])
    joblib.dump(classifier, model_path)
    logging.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)