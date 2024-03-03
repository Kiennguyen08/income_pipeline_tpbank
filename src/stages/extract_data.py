import argparse
import logging
from pathlib import Path
from typing import Dict, Text

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import numpy as np


def _minmax_scale_columns(df: pd.DataFrame, columns_to_scale):
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    scaled_df = df.copy()
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit and transform the specified columns using Min-Max scaling
    scaled_df[columns_to_scale] = scaler.fit_transform(scaled_df[columns_to_scale])
    return scaled_df


def _preprocessing(
    df: pd.DataFrame,
    num_imputer: SimpleImputer = SimpleImputer(strategy="median"),
    cat_imputer: SimpleImputer = SimpleImputer(strategy="most_frequent"),
    label_encoder: LabelEncoder = LabelEncoder(),
    scaler: MinMaxScaler = MinMaxScaler(),
):
    df.replace("?", np.nan, inplace=True)
    df.columns = [
        "AGE",
        "WORKCLASS",
        "EDUCATION",
        "EDUCATIONAL-NUM",
        "MARITAL-STATUS",
        "OCCUPATION",
        "RELATIONSHIP",
        "GENDER",
        "CAPITAL-GAIN",
        "CAPITAL-LOSS",
        "HOURS-PER-WEEK",
        "INCOME",
    ]
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    for i in categorical_cols:
        df[i] = label_encoder.fit_transform(df[i])
    df = _minmax_scale_columns(df, df.columns[:-1])
    return df


def extract_data(config_path: Text) -> None:
    """Extract and preprocessing data

    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="EXTRACT_DATA: %(message)s"
    )

    logging.info("Load raw data")
    raw_data_path: Dict = Path(config["data"]["raw_data"])
    raw_data: pd.DataFrame = pd.read_csv(raw_data_path)
    raw_data = raw_data.drop(columns="0")

    logging.info("Preprocess data")
    processed_data = _preprocessing(raw_data)

    logging.info("Extract reference and current data")
    test_size: float = float(config["data"]["test_size_ratio"])
    train_data, test_data = train_test_split(
        processed_data, test_size=test_size, random_state=42
    )

    print(type(train_data))

    logging.info("Save train_data and test_data data")
    train_data.to_csv(config["data"]["train_data"])
    test_data.to_csv(config["data"]["test_data"])


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    extract_data(config_path=args.config)