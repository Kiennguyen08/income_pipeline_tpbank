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


def preprocessing(config_path: str):
    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="PREPROCESS: %(message)s"
    )

    logging.info("Load raw data")
    raw_data_path: Dict = Path(config["data"]["raw_data"])
    df: pd.DataFrame = pd.read_csv(raw_data_path)

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    label_encoder = LabelEncoder()

    df.replace("?", np.nan, inplace=True)
    df = df.drop(df.columns[0], axis=1)
    if "INCOME" in df.columns:
        df = df.drop("INCOME", axis=1)
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    for i in categorical_cols:
        df[i] = label_encoder.fit_transform(df[i])
    df = _minmax_scale_columns(df, df.columns[:-1])
    print('df', df.head(5))
    logging.info("Save train_data and test_data data")
    df.to_csv(config["data"]["test_data"])


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    preprocessing(config_path=args.config)
