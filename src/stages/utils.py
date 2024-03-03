from typing import List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


def _minmax_scale_columns(df: pd.DataFrame, columns_to_scale):
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    scaled_df = df.copy()
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit and transform the specified columns using Min-Max scaling
    scaled_df[columns_to_scale] = scaler.fit_transform(scaled_df[columns_to_scale])
    return scaled_df


def _preprocessing(
    df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str]
):
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    label_encoder = LabelEncoder()
    df.replace("?", np.nan, inplace=True)
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    for i in categorical_cols:
        df[i] = label_encoder.fit_transform(df[i])
    df = _minmax_scale_columns(df, df.columns[:-1])
    return df
