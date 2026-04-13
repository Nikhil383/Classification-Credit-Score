from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ProjectConfig


COLUMNS_TO_DROP = [
    "ID",
    "Customer_ID",
    "Name",
    "SSN",
    "Month",
    "Type_of_Loan",
    "Credit_History_Age",
]


MISSING_VALUE_TOKENS = {"_": pd.NA, "": pd.NA, "NA": pd.NA}


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace(MISSING_VALUE_TOKENS)

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].astype("string").str.strip()

    return cleaned


def _drop_leakage_and_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in COLUMNS_TO_DROP:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=column)
    return cleaned


def load_and_split_data(config: ProjectConfig):
    df = pd.read_csv(config.raw_data_path)
    df = _drop_leakage_and_identifier_columns(_clean_frame(df))
    df = df.dropna(subset=[config.target_column])

    y = df[config.target_column]
    X = df.drop(columns=[config.target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def prepare_features_for_inference(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    prepared = _drop_leakage_and_identifier_columns(_clean_frame(df))
    if config.target_column in prepared.columns:
        prepared = prepared.drop(columns=[config.target_column])
    return prepared
