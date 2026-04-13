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


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace({"_": pd.NA, "": pd.NA, "NA": pd.NA})

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].astype("string").str.strip()

    for column in COLUMNS_TO_DROP:
        if column in cleaned.columns:
            cleaned = cleaned.drop(columns=column)

    return cleaned


def load_and_split_data(config: ProjectConfig):
    df = pd.read_csv(config.raw_data_path)
    df = _clean_frame(df)
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
