from __future__ import annotations

import joblib
import pandas as pd

from .config import ProjectConfig


def predict_from_dataframe(df: pd.DataFrame, config: ProjectConfig | None = None):
    config = config or ProjectConfig()
    model = joblib.load(config.model_path)
    return model.predict(df)


def predict_from_csv(input_csv: str, output_csv: str, config: ProjectConfig | None = None) -> None:
    config = config or ProjectConfig()
    df = pd.read_csv(input_csv)
    predictions = predict_from_dataframe(df, config=config)
    scored = df.copy()
    scored[config.target_column] = predictions
    scored.to_csv(output_csv, index=False)
