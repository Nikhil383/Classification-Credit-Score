from __future__ import annotations

import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from .config import ProjectConfig
from .data import load_and_split_data
from .preprocess import build_preprocessor


def run_training(config: ProjectConfig | None = None) -> dict:
    config = config or ProjectConfig()
    config.ensure_artifact_dir()

    X_train, X_test, y_train, y_test = load_and_split_data(config)
    preprocessor = build_preprocessor(X_train)

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=config.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }

    joblib.dump(pipeline, config.model_path)
    with config.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
