# Classification-Credit-Score

End-to-end data science project for **credit score classification** using the `credit_score.csv` dataset.

## What is now included

This repository now includes a complete machine learning workflow:

- Data loading and cleaning.
- Feature preprocessing for numeric and categorical columns.
- Model training pipeline with `RandomForestClassifier`.
- Evaluation and metrics export.
- Persisted model artifact for reuse.
- Batch prediction from CSV files.
- CLI entry point for training and inference.

## Project structure

```text
.
├── credit_score.csv
├── credit_score.ipynb
├── README.md
├── requirements.txt
├── scripts/
│   ├── train.py
│   └── predict.py
└── src/
    └── credit_score_pipeline/
        ├── __init__.py
        ├── cli.py
        ├── config.py
        ├── data.py
        ├── predict.py
        ├── preprocess.py
        └── train.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Train the model

```bash
python -m credit_score_pipeline.cli train --pretty
```

Training outputs are saved to:

- `artifacts/credit_score_model.joblib`
- `artifacts/metrics.json`

## Run batch prediction

```bash
python -m credit_score_pipeline.cli predict --input new_customers.csv --output scored_customers.csv
```

The output file will contain all input columns plus the predicted `Credit_Score`.

## Notebook

The notebook `credit_score.ipynb` remains available for interactive analysis.

## License

This project is licensed under the MIT License.
