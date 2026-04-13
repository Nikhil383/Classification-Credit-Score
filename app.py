from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_score_pipeline.config import ProjectConfig
from credit_score_pipeline.predict import predict_from_dataframe
from credit_score_pipeline.train import run_training

st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="wide")

config = ProjectConfig()

st.title("💳 Credit Score Classification")
st.caption("End-to-end data science app with training and batch inference.")

with st.sidebar:
    st.header("Pipeline controls")
    if st.button("Train / Retrain model", use_container_width=True):
        with st.spinner("Training model. This may take a while..."):
            metrics = run_training(config)
        st.success("Training completed and artifacts saved.")
        st.json(metrics)

    st.divider()
    st.write("Model artifact path:")
    st.code(str(config.model_path))

model_exists = Path(config.model_path).exists()
if not model_exists:
    st.warning("No trained model found. Train the model from the sidebar before running predictions.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Batch prediction")
    uploaded = st.file_uploader("Upload CSV file for scoring", type=["csv"])

    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(input_df.head(20), use_container_width=True)

        if model_exists:
            if st.button("Generate predictions", type="primary"):
                with st.spinner("Scoring uploaded records..."):
                    preds = predict_from_dataframe(input_df, config=config)
                    output_df = input_df.copy()
                    output_df[config.target_column] = preds

                st.success("Predictions generated.")
                st.dataframe(output_df.head(20), use_container_width=True)
                st.download_button(
                    label="Download predictions CSV",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name="credit_score_predictions.csv",
                    mime="text/csv",
                )

with col2:
    st.subheader("Latest metrics")
    if config.metrics_path.exists():
        with config.metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with st.expander("Classification report"):
            st.json(metrics.get("classification_report", {}))
    else:
        st.info("Metrics file not found yet. Train model first.")
