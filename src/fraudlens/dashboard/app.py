"""Streamlit reporting and fraud-check interface.

Run with:  streamlit run src/fraudlens/dashboard/app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fraudlens.database.connection import get_engine
from fraudlens.database.models import EtlRun, ModelVersion, Prediction

st.set_page_config(page_title="FraudLens", page_icon="🔍", layout="wide")

API_URL = os.getenv("FRAUDLENS_API_URL", "http://localhost:8000")
TIMEOUT = 60


@st.cache_data(ttl=60)
def load_overview() -> dict:
    with Session(get_engine()) as session:
        total = session.execute(select(func.count()).select_from(Prediction)).scalar_one()
        flagged = session.execute(
            select(func.count()).select_from(Prediction).where(Prediction.predicted_label == 1)
        ).scalar_one()
        confirmed = session.execute(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.review_status == "confirmed_fraud")
        ).scalar_one()
        return {
            "predictions": total,
            "flagged": flagged,
            "confirmed_fraud": confirmed,
            "flag_rate": (flagged / total) if total else 0.0,
        }


@st.cache_data(ttl=60)
def load_recent_predictions(limit: int = 200) -> pd.DataFrame:
    with Session(get_engine()) as session:
        rows = session.execute(
            select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
        ).scalars().all()
    return pd.DataFrame(
        [
            {
                "id": row.id,
                "probability": row.fraud_probability,
                "predicted_label": row.predicted_label,
                "review_status": row.review_status,
                "created_at": row.created_at,
            }
            for row in rows
        ]
    )


@st.cache_data(ttl=60)
def load_etl_runs(limit: int = 50) -> pd.DataFrame:
    with Session(get_engine()) as session:
        rows = session.execute(select(EtlRun).order_by(EtlRun.started_at.desc()).limit(limit)).scalars().all()
    return pd.DataFrame(
        [
            {
                "run_id": row.id,
                "status": row.status,
                "rows_extracted": row.rows_extracted,
                "rows_loaded": row.rows_loaded,
                "rows_rejected": row.rows_rejected,
                "started_at": row.started_at,
            }
            for row in rows
        ]
    )


@st.cache_data(ttl=60)
def load_model_versions() -> pd.DataFrame:
    with Session(get_engine()) as session:
        rows = session.execute(select(ModelVersion)).scalars().all()
    return pd.DataFrame(
        [{"version": r.version, "status": r.status, "threshold": r.threshold, "created_at": r.created_at} for r in rows]
    )


def show_verdict(result: dict) -> None:
    """Display the FraudLens verdict banner plus prediction details."""
    if result.get("predicted_label") == 1:
        st.error("🔴 Fraud Alert")
    else:
        st.success("🟢 Transaction Approved")

    probability = result.get("fraud_probability", 0.0)
    st.progress(min(max(probability, 0.0), 1.0), text=f"fraud probability: {probability:.2%}")

    with st.expander("Details"):
        st.json(
            {
                "source": result.get("source"),
                "risk_level": result.get("risk_level"),
                "status": result.get("status"),
                "model_version": result.get("model_version"),
                "threshold": result.get("threshold"),
                "prediction_id": result.get("prediction_id"),
                "ocr_engine": result.get("ocr_engine"),
                "extracted_fields": result.get("extracted_fields"),
                "imputed_features": result.get("imputed_features"),
            },
            expanded=True,
        )


def fraud_check_page() -> None:
    st.subheader("Fraud Check")
    st.caption(
        "Enter a transaction as text or upload a screenshot. Amount and Type are required "
        "in the text; any balance fields the model needs are filled from training medians."
    )

    tab_text, tab_image = st.tabs(["📝 Text input", "🖼️ Image upload"])

    with tab_text:
        default = "amount=2350, type=TRANSFER, oldbalanceOrg=12000, newbalanceOrig=0"
        text_input = st.text_area("Transaction text", value=default, height=100)
        if st.button("Check transaction", key="text_check", type="primary"):
            if not text_input.strip():
                st.warning("Enter the transaction details first.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/predictions/text",
                        json={"text": text_input},
                        timeout=TIMEOUT,
                    )
                except requests.ConnectionError:
                    st.error(f"Cannot reach the FraudLens API at {API_URL}. Start it with: uvicorn fraudlens.api.main:app")
                else:
                    if response.status_code == 201:
                        show_verdict(response.json())
                    elif response.status_code == 503:
                        st.error("No active model is deployed. Train one first: fraudlens-train")
                    else:
                        st.error(f"API error {response.status_code}: {response.text}")

    with tab_image:
        image_file = st.file_uploader(
            "Transaction screenshot (png, jpg)", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"]
        )
        if st.button("Scan image", key="image_check", type="primary") and image_file is not None:
            try:
                response = requests.post(
                    f"{API_URL}/predictions/image",
                    files={"file": (image_file.name, image_file.getvalue())},
                    timeout=TIMEOUT,
                )
            except requests.ConnectionError:
                st.error(f"Cannot reach the FraudLens API at {API_URL}. Start it with: uvicorn fraudlens.api.main:app")
            else:
                if response.status_code == 201:
                    show_verdict(response.json())
                elif response.status_code == 503:
                    st.error("No active model is deployed. Train one first: fraudlens-train")
                elif response.status_code == 422:
                    st.warning(f"Could not process the image: {response.json().get('detail', '')}")
                else:
                    st.error(f"API error {response.status_code}: {response.text}")


def dashboard_page() -> None:
    st.title("FraudLens")
    st.caption("Transaction risk detection, ETL and monitoring")

    overview = load_overview()
    columns = st.columns(4)
    columns[0].metric("Predictions", f"{overview['predictions']:,}")
    columns[1].metric("Flagged for review", f"{overview['flagged']:,}")
    columns[2].metric("Confirmed fraud", f"{overview['confirmed_fraud']:,}")
    columns[3].metric("Flag rate", f"{overview['flag_rate']:.2%}")

    predictions = load_recent_predictions()
    etl_runs = load_etl_runs()
    versions = load_model_versions()

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Recent predictions")
        if predictions.empty:
            st.info("No predictions recorded yet.")
        else:
            st.dataframe(predictions, use_container_width=True)
            st.bar_chart(predictions.set_index("created_at")["probability"])

    with right:
        st.subheader("ETL runs")
        if etl_runs.empty:
            st.info("No ETL runs recorded yet.")
        else:
            st.dataframe(etl_runs, use_container_width=True)

        st.subheader("Model versions")
        if versions.empty:
            st.info("No model versions registered yet.")
        else:
            st.dataframe(versions, use_container_width=True)


def main() -> None:
    page = st.sidebar.radio("FraudLens", ["Dashboard", "Fraud Check"], label_visibility="collapsed")
    if page == "Fraud Check":
        fraud_check_page()
    else:
        dashboard_page()


if __name__ == "__main__":
    main()
