"""Shared constants and model/artifact loading helpers for the Streamlit app."""
import os

import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CATEGORICAL_COLUMNS = [
    "Body Type",
    "Sex",
    "Diet",
    "How Often Shower",
    "Heating Energy Source",
    "Transport",
    "Vehicle Type",
    "Social Activity",
    "Frequency of Traveling by Air",
    "Waste Bag Size",
    "Energy efficiency",
]

RECYCLING_OPTIONS = ["Paper", "Plastic", "Glass", "Metal"]
COOKING_OPTIONS = ["Stove", "Oven", "Microwave", "Grill", "Airfryer"]

WASTE_BAG_SIZE_MAP = {"small": 1, "medium": 2, "large": 3, "extra large": 4}
VEHICLE_IMPACT_MAP = {
    "petrol": 2.3,
    "diesel": 2.7,
    "hybrid": 1.5,
    "lpg": 1.8,
    "electric": 0.5,
    "None": 0,
}

GRADE_THRESHOLDS = [
    (2000, "A", "#10B981"),
    (4000, "B", "#84CC16"),
    (6000, "C", "#FACC15"),
    (8000, "D", "#F97316"),
    (float("inf"), "F", "#EF4444"),
]

CLUSTER_COLORS = {
    "Eco Warrior": "#10B981",
    "Green Beginner": "#59B896",
    "Average Consumer": "#FACC15",
    "High Emitter": "#EF4444",
}


class ModelLoadError(Exception):
    """Raised when required model artifacts are missing from models/."""


def _encoder_filename(column: str) -> str:
    return f"le_{column.replace(' ', '_')}.pkl"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler, encoders, kmeans, cluster labels and feature names.

    Returns a dict with keys: model, scaler, kmeans, cluster_labels,
    feature_names, encoders (dict keyed by column name).
    """
    required_files = {
        "model": "carbon_model.pkl",
        "scaler": "scaler.pkl",
        "kmeans": "kmeans_model.pkl",
        "cluster_labels": "cluster_labels.pkl",
        "feature_names": "feature_names.pkl",
    }
    encoder_files = {col: _encoder_filename(col) for col in CATEGORICAL_COLUMNS}

    missing = [
        fname
        for fname in {**required_files, **encoder_files}.values()
        if not os.path.exists(os.path.join(MODELS_DIR, fname))
    ]
    if missing:
        raise ModelLoadError(
            "Missing model artifact(s): "
            + ", ".join(missing)
            + ". Please run the notebook first to generate models "
            "(notebooks/model_training.ipynb) before launching the app."
        )

    artifacts = {
        key: joblib.load(os.path.join(MODELS_DIR, fname))
        for key, fname in required_files.items()
    }
    artifacts["encoders"] = {
        col: joblib.load(os.path.join(MODELS_DIR, fname))
        for col, fname in encoder_files.items()
    }
    return artifacts


def get_dropdown_options(encoders, column):
    """Return the valid category options for a column, as seen during training."""
    return list(encoders[column].classes_)


def get_vehicle_type_options(encoders):
    """Vehicle Type options excluding the placeholder 'None' (no vehicle)."""
    return [opt for opt in get_dropdown_options(encoders, "Vehicle Type") if opt != "None"]


@st.cache_data(show_spinner=False)
def load_cleaned_dataset():
    """Load the cleaned dataset (post Phase-1 cleaning), or None if absent."""
    path = os.path.join(DATA_DIR, "carbon_emission_cleaned.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[])


@st.cache_data(show_spinner=False)
def get_dataset_average_emission():
    """Mean CarbonEmission across the cleaned dataset, or None if unavailable."""
    df = load_cleaned_dataset()
    if df is None or df.empty:
        return None
    return float(df["CarbonEmission"].mean())


@st.cache_data(show_spinner=False)
def load_cleaned_dataset():
    """Load the cleaned dataset used for training, for stats/comparisons in the UI."""
    path = os.path.join(DATA_DIR, "carbon_emission_cleaned.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=[])


@st.cache_data(show_spinner=False)
def get_dataset_average_emission():
    """Mean CarbonEmission across the cleaned dataset, or None if unavailable."""
    df = load_cleaned_dataset()
    if df is None or "CarbonEmission" not in df.columns:
        return None
    return float(df["CarbonEmission"].mean())
