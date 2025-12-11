# src/ml_model.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
LATE_PATH = MODEL_DIR / "late_classifier.joblib"
ETA_PATH = MODEL_DIR / "eta_regressor.joblib"


def _load_models():
    """
    Load models fresh from disk on each call.
    Slower, but ensures Streamlit training + FastAPI prediction stay in sync
    without restarting the server.
    """
    clf = joblib.load(LATE_PATH)
    reg = joblib.load(ETA_PATH)
    return clf, reg


def predict_delay(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    features should contain at least:
      - distance_km
      - estimated_time_min
      - traffic
      - weather
      - warehouse

    Extra keys are ignored by the Pipeline's ColumnTransformer.
    """

    clf, reg = _load_models()
    df = pd.DataFrame([features])

    proba = clf.predict_proba(df)[0, 1]
    is_late = bool(proba >= 0.5)
    label = "Late" if is_late else "On-time"

    eta_delay = float(reg.predict(df)[0])  # minutes +/-

    return {
        "prediction": label,
        "probability_late": float(proba),
        "eta_delay_minutes": eta_delay,
    }