# src/train_models.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import settings
from .databricks_db import query_as_dicts

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_training_data() -> pd.DataFrame:
    """
    Load training data from Gold.fact_trips (or equivalent).
    Adjust fq_table if your name is different.
    """
    fq_table = f"{settings.dbx_catalog}.gold.fact_trips"
    sql = f"SELECT * FROM {fq_table}"
    rows = query_as_dicts(sql)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows returned from Gold.fact_trips for training.")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X, y_class, y_reg from Gold table.
    Expects columns:
      - distance_km, estimated_time_min
      - delay_minutes
      - is_late
      - traffic, weather, warehouse
    """

    required_cols = [
        "distance_km",
        "estimated_time_min",
        "delay_minutes",
        "is_late",
        "traffic",
        "weather",
        "warehouse",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in Gold table for ML: {missing}")

    df = df.dropna(subset=required_cols).copy()

    df["distance_km"] = df["distance_km"].astype(float)
    df["estimated_time_min"] = df["estimated_time_min"].astype(float)
    df["delay_minutes"] = df["delay_minutes"].astype(float)
    df["is_late"] = df["is_late"].astype(int)  # 0/1

    feature_cols_num = ["distance_km", "estimated_time_min"]
    feature_cols_cat = ["traffic", "weather", "warehouse"]

    X = df[feature_cols_num + feature_cols_cat]
    y_class = df["is_late"]
    y_reg = df["delay_minutes"]

    return X, y_class, y_reg


def train_all_models() -> Dict[str, Any]:
    """
    Train classifier (late vs on-time) + regressor (delay minutes),
    save them to models/, and return metrics.
    """

    df = load_training_data()
    X, y_class, y_reg = prepare_features(df)

    numeric_features = ["distance_km", "estimated_time_min"]
    categorical_features = ["traffic", "weather", "warehouse"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # ----------------- CLASSIFIER -----------------
    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    clf_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "support": int(len(y_test)),
    }

    joblib.dump(clf, MODEL_DIR / "late_classifier.joblib")

    # ----------------- REGRESSOR -----------------
    reg = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    random_state=42,
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                ),
            ),
        ]
    )

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg.fit(Xr_train, yr_train)

    yr_pred = reg.predict(Xr_test)

    reg_metrics = {
        "mae": float(mean_absolute_error(yr_test, yr_pred)),
        "r2": float(r2_score(yr_test, yr_pred)),
        "support": int(len(yr_test)),
    }

    joblib.dump(reg, MODEL_DIR / "eta_regressor.joblib")

    return {
        "classifier": clf_metrics,
        "regressor": reg_metrics,
        "model_paths": {
            "late_classifier": str(MODEL_DIR / "late_classifier.joblib"),
            "eta_regressor": str(MODEL_DIR / "eta_regressor.joblib"),
        },
    }


if __name__ == "__main__":
    metrics = train_all_models()
    print("Training finished.")
    print(metrics)