"""
ml/predict.py
-------------
Inference functions used by the Streamlit dashboard (Tab 6 — ML Insights).
Loads serialized models from ml/models/ and provides:
  - classify_intern_status(features_dict) → status label + probabilities
  - cluster_intern(features_dict)         → cluster label
  - predict_test_score(features_dict)     → predicted test score %
  - load_all_results()                    → pre-computed DataFrames for dashboard
"""

import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")


def _load(filename: str):
    return joblib.load(os.path.join(MODELS_DIR, filename))


# ── Cached loaders (lazy) ─────────────────────────────────────────────────────
_classifier     = None
_label_encoder  = None
_kmeans_pipeline = None
_cluster_label_map = None
_regressor      = None


def get_classifier():
    global _classifier, _label_encoder
    if _classifier is None:
        _classifier    = _load("classifier.pkl")
        _label_encoder = _load("label_encoder.pkl")
    return _classifier, _label_encoder


def get_kmeans():
    global _kmeans_pipeline, _cluster_label_map
    if _kmeans_pipeline is None:
        _kmeans_pipeline   = _load("kmeans_pipeline.pkl")
        _cluster_label_map = _load("cluster_label_map.pkl")
    return _kmeans_pipeline, _cluster_label_map


def get_regressor():
    global _regressor
    if _regressor is None:
        _regressor = _load("regressor.pkl")
    return _regressor


# ── Public API ────────────────────────────────────────────────────────────────

def classify_intern_status(progress_pct: float, assignment_ratio: float,
                           kc_pct: float, test_pct: float) -> dict:
    """Returns predicted status label + class probabilities."""
    clf, le = get_classifier()
    X = np.array([[progress_pct, assignment_ratio, kc_pct, test_pct]])
    label_idx = clf.predict(X)[0]
    probas    = clf.predict_proba(X)[0]
    return {
        "status": le.inverse_transform([label_idx])[0],
        "probabilities": dict(zip(le.classes_, probas.round(3))),
    }


def cluster_intern(total_hours: float, distinct_activities: float,
                   avg_progress: float, avg_assignment_ratio: float) -> str:
    """Returns cluster label string: High Performer / Average Performer / At Risk."""
    pipeline, label_map = get_kmeans()
    X = np.array([[total_hours, distinct_activities, avg_progress, avg_assignment_ratio]])
    raw = pipeline.predict(X)[0]
    return label_map.get(raw, "Unknown")


def predict_test_score(progress_pct: float, kc_pct: float,
                       assignment_ratio: float) -> float:
    """Returns predicted test score as a percentage."""
    reg = get_regressor()
    X = np.array([[progress_pct, kc_pct, assignment_ratio]])
    return float(reg.predict(X)[0])


def load_all_results() -> dict[str, pd.DataFrame]:
    """Load pre-computed result DataFrames for dashboard display."""
    results = {}
    for name in ["cluster_results", "regression_results", "pca_results"]:
        path = os.path.join(MODELS_DIR, f"{name}.parquet")
        if os.path.exists(path):
            results[name] = pd.read_parquet(path)
    return results
