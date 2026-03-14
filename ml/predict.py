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
_classifier      = None
_label_encoder   = None
_clf_features    = None

_kmeans_pipeline = None
_cluster_label_map = None
_km_features     = None

_regressor       = None
_reg_features    = None


def get_classifier():
    global _classifier, _label_encoder, _clf_features
    if _classifier is None:
        _classifier    = _load("classifier.pkl")
        _label_encoder = _load("label_encoder.pkl")
        _clf_features  = _load("classifier_features.pkl")
    return _classifier, _label_encoder, _clf_features


def get_kmeans():
    global _kmeans_pipeline, _cluster_label_map, _km_features
    if _kmeans_pipeline is None:
        _kmeans_pipeline   = _load("kmeans_pipeline.pkl")
        _cluster_label_map = _load("cluster_label_map.pkl")
        _km_features       = _load("clustering_features.pkl")
    return _kmeans_pipeline, _cluster_label_map, _km_features


def get_regressor():
    global _regressor, _reg_features
    if _regressor is None:
        _regressor    = _load("regressor.pkl")
        _reg_features = _load("regressor_features.pkl")
    return _regressor, _reg_features

# ── Dynamic Input Builder ─────────────────────────────────────────────────────

def build_feature_array(kwargs_dict: dict, feature_list: list) -> np.ndarray:
    """Safely construct a 1D feature array matching the exact training shape."""
    row = []
    for f in feature_list:
        val = kwargs_dict.get(f, 0.0)
        row.append(float(val))
    return np.array([row])



# ── Public API ────────────────────────────────────────────────────────────────

def classify_intern_status(**kwargs) -> dict:
    """Returns predicted status label + class probabilities.
    Requires domain hours, completion states, and course dummy indicator."""
    clf, le, features = get_classifier()
    X = build_feature_array(kwargs, features)
    label_idx = clf.predict(X)[0]
    probas    = clf.predict_proba(X)[0]
    return {
        "status": le.inverse_transform([label_idx])[0],
        "probabilities": dict(zip(le.classes_, probas.round(3))),
    }


def cluster_intern(**kwargs) -> str:
    """Returns cluster label string: High Performer / Average Performer / At Risk.
    Requires total hours, domain hours, distinct activities, avg progress/assignment."""
    pipeline, label_map, features = get_kmeans()
    X = build_feature_array(kwargs, features)
    raw = pipeline.predict(X)[0]
    return label_map.get(raw, "Unknown")


def predict_test_score(**kwargs) -> float:
    """Returns predicted test score as a percentage.
    Requires domain hours, kc_pct, assignment_ratio, and course dummy indicator."""
    reg, features = get_regressor()
    X = build_feature_array(kwargs, features)
    return float(reg.predict(X)[0])


def load_all_results() -> dict[str, pd.DataFrame]:
    """Load pre-computed result DataFrames for dashboard display."""
    results = {}
    for name in ["cluster_results", "regression_results", "pca_results"]:
        path = os.path.join(MODELS_DIR, f"{name}.parquet")
        if os.path.exists(path):
            results[name] = pd.read_parquet(path)
    return results
