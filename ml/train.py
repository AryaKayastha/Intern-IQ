"""
ml/train.py
-----------
Trains 3 ML models on the merged (real + synthetic) Gold layer data:
  Model 1: Random Forest Classifier → predict Overall Status
  Model 2: K-Means Clustering (k=3) → segment interns by work pattern
  Model 3: Ridge Regression         → predict test score %

Serializes each model to ml/models/  using joblib.
Validates classifer + regressor on real-only rows.
"""

import os
import sys
import warnings
import duckdb
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, mean_absolute_error
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, os.getenv("DUCKDB_PATH", "db/intern_platform.duckdb"))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# Data loading from Gold layer
# ============================================================================

def load_data(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        perf_df  : Silver layer aggregations (one row per intern, richly dimensional)
        prog_df  : Silver layer course facts + domain hours
    """
    # 1. Course Progress Features (prog_df)
    prog_df = con.execute("""
        SELECT 
            f.intern_id,
            i.full_name,
            c.course_name,
            f.progress_pct, 
            f.assignment_ratio,
            f.kc_pct, 
            f.test_pct,
            f.overall_status,
            f.data_source
        FROM fact_lms_progress f
        JOIN dim_course c ON f.course_id = c.course_id
        JOIN dim_intern i ON f.intern_id = i.intern_id
    """).df()

    # Pre-calculate domain hours per intern from fact_eod_log + dim_activity
    domain_hours = con.execute("""
        SELECT 
            f.intern_id,
            SUM(CASE WHEN a.activity_category = 'SQL' THEN f.hours ELSE 0 END) AS hours_sql,
            SUM(CASE WHEN a.activity_category = 'PySpark' THEN f.hours ELSE 0 END) AS hours_pyspark,
            SUM(CASE WHEN a.activity_category = 'NumPy/Pandas' THEN f.hours ELSE 0 END) AS hours_numpy,
            SUM(CASE WHEN a.activity_category = 'BI/Reporting' THEN f.hours ELSE 0 END) AS hours_bi
        FROM fact_eod_log f
        JOIN dim_activity a ON f.activity_id = a.activity_id
        GROUP BY f.intern_id
    """).df()
    
    # Merge domain hours so classifiers/regressors can correlate domain practice with passing a course
    prog_df = prog_df.merge(domain_hours, on="intern_id", how="left").fillna(0)

    # 2. Intern Clustering Features (perf_df)
    perf_df = con.execute("""
        SELECT 
            i.intern_id,
            i.full_name,
            i.data_source,
            COALESCE(SUM(f.hours), 0) AS total_hours,
            COUNT(DISTINCT f.activity_id) AS distinct_activities,
            SUM(CASE WHEN d.day_of_week IN ('Saturday', 'Sunday') THEN f.hours ELSE 0 END) AS weekend_hours
        FROM dim_intern i
        LEFT JOIN fact_eod_log f ON i.intern_id = f.intern_id
        LEFT JOIN dim_date d ON f.date = d.date
        GROUP BY i.intern_id, i.full_name, i.data_source
    """).df()
    
    # Merge domain hours directly into the performance df for clustering
    perf_df = perf_df.merge(domain_hours, on="intern_id", how="left").fillna(0)

    # Add basic LMS aggregations for perf_df
    prog_agg = con.execute("""
        SELECT 
            intern_id,
            AVG(progress_pct) AS avg_progress_pct,
            AVG(assignment_ratio) AS avg_assignment_ratio,
            AVG(kc_pct) AS avg_kc_pct,
            AVG(test_pct) AS avg_test_pct,
            COUNT(CASE WHEN overall_status = 'Completed' THEN 1 END) AS courses_completed,
            COUNT(f.course_id) AS total_activity_entries
        FROM fact_lms_progress f
        GROUP BY intern_id
    """).df()
    
    perf_df = perf_df.merge(prog_agg, on="intern_id", how="left").fillna(0)
    
    return perf_df, prog_df



# ============================================================================
# Model 1 — Performance Classifier (Random Forest)
# ============================================================================

def train_classifier(prog_df: pd.DataFrame) -> dict:
    """
    Features: progress_pct, assignment_ratio, kc_pct, test_pct (from gold_course_progress)
    Target  : overall_status (Completed / In Progress / Not Started)
    Validate: on real-only rows
    """
    print("\n--- Model 1: Performance Classifier ---")
    df = prog_df.copy()

    feature_cols = [
        "progress_pct", "assignment_ratio", "kc_pct", "test_pct",
        "hours_sql", "hours_pyspark", "hours_numpy", "hours_bi"
    ]
    
    # One-hot encode course_name
    df = pd.get_dummies(df, columns=["course_name"], drop_first=False, dtype=float)
    
    # Combine original features + new dummy columns
    course_dummy_cols = [c for c in df.columns if c.startswith("course_name_")]
    feature_cols.extend(course_dummy_cols)

    # Fill nulls with 0 for features
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0)

    # Encode target
    le = LabelEncoder()
    df["status_enc"] = le.fit_transform(df["overall_status"])

    X = df[feature_cols].values
    y = df["status_enc"].values

    # Train on ALL data (merged)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
    ])
    clf.fit(X, y)

    # Evaluate on real-only rows
    real_df = df[df["data_source"] == "real"].copy()
    if len(real_df) > 5:
        X_real  = real_df[feature_cols].values
        y_real  = real_df["status_enc"].values
        y_pred  = clf.predict(X_real)
        acc     = accuracy_score(y_real, y_pred)
        print(f"  Accuracy on real-only rows: {acc:.2%}")
        print(f"  Classes: {le.classes_.tolist()}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_real, y_pred)}")
    else:
        acc = accuracy_score(y, clf.predict(X))
        print(f"  Accuracy (all data): {acc:.2%}")

    # Save
    joblib.dump(clf, os.path.join(MODELS_DIR, "classifier.pkl"))
    joblib.dump(le,  os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "classifier_features.pkl"))
    print("  Saved: ml/models/classifier.pkl, label_encoder.pkl, classifier_features.pkl")
    return {"model": clf, "label_encoder": le, "accuracy": acc, "feature_cols": feature_cols}


# ============================================================================
# Model 2 — Intern Clustering (K-Means, k=3)
# ============================================================================

def train_clustering(perf_df: pd.DataFrame) -> dict:
    """
    Features: total_hours, distinct_activities, avg_progress_pct, avg_assignment_ratio
    Output  : cluster labels added to intern performance table
    """
    print("\n--- Model 2: Intern Clustering (K-Means k=3) ---")
    df = perf_df.copy()

    feature_cols = [
        "total_hours", "distinct_activities", "avg_progress_pct", "avg_assignment_ratio",
        "hours_sql", "hours_pyspark", "hours_numpy", "hours_bi", "weekend_hours"
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("kmeans",  KMeans(n_clusters=3, random_state=42, n_init=10)),
    ])
    pipeline.fit(X)

    df["cluster_raw"] = pipeline.predict(X)

    # Label clusters by total_hours centroid (descending order → 0=High, 1=Average, 2=At Risk)
    cluster_means = df.groupby("cluster_raw")["total_hours"].mean().sort_values(ascending=False)
    cluster_label_map = {
        cluster_means.index[0]: "High Performer",
        cluster_means.index[1]: "Average Performer",
        cluster_means.index[2]: "At Risk",
    }
    df["cluster_label"] = df["cluster_raw"].map(cluster_label_map)

    # Show cluster distribution
    dist = df.groupby("cluster_label")[["total_hours", "avg_progress_pct"]].mean().round(2)
    print(f"  Cluster distribution:\n{df['cluster_label'].value_counts().to_string()}")
    print(f"  Cluster averages:\n{dist.to_string()}")

    # Save pipeline + label map + result df
    joblib.dump(pipeline,          os.path.join(MODELS_DIR, "kmeans_pipeline.pkl"))
    joblib.dump(cluster_label_map, os.path.join(MODELS_DIR, "cluster_label_map.pkl"))
    joblib.dump(feature_cols,      os.path.join(MODELS_DIR, "clustering_features.pkl"))

    # Save cluster assignments for use in dashboard
    cluster_result = df[["intern_id", "full_name", "cluster_raw", "cluster_label",
                          "total_hours", "distinct_activities",
                          "avg_progress_pct", "avg_assignment_ratio",
                          "avg_kc_pct", "avg_test_pct",
                          "total_activity_entries", "courses_completed", "data_source"]].copy()
    cluster_result.to_parquet(os.path.join(MODELS_DIR, "cluster_results.parquet"), index=False)
    print("  Saved: ml/models/kmeans_pipeline.pkl, ml/models/cluster_results.parquet")
    return {"pipeline": pipeline, "cluster_label_map": cluster_label_map, "cluster_df": cluster_result}


# ============================================================================
# Model 3 — Test Score Regression (Ridge)
# ============================================================================

def train_regression(prog_df: pd.DataFrame) -> dict:
    """
    Features: progress_pct, kc_pct, assignment_ratio
    Target  : test_pct (overall test score %)
    Validate: real-only rows
    """
    print("\n--- Model 3: Test Score Regressor (Ridge) ---")
    df = prog_df.dropna(subset=["test_pct"]).copy()
    print(f"  Rows with test scores: {len(df)} (of {len(prog_df)})")

    feature_cols = [
        "progress_pct", "kc_pct", "assignment_ratio",
        "hours_sql", "hours_pyspark", "hours_numpy", "hours_bi"
    ]
    
    original_course_name = df["course_name"].copy()
    
    df = pd.get_dummies(df, columns=["course_name"], drop_first=False, dtype=float)
    df["course_name"] = original_course_name
    
    course_dummy_cols = [c for c in df.columns if c.startswith("course_name_")]
    feature_cols.extend(course_dummy_cols)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols].values
    y = df["test_pct"].values

    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0)),
    ])
    reg.fit(X, y)

    # Evaluate on real-only rows
    real_df = df[df["data_source"] == "real"].copy()
    if len(real_df) > 3:
        X_r = real_df[feature_cols].values
        y_r = real_df["test_pct"].values
        y_p = reg.predict(X_r)
        r2  = r2_score(y_r, y_p)
        mae = mean_absolute_error(y_r, y_p)
        print(f"  Real-only R²: {r2:.3f}  |  MAE: {mae:.2f}%")
    else:
        y_p = reg.predict(X)
        r2  = r2_score(y, y_p)
        mae = mean_absolute_error(y, y_p)
        print(f"  R² (all data): {r2:.3f}  |  MAE: {mae:.2f}%")

    # Save predictions alongside actuals for dashboard
    df["predicted_test_pct"] = reg.predict(X)
    pred_df = df[["intern_id", "full_name", "course_name",
                  "test_pct", "predicted_test_pct", "data_source"]].copy()
    pred_df.to_parquet(os.path.join(MODELS_DIR, "regression_results.parquet"), index=False)

    joblib.dump(reg, os.path.join(MODELS_DIR, "regressor.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "regressor_features.pkl"))
    print("  Saved: ml/models/regressor.pkl, regressor_features.pkl, ml/models/regression_results.parquet")
    return {"model": reg, "r2": r2, "mae": mae, "feature_cols": feature_cols}


# ============================================================================
# PCA helper (for 2-D cluster scatter plot in dashboard)
# ============================================================================

def compute_pca(perf_df: pd.DataFrame, cluster_df: pd.DataFrame) -> None:
    from sklearn.decomposition import PCA

    feature_cols = [
        "total_hours", "distinct_activities", "avg_progress_pct", "avg_assignment_ratio",
        "hours_sql", "hours_pyspark", "hours_numpy", "hours_bi", "weekend_hours"
    ]
    df = perf_df[["intern_id"] + [c for c in feature_cols if c in perf_df.columns]].copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(df[feature_cols])
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]

    pca_df = df[["intern_id", "pca_x", "pca_y"]].merge(
        cluster_df[["intern_id", "cluster_label", "data_source", "full_name"]],
        on="intern_id", how="left"
    )
    pca_df.to_parquet(os.path.join(MODELS_DIR, "pca_results.parquet"), index=False)
    print("  Saved: ml/models/pca_results.parquet")


# ============================================================================
# MAIN
# ============================================================================

def run_training() -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    perf_df, prog_df = load_data(con)
    con.close()

    print(f"\nLoaded: {len(perf_df)} intern performance rows, {len(prog_df)} course progress rows\n")

    clf_result = train_classifier(prog_df)
    clust_result = train_clustering(perf_df)
    reg_result  = train_regression(prog_df)
    compute_pca(perf_df, clust_result["cluster_df"])

    print("\n✅ All models trained and saved to ml/models/\n")


if __name__ == "__main__":
    run_training()
