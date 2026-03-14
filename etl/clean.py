"""
etl/clean.py
------------
Data cleaning and transformation for EOD and LMS DataFrames.
Produces cleaned DataFrames ready for the Silver layer.
"""

import hashlib
import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Lowercase, strip extra spaces, used for intern_id and joins."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def make_intern_id(full_name: str) -> str:
    """SHA-256 hash (first 10 hex chars) of normalised full name."""
    return hashlib.sha256(normalize_name(full_name).encode()).hexdigest()[:10]


def parse_fraction(value) -> tuple[float | None, float | None, float | None]:
    """
    Parse '3/3' or '443/470' → (numerator, denominator, ratio).
    Returns (None, None, None) for missing / '-' values.
    """
    if pd.isna(value) or str(value).strip() in ("-", "", "nan"):
        return None, None, None
    s = str(value).strip()
    match = re.match(r"^([\d.]+)\s*/\s*([\d.]+)$", s)
    if match:
        num = float(match.group(1))
        den = float(match.group(2))
        ratio = (num / den * 100) if den > 0 else None
        return num, den, ratio
    # Might already be a plain number
    try:
        return float(s), None, None
    except ValueError:
        return None, None, None


# ---------------------------------------------------------------------------
# EOD cleaning
# ---------------------------------------------------------------------------

def clean_eod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: Date, First Name, Last Name, Activity, Hours, data_source
    Returns cleaned DataFrame with intern_id, full_name, parsed dates, validated hours.
    """
    df = df.copy()

    # --- Names ---
    df["First Name"] = df["First Name"].astype(str).str.strip()
    df["Last Name"]  = df["Last Name"].astype(str).str.strip()
    df["full_name"]  = df["First Name"] + " " + df["Last Name"]
    df["intern_id"]  = df["full_name"].apply(make_intern_id)

    # --- Date ---
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    invalid_dates = df["date"].isna().sum()
    if invalid_dates:
        print(f"  ⚠ EOD: {invalid_dates} rows with unparseable dates set to NaT")

    # --- Hours ---
    df["hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    outlier_mask = (df["hours"] < 0.5) | (df["hours"] > 4.0)
    df["hours_outlier_flag"] = outlier_mask.astype(int)
    n_outliers = outlier_mask.sum()
    if n_outliers:
        print(f"  ⚠ EOD: {n_outliers} hours outliers flagged (outside 0.5–4.0 range)")

    # --- Activity ---
    df["activity"] = df["Activity"].astype(str).str.strip()

    # --- Drop raw columns, keep clean ones ---
    clean_cols = [
        "intern_id", "full_name", "First Name", "Last Name",
        "date", "activity", "hours", "hours_outlier_flag", "data_source"
    ]
    return df[clean_cols].rename(columns={"First Name": "first_name", "Last Name": "last_name"})


# ---------------------------------------------------------------------------
# LMS cleaning
# ---------------------------------------------------------------------------

def clean_lms(df: pd.DataFrame, course_name_override: str | None = None) -> pd.DataFrame:
    """
    Input columns: User Name, Course Name, Start Date, End Date, Mentor Name,
                   Progress (%), Completed Assignment, Reviewed / Submitted,
                   Overall Knowledge Check, Overall Test, Reviewed / Total Assignments,
                   Overall Status, data_source
    Returns cleaned DataFrame with parsed numeric fields and intern_id.
    """
    df = df.copy()

    # --- Intern ID ---
    df["full_name"] = df["User Name"].astype(str).str.strip()
    df["intern_id"] = df["full_name"].apply(make_intern_id)

    # --- Course name ---
    if course_name_override:
        df["course_name"] = course_name_override
    else:
        df["course_name"] = df["Course Name"].astype(str).str.strip()

    # --- Dates ---
    for col, new_col in [("Start Date", "start_date"), ("End Date", "end_date")]:
        if col in df.columns:
            df[new_col] = pd.to_datetime(df[col], errors="coerce")

    # --- Progress (%) → integer ---
    if "Progress (%)" in df.columns:
        df["progress_pct"] = (
            df["Progress (%)"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # --- Completed Assignment: "3/3" → completed_count, total_assignments ---
    if "Completed Assignment" in df.columns:
        parsed = df["Completed Assignment"].apply(parse_fraction)
        df["completed_count"]    = [p[0] for p in parsed]
        df["total_assignments"]  = [p[1] for p in parsed]
        df["assignment_ratio"]   = [p[2] for p in parsed]

    # --- Overall Knowledge Check: "443/470" → kc_score, kc_max, kc_pct ---
    if "Overall Knowledge Check" in df.columns:
        parsed = df["Overall Knowledge Check"].apply(parse_fraction)
        df["kc_score"] = [p[0] for p in parsed]
        df["kc_max"]   = [p[1] for p in parsed]
        df["kc_pct"]   = [p[2] for p in parsed]

    # --- Overall Test: "25/40" or "-" → test_score, test_max, test_pct ---
    if "Overall Test" in df.columns:
        parsed = df["Overall Test"].apply(parse_fraction)
        df["test_score"] = [p[0] for p in parsed]
        df["test_max"]   = [p[1] for p in parsed]
        df["test_pct"]   = [p[2] for p in parsed]

    # --- Reviewed / Submitted ---
    for col_name, score_col, max_col, pct_col in [
        ("Reviewed / Submitted",         "reviewed",  "submitted",  "reviewed_pct"),
        ("Reviewed / Total Assignments",  "reviewed2", "total2",     "reviewed2_pct"),
    ]:
        if col_name in df.columns:
            parsed = df[col_name].apply(parse_fraction)
            df[score_col] = [p[0] for p in parsed]
            df[max_col]   = [p[1] for p in parsed]
            df[pct_col]   = [p[2] for p in parsed]

    # --- Overall Status normalisation ---
    if "Overall Status" in df.columns:
        status_map = {
            "completed":   "Completed",
            "in progress": "In Progress",
            "not started": "Not Started",
        }
        df["overall_status"] = (
            df["Overall Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(status_map)
            .fillna("Unknown")
        )

    # --- Mentor Name: keep raw, explode later in warehouse ---
    if "Mentor Name" in df.columns:
        df["mentor_name_raw"] = df["Mentor Name"].astype(str).str.strip()

    clean_cols = [
        "intern_id", "full_name", "course_name",
        "start_date", "end_date",
        "progress_pct",
        "completed_count", "total_assignments", "assignment_ratio",
        "kc_score", "kc_max", "kc_pct",
        "test_score", "test_max", "test_pct",
        "reviewed", "submitted", "reviewed_pct",
        "overall_status",
        "mentor_name_raw",
        "data_source",
    ]
    # Only keep columns that were actually created
    existing = [c for c in clean_cols if c in df.columns]
    return df[existing]


# ---------------------------------------------------------------------------
# Mentor explode helper (used by warehouse.py)
# ---------------------------------------------------------------------------

def explode_mentors(lms_clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand comma-separated mentor names into separate rows.
    Returns DataFrame with columns: intern_id, course_name, mentor_name (single name).
    """
    df = lms_clean_df[["intern_id", "course_name", "mentor_name_raw"]].copy()
    df["mentor_list"] = df["mentor_name_raw"].str.split(",")
    df = df.explode("mentor_list")
    df["mentor_name"] = df["mentor_list"].astype(str).str.strip()
    df = df[df["mentor_name"] != ""].drop(columns=["mentor_list", "mentor_name_raw"])
    return df.drop_duplicates().reset_index(drop=True)
