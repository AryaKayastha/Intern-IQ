"""
etl/warehouse.py
----------------
Builds the DuckDB Silver (star schema) and Gold (aggregation) layers
from cleaned DataFrames produced by clean.py.

Also generates a data-quality report printed to stdout.

Run order:
  python etl/ingest.py     ← writes Bronze tables
  python etl/warehouse.py  ← builds Silver + Gold (calls ingest + clean internally)
"""

import os
import hashlib
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, os.getenv("DUCKDB_PATH", "db/intern_platform.duckdb"))


# ── helpers to generate surrogate keys ──────────────────────────────────────
def _id(val: str) -> str:
    return hashlib.sha256(str(val).strip().lower().encode()).hexdigest()[:10]


# ============================================================================
# SILVER LAYER
# ============================================================================

def build_silver(con: duckdb.DuckDBPyConnection,
                 eod_clean: pd.DataFrame,
                 lms_map: dict[str, pd.DataFrame],
                 mentor_df: pd.DataFrame) -> None:
    """Write all Silver dimension + fact tables."""

    print("\n=== Silver Layer ===")

    # ── dim_intern ──────────────────────────────────────────────────────────
    # Gather unique interns from EOD + all LMS
    eod_interns = eod_clean[["intern_id", "full_name", "first_name", "last_name"]].drop_duplicates()
    lms_interns = pd.concat([
        df[["intern_id", "full_name"]] for df in lms_map.values()
    ]).drop_duplicates(subset=["intern_id"], keep="first")
    lms_interns["first_name"] = lms_interns["full_name"].str.split().str[0]
    lms_interns["last_name"]  = lms_interns["full_name"].str.split().str[1:].str.join(" ")

    dim_intern = (
        pd.concat([eod_interns, lms_interns], ignore_index=True)
        .drop_duplicates(subset=["intern_id"], keep="first")
        .reset_index(drop=True)
    )
    con.execute("DROP TABLE IF EXISTS dim_intern")
    con.execute("CREATE TABLE dim_intern AS SELECT * FROM dim_intern")
    print(f"  [SUCCESS] dim_intern: {len(dim_intern)} interns")

    # ── dim_course ──────────────────────────────────────────────────────────
    course_rows = []
    for key, df in lms_map.items():
        course_name = df["course_name"].iloc[0] if "course_name" in df.columns else key
        start = df["start_date"].min() if "start_date" in df.columns else None
        end   = df["end_date"].max()   if "end_date"   in df.columns else None
        course_rows.append({
            "course_id":   _id(course_name),
            "course_key":  key,
            "course_name": course_name,
            "start_date":  start,
            "end_date":    end,
        })
    dim_course = pd.DataFrame(course_rows)
    con.execute("DROP TABLE IF EXISTS dim_course")
    con.execute("CREATE TABLE dim_course AS SELECT * FROM dim_course")
    print(f"  [SUCCESS] dim_course: {len(dim_course)} courses")

    # ── dim_mentor ──────────────────────────────────────────────────────────
    unique_mentors = mentor_df[["mentor_name"]].drop_duplicates()
    unique_mentors["mentor_id"] = unique_mentors["mentor_name"].apply(_id)
    dim_mentor = unique_mentors[["mentor_id", "mentor_name"]].reset_index(drop=True)
    con.execute("DROP TABLE IF EXISTS dim_mentor")
    con.execute("CREATE TABLE dim_mentor AS SELECT * FROM dim_mentor")
    print(f"  [SUCCESS] dim_mentor: {len(dim_mentor)} mentors")

    # ── dim_activity ────────────────────────────────────────────────────────
    activity_names = eod_clean["activity"].dropna().unique()
    activity_category_map = {
        "PySpark Session":           "PySpark",
        "NumPy Practice":            "NumPy/Pandas",
        "Power BI Dashboard Work":   "BI/Reporting",
        "PL/SQL Concepts":           "SQL",
        "SQL Revision":              "SQL",
        "Data Engineering Course":   "Data Engineering",
        "Pandas Practice":           "NumPy/Pandas",
        "Pandas Exam Preparation":   "NumPy/Pandas",
        "PySpark LMS Learning":      "PySpark",
        "Advanced SQL Practice":     "SQL",
        "Spark Architecture Study":  "PySpark",
        "Project Research":          "Research",
    }
    dim_activity = pd.DataFrame({
        "activity_id":       [_id(a) for a in activity_names],
        "activity_name":     list(activity_names),
        "activity_category": [activity_category_map.get(a, "Other") for a in activity_names],
    })
    con.execute("DROP TABLE IF EXISTS dim_activity")
    con.execute("CREATE TABLE dim_activity AS SELECT * FROM dim_activity")
    print(f"  [SUCCESS] dim_activity: {len(dim_activity)} activity types")

    # ── dim_date ────────────────────────────────────────────────────────────
    all_dates = eod_clean["date"].dropna().dt.date.unique()
    dim_date = pd.DataFrame({
        "date":         all_dates,
        "day_of_week":  [pd.Timestamp(d).day_name() for d in all_dates],
        "week_number":  [pd.Timestamp(d).isocalendar()[1] for d in all_dates],
        "month":        [pd.Timestamp(d).month for d in all_dates],
        "month_name":   [pd.Timestamp(d).strftime("%B") for d in all_dates],
        "year":         [pd.Timestamp(d).year for d in all_dates],
    })
    con.execute("DROP TABLE IF EXISTS dim_date")
    con.execute("CREATE TABLE dim_date AS SELECT * FROM dim_date")
    print(f"  [SUCCESS] dim_date: {len(dim_date)} distinct dates")

    # ── bridge_intern_mentor ─────────────────────────────────────────────────
    bridge = mentor_df.merge(
        dim_mentor, on="mentor_name", how="left"
    ).merge(
        dim_course[["course_key", "course_id"]], on="course_key", how="left"
    )[["intern_id", "course_id", "mentor_id"]].drop_duplicates()
    con.execute("DROP TABLE IF EXISTS bridge_intern_mentor")
    con.execute("CREATE TABLE bridge_intern_mentor AS SELECT * FROM bridge")
    print(f"  [SUCCESS] bridge_intern_mentor: {len(bridge)} rows")

    # ── fact_eod_log ────────────────────────────────────────────────────────
    fact_eod = eod_clean.merge(
        dim_activity[["activity_id", "activity_name"]].rename(columns={"activity_name": "activity"}),
        on="activity", how="left"
    )
    fact_eod["log_id"] = range(len(fact_eod))
    fact_eod["date_only"] = fact_eod["date"].dt.date
    fact_eod = fact_eod[[
        "log_id", "intern_id", "date_only", "activity_id",
        "hours", "hours_outlier_flag"
    ]].rename(columns={"date_only": "date"})
    con.execute("DROP TABLE IF EXISTS fact_eod_log")
    con.execute("CREATE TABLE fact_eod_log AS SELECT * FROM fact_eod")
    print(f"  [SUCCESS] fact_eod_log: {len(fact_eod)} rows")

    # ── fact_lms_progress ───────────────────────────────────────────────────
    lms_all = []
    for key, df in lms_map.items():
        tmp = df.copy()
        tmp["course_key"] = key
        lms_all.append(tmp)
    lms_concat = pd.concat(lms_all, ignore_index=True)
    fact_lms = lms_concat.merge(
        dim_course[["course_id", "course_key"]], on="course_key", how="left"
    )
    fact_lms["progress_id"] = range(len(fact_lms))
    keep_cols = [
        "progress_id", "intern_id", "course_id",
        "progress_pct",
        "completed_count", "total_assignments", "assignment_ratio",
        "kc_score", "kc_max", "kc_pct",
        "test_score", "test_max", "test_pct",
        "overall_status",
    ]
    existing = [c for c in keep_cols if c in fact_lms.columns]
    fact_lms = fact_lms[existing]
    con.execute("DROP TABLE IF EXISTS fact_lms_progress")
    con.execute("CREATE TABLE fact_lms_progress AS SELECT * FROM fact_lms")
    print(f"  [SUCCESS] fact_lms_progress: {len(fact_lms)} rows")


# ============================================================================
# GOLD LAYER
# ============================================================================

def build_gold(con: duckdb.DuckDBPyConnection) -> None:
    """Build all Gold aggregation tables via SQL on Silver tables."""

    print("\n=== Gold Layer ===")

    # ── gold_weekly_hours ────────────────────────────────────────────────────
    con.execute("DROP TABLE IF EXISTS gold_weekly_hours")
    con.execute("""
        CREATE TABLE gold_weekly_hours AS
        SELECT
            f.intern_id,
            i.full_name,
            d.year,
            d.week_number,
            d.month_name,
            COUNT(*)             AS activity_count,
            SUM(f.hours)         AS total_hours,
            AVG(f.hours)         AS avg_hours_per_activity,
            COUNT(DISTINCT f.date) AS active_days
        FROM fact_eod_log f
        JOIN dim_intern i ON f.intern_id = i.intern_id
        JOIN dim_date   d ON f.date = d.date
        GROUP BY f.intern_id, i.full_name, d.year, d.week_number, d.month_name
        ORDER BY f.intern_id, d.year, d.week_number
    """)
    n = con.execute("SELECT COUNT(*) FROM gold_weekly_hours").fetchone()[0]
    print(f"  [SUCCESS] gold_weekly_hours: {n} rows")

    # ── gold_activity_summary ────────────────────────────────────────────────
    con.execute("DROP TABLE IF EXISTS gold_activity_summary")
    con.execute("""
        CREATE TABLE gold_activity_summary AS
        SELECT
            f.intern_id,
            i.full_name,
            a.activity_name,
            a.activity_category,
            COUNT(*)          AS activity_count,
            SUM(f.hours)      AS total_hours,
            AVG(f.hours)      AS avg_hours
        FROM fact_eod_log f
        JOIN dim_intern  i ON f.intern_id  = i.intern_id
        JOIN dim_activity a ON f.activity_id = a.activity_id
        GROUP BY f.intern_id, i.full_name, a.activity_name, a.activity_category
    """)
    n = con.execute("SELECT COUNT(*) FROM gold_activity_summary").fetchone()[0]
    print(f"  [SUCCESS] gold_activity_summary: {n} rows")

    # ── gold_course_progress ─────────────────────────────────────────────────
    con.execute("DROP TABLE IF EXISTS gold_course_progress")
    con.execute("""
        CREATE TABLE gold_course_progress AS
        SELECT
            f.intern_id,
            i.full_name,
            c.course_name,
            f.progress_pct,
            f.overall_status,
            f.assignment_ratio,
            f.kc_pct,
            f.test_pct,
            f.completed_count,
            f.total_assignments
        FROM fact_lms_progress f
        JOIN dim_intern i ON f.intern_id = i.intern_id
        JOIN dim_course c ON f.course_id  = c.course_id
    """)
    n = con.execute("SELECT COUNT(*) FROM gold_course_progress").fetchone()[0]
    print(f"  [SUCCESS] gold_course_progress: {n} rows")

    # ── gold_intern_performance ──────────────────────────────────────────────
    con.execute("DROP TABLE IF EXISTS gold_intern_performance")
    con.execute("""
        CREATE TABLE gold_intern_performance AS
        SELECT
            i.intern_id,
            i.full_name,
            COALESCE(e.total_hours,     0)   AS total_hours,
            COALESCE(e.activity_count,  0)   AS total_activity_entries,
            COALESCE(e.distinct_activities, 0) AS distinct_activities,
            COALESCE(l.avg_progress,    0)   AS avg_progress_pct,
            COALESCE(l.avg_kc_pct,      0)   AS avg_kc_pct,
            COALESCE(l.avg_test_pct,    0)   AS avg_test_pct,
            COALESCE(l.courses_completed, 0) AS courses_completed,
            COALESCE(l.avg_assignment_ratio, 0) AS avg_assignment_ratio
        FROM dim_intern i
        LEFT JOIN (
            SELECT
                intern_id,
                SUM(hours)              AS total_hours,
                COUNT(*)                AS activity_count,
                COUNT(DISTINCT activity_id) AS distinct_activities
            FROM fact_eod_log
            GROUP BY intern_id
        ) e ON i.intern_id = e.intern_id
        LEFT JOIN (
            SELECT
                intern_id,
                AVG(progress_pct)       AS avg_progress,
                AVG(kc_pct)             AS avg_kc_pct,
                AVG(test_pct)           AS avg_test_pct,
                COUNT(CASE WHEN overall_status = 'Completed' THEN 1 END) AS courses_completed,
                AVG(assignment_ratio)   AS avg_assignment_ratio
            FROM fact_lms_progress
            GROUP BY intern_id
        ) l ON i.intern_id = l.intern_id
    """)
    n = con.execute("SELECT COUNT(*) FROM gold_intern_performance").fetchone()[0]
    print(f"  [SUCCESS] gold_intern_performance: {n} rows")

    # ── gold_mentor_workload ─────────────────────────────────────────────────
    con.execute("DROP TABLE IF EXISTS gold_mentor_workload")
    con.execute("""
        CREATE TABLE gold_mentor_workload AS
        SELECT
            m.mentor_id,
            m.mentor_name,
            c.course_name,
            COUNT(DISTINCT b.intern_id)          AS intern_count,
            AVG(f.progress_pct)                  AS avg_mentee_progress,
            SUM(CASE WHEN f.overall_status = 'Completed' THEN 1 ELSE 0 END) AS completed_count,
            SUM(CASE WHEN f.overall_status = 'Not Started' THEN 1 ELSE 0 END) AS not_started_count
        FROM bridge_intern_mentor b
        JOIN dim_mentor  m ON b.mentor_id  = m.mentor_id
        JOIN dim_course  c ON b.course_id  = c.course_id
        JOIN fact_lms_progress f ON b.intern_id = f.intern_id AND b.course_id = f.course_id
        GROUP BY m.mentor_id, m.mentor_name, c.course_name
    """)
    n = con.execute("SELECT COUNT(*) FROM gold_mentor_workload").fetchone()[0]
    print(f"  [SUCCESS] gold_mentor_workload: {n} rows")


# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def print_quality_report(con: duckdb.DuckDBPyConnection,
                          eod_clean: pd.DataFrame,
                          lms_map: dict[str, pd.DataFrame]) -> None:
    print("\n=== Data Quality Report ===")

    # Row counts per layer
    tables = {
        "Bronze": ["raw_eod_activities", "raw_lms_python", "raw_lms_sql", "raw_lms_numpy_pandas", "raw_lms_pyspark"],
        "Silver": ["dim_intern", "dim_course", "dim_mentor", "dim_activity",
                   "fact_eod_log", "fact_lms_progress"],
        "Gold":   ["gold_weekly_hours", "gold_activity_summary",
                   "gold_course_progress", "gold_intern_performance", "gold_mentor_workload"],
    }
    for layer, tbl_list in tables.items():
        print(f"\n  {layer} layer:")
        for t in tbl_list:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"    {t}: {n} rows")
            except Exception:
                print(f"    {t}: (not available)")

    # Row counts
    data_source_dist = con.execute("""
        SELECT COUNT(*) AS cnt
        FROM raw_eod_activities
    """).df()
    print("\n  EOD total rows:")
    for _, row in data_source_dist.iterrows():
        print(f"    Total: {row['cnt']} rows")

    # Outlier flags
    n_outliers = con.execute(
        "SELECT COUNT(*) FROM fact_eod_log WHERE hours_outlier_flag = 1"
    ).fetchone()[0]
    print(f"\n  Hours outliers (outside 0.5–4.0): {n_outliers}")

    # Null counts in fact_lms_progress key columns
    null_report = con.execute("""
        SELECT
            COUNT(*) - COUNT(test_pct)   AS null_test_pct,
            COUNT(*) - COUNT(kc_pct)     AS null_kc_pct,
            COUNT(*) - COUNT(progress_pct) AS null_progress_pct
        FROM fact_lms_progress
    """).df()
    print(f"\n  fact_lms_progress nulls:")
    for col in null_report.columns:
        print(f"    {col}: {null_report[col].iloc[0]}")


# ============================================================================
# MAIN
# ============================================================================

def run_warehouse() -> None:
    from etl.ingest import run_ingestion
    from etl.clean  import clean_eod, clean_lms, explode_mentors

    # Step 1 – Ingest (Bronze)
    run_ingestion()
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    # Load complete Bronze layer for Silver/Gold generation
    eod_raw = con.execute("SELECT * FROM raw_eod_activities").df()
    
    lms_raw = {}
    for key in ["python", "sql", "numpy_pandas", "pyspark"]:
        lms_raw[key] = con.execute(f"SELECT * FROM raw_lms_{key}").df()

    # Step 2 – Clean
    print("\n=== Cleaning Data ===")
    eod_clean = clean_eod(eod_raw)
    print(f"  [SUCCESS] EOD cleaned: {len(eod_clean)} rows")

    COURSE_NAMES = {
        "python":       "Basic Python Programming",
        "sql":          "Basic SQL",
        "numpy_pandas": "Data Processing using NumPy & Pandas",
        "pyspark":      "Data Processing using PySpark",
    }
    lms_map: dict[str, pd.DataFrame] = {}
    for key, df in lms_raw.items():
        cleaned = clean_lms(df, course_name_override=COURSE_NAMES[key])
        cleaned["course_key"] = key
        lms_map[key] = cleaned
        print(f"  [SUCCESS] LMS [{key}] cleaned: {len(cleaned)} rows")

    # Explode mentors
    mentor_parts = [
        explode_mentors(df).assign(course_key=key)
        for key, df in lms_map.items()
    ]
    mentor_df = pd.concat(mentor_parts, ignore_index=True)

    # Step 3 – Silver + Gold
    build_silver(con, eod_clean, lms_map, mentor_df)
    build_gold(con)
    print_quality_report(con, eod_clean, lms_map)
    con.close()
    print("\n[SUCCESS] Warehouse build complete.\n")


if __name__ == "__main__":
    run_warehouse()
