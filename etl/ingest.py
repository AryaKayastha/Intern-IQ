"""
etl/ingest.py
-------------
Load and merge real + synthetic Excel files.
Writes merged DataFrames to the Bronze layer in DuckDB.

Merge rules (from project plan):
  EOD  : simple concat — every row is a unique daily activity entry
  LMS  : keep all real rows, add only synthetic interns not already present
          (drop_duplicates on 'User Name', keeping first/real occurrence)
"""

import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "dataset", "raw")
SYNTH_DIR = os.path.join(BASE_DIR, "dataset", "synthesize")
DB_PATH   = os.path.join(BASE_DIR, os.getenv("DUCKDB_PATH", "db/intern_platform.duckdb"))

# ---------------------------------------------------------------------------
# File mapping  (real filename → synthetic filename)
# ---------------------------------------------------------------------------
EOD_REAL   = "intern_eod_last3months_random (1).xlsx"
EOD_SYNTH  = "intern_eod_synthetic.xlsx"

LMS_FILES = {
    "python":      ("assignment_submissions_progress_Basic Python Programming.xlsx",
                    "assignment_submissions_progress_synthetic_python.xlsx"),
    "sql":         ("assignment_submissions_progress_Basic SQL (1).xlsx",
                    "assignment_submissions_progress_synthetic_sql.xlsx"),
    "numpy_pandas":("assignment_submissions_progress_Data Processing using NumPy  Pa.xlsx",
                    "assignment_submissions_progress_synthetic_numpy.xlsx"),
    "pyspark":     ("assignment_submissions_progress_Data Processing using Pyspark.xlsx",
                    "assignment_submissions_progress_synthetic_pyspark.xlsx"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_excel(directory: str, filename: str) -> pd.DataFrame:
    path = os.path.join(directory, filename)
    return pd.read_excel(path, engine="openpyxl")


def merge_eod() -> pd.DataFrame:
    """Simple concat of real + synthetic EOD logs."""
    real  = read_excel(RAW_DIR,   EOD_REAL);  real["data_source"]  = "real"
    synth = read_excel(SYNTH_DIR, EOD_SYNTH); synth["data_source"] = "synthetic"
    merged = pd.concat([real, synth], ignore_index=True)
    print(f"  EOD merged: {len(real)} real + {len(synth)} synthetic = {len(merged)} rows")
    return merged


def merge_lms(course_key: str) -> pd.DataFrame:
    """Keep all real rows; add only synthetic interns absent from real file."""
    real_file, synth_file = LMS_FILES[course_key]
    real  = read_excel(RAW_DIR,   real_file);  real["data_source"]  = "real"
    synth = read_excel(SYNTH_DIR, synth_file); synth["data_source"] = "synthetic"

    # Normalise User Name for comparison
    real_names  = set(real["User Name"].str.strip().str.lower())
    new_only    = synth[~synth["User Name"].str.strip().str.lower().isin(real_names)]
    merged      = pd.concat([real, new_only], ignore_index=True)
    print(f"  LMS [{course_key}]: {len(real)} real + {len(new_only)} new synthetic = {len(merged)} rows")
    return merged


# ---------------------------------------------------------------------------
# Write to DuckDB Bronze layer
# ---------------------------------------------------------------------------
def write_bronze(con: duckdb.DuckDBPyConnection,
                 df: pd.DataFrame,
                 table_name: str) -> None:
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if table_name not in tables:
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        print(f"  ✔ Bronze table '{table_name}' created ({len(df)} rows)")
    else:
        con.execute(f"CREATE TEMP TABLE staging AS SELECT * FROM df")
        # Upsert: Insert rows from staging that aren't already in the target table
        res = con.execute(f"INSERT INTO {table_name} SELECT * FROM staging EXCEPT SELECT * FROM {table_name}")
        inserted_count = res.fetchone()[0] if res.description else "unknown"
        con.execute("DROP TABLE staging")
        print(f"  ✔ Bronze table '{table_name}' upserted ({inserted_count} new rows added)")


def run_ingestion() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    print("\n=== Bronze Layer Ingestion ===")

    # EOD
    eod_df = merge_eod()
    write_bronze(con, eod_df, "raw_eod_activities")

    # LMS per course
    lms_dfs: dict[str, pd.DataFrame] = {}
    for key in LMS_FILES:
        df = merge_lms(key)
        table_name = f"raw_lms_{key}"
        write_bronze(con, df, table_name)
        lms_dfs[key] = df

    con.close()
    print("\n✅ Bronze ingestion complete.\n")


if __name__ == "__main__":
    run_ingestion()
