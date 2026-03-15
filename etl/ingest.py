"""
etl/ingest.py
-------------
Load Excel files.
Writes DataFrames to the Bronze layer in DuckDB.
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
RAW_DIR  = os.path.join(BASE_DIR, "dataset", "raw")
SYN_DIR  = os.path.join(BASE_DIR, "dataset", "synthesize")
DB_PATH  = os.path.join(BASE_DIR, os.getenv("DUCKDB_PATH", "db/intern_platform.duckdb"))

# ---------------------------------------------------------------------------
# File mapping
# ---------------------------------------------------------------------------
EOD_FILE_RAW = "intern_eod_last3months_random (1).xlsx"
EOD_FILE_SYN = "intern_eod_synthetic.xlsx"

LMS_FILES_RAW = {
    "python":       "assignment_submissions_progress_Basic_Python_Programming.xlsx",
    "sql":          "assignment_submissions_progress_Basic_SQL.xlsx",
    "numpy_pandas": "assignment_submissions_progress_Data_Processing_using_NumPy_Pa.xlsx",
    "pyspark":      "assignment_submissions_progress_Data_Processing_using_Pyspark.xlsx",
}

LMS_FILES_SYN = {
    "python":       "assignment_submissions_progress_synthetic_python.xlsx",
    "sql":          "assignment_submissions_progress_synthetic_sql.xlsx",
    "numpy_pandas": "assignment_submissions_progress_synthetic_numpy.xlsx",
    "pyspark":      "assignment_submissions_progress_synthetic_pyspark.xlsx",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_excel(directory: str, filename: str) -> pd.DataFrame:
    path = os.path.join(directory, filename)
    return pd.read_excel(path, engine="openpyxl")


def ingest_eod() -> pd.DataFrame:
    """Read EOD logs."""
    raw_df = read_excel(RAW_DIR, EOD_FILE_RAW)
    syn_df = read_excel(SYN_DIR, EOD_FILE_SYN)
    df = pd.concat([raw_df, syn_df], ignore_index=True)
    print(f"  EOD loaded: {len(df)} rows")
    return df


def ingest_lms(course_key: str) -> pd.DataFrame:
    """Read LMS data."""
    raw_file = LMS_FILES_RAW[course_key]
    syn_file = LMS_FILES_SYN[course_key]
    raw_df = read_excel(RAW_DIR, raw_file)
    syn_df = read_excel(SYN_DIR, syn_file)
    df = pd.concat([raw_df, syn_df], ignore_index=True)
    print(f"  LMS [{course_key}] loaded: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Write to DuckDB Bronze layer
# ---------------------------------------------------------------------------
def write_bronze(con: duckdb.DuckDBPyConnection,
                 df: pd.DataFrame,
                 table_name: str) -> None:
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if table_name not in tables:
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        print(f"  [SUCCESS] Bronze table '{table_name}' created ({len(df)} rows)")
    else:
        con.execute(f"CREATE TEMP TABLE staging AS SELECT * FROM df")
        # Upsert: Insert rows from staging that aren't already in the target table
        res = con.execute(f"INSERT INTO {table_name} SELECT * FROM staging EXCEPT SELECT * FROM {table_name}")
        inserted_count = res.fetchone()[0] if res.description else "unknown"
        con.execute("DROP TABLE staging")
        print(f"  [SUCCESS] Bronze table '{table_name}' upserted ({inserted_count} new rows added)")


def run_ingestion() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)

    print("\n=== Bronze Layer Ingestion ===")

    # EOD
    eod_df = ingest_eod()
    write_bronze(con, eod_df, "raw_eod_activities")

    # LMS per course
    lms_dfs: dict[str, pd.DataFrame] = {}
    for key in LMS_FILES_RAW:
        df = ingest_lms(key)
        table_name = f"raw_lms_{key}"
        write_bronze(con, df, table_name)
        lms_dfs[key] = df

    con.close()
    print("\n[SUCCESS] Bronze ingestion complete.\n")


if __name__ == "__main__":
    run_ingestion()
