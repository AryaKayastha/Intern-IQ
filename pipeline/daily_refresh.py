"""
pipeline/daily_refresh.py
-------------------------
Orchestrates the incremental daily refresh of the InternIQ platform.
Runs:
  1. ingest.py (upserts new Bronze rows)
  2. warehouse.py (rebuilds Silver & Gold layers using complete Bronze data)
  3. embeddings.py (updates the GenAI vector store with the latest summaries)
  4. train.py (optionally retrains ML models — set to run once a week)

Can be run manually for an immediate refresh, or left running to trigger automatically
every day at exactly 6:00 PM (18:00).
"""

import os
import sys
import time
import schedule
import subprocess
from datetime import datetime

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_CMD = sys.executable

def run_script(script_path: str, name: str) -> bool:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏳ Starting: {name}...")
    try:
        subprocess.run([PYTHON_CMD, os.path.join(BASE_DIR, script_path)], check=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Success: {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ ERROR: {name} failed with exit code {e.returncode}")
        return False

def daily_refresh_job():
    print(f"\n{'='*50}")
    print(f"🔄 INITIATING DAILY REFRESH: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # 1. Ingest New Data (Upsert to Bronze) & 2. Rebuild Warehouse (Silver + Gold)
    # Note: warehouse.py automatically calls ingest.py locally first
    if not run_script(os.path.join("etl", "warehouse.py"), "ETL Pipeline (Bronze → Silver → Gold)"):
        return

    # 4. Retrain ML Models (Optional daily, but usually good to keep fresh)
    if not run_script(os.path.join("ml", "train.py"), "ML Model Retraining"):
        return

    print(f"\n🎉 DAILY REFRESH COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="InternIQ Daily Refresh Orchestrator")
    parser.add_argument("--now", action="store_true", help="Run the refresh immediately once, instead of scheduling.")
    args = parser.parse_args()

    if args.now:
        daily_refresh_job()
        sys.exit(0)

    # Schedule the job every day at 18:00 (6:00 PM)
    target_time = "18:00"
    schedule.every().day.at(target_time).do(daily_refresh_job)

    print(f"\n🕒 Scheduler started! The pipeline will refresh automatically every day at {target_time}.")
    print("   Keep this script running in the background. Press Ctrl+C to exit.\n")

    # Infinite loop to keep the scheduler checking
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every 60 seconds
    except KeyboardInterrupt:
        print("\n⏹️ Scheduler stopped by user.")
