# InternIQ Platform — Progress Tracker

> **Project:** Intern Activity Intelligence & Productivity Analytics Platform  
> **Hackathon:** Kenexai 2k26 — CHARUSAT, Changa | 24-Hour Round 2  
> **Last Updated:** 2026-03-14

---

## ✅ Module 0: Data Analysis & Synthesis — DONE
- [x] Analyzed all 5 original Excel files
- [x] Generated synthetic data (20 new interns, Feb–Mar 2026)
- [x] Validated synthetic files match schema

---

## ✅ Module 1: Project Structure Setup — DONE
- [x] Created project folder structure (`etl/`, `ml/`, `genai/`, `app/`, `db/`, `data/`)
- [x] Created `requirements.txt`
- [x] Created `.env` configuration file
- [x] Created `Dockerfile`
- [x] Created `docker-compose.yml`

---

## ✅ Module 2: ETL Pipeline — DONE
- [x] `etl/ingest.py` — Load & merge real + synthetic files
  - [x] EOD: simple concat with `data_source` column → **6,330 rows**
  - [x] LMS: deduplication (keep real rows) → **60 rows per course**
- [x] `etl/clean.py` — All data cleaning transforms
  - [x] Parsed date strings, created intern_id, parsed fractions, stripped %, handled NULLs
  - [x] Exploded mentor names, normalized intern names
- [x] `etl/warehouse.py` — Bronze → Silver → Gold pipeline

---

## ✅ Module 3: DuckDB Data Warehouse — DONE
- [x] **Bronze:** `raw_eod_activities` (6,330 rows), `raw_lms_*` (60 rows × 4 courses)
- [x] **Silver:** `dim_intern` (61), `dim_course` (4), `dim_mentor` (26), `dim_activity` (12), `dim_date` (72), `fact_eod_log` (6,330), `fact_lms_progress` (240)
- [x] **Gold:** `gold_weekly_hours` (380), `gold_activity_summary` (712), `gold_course_progress` (240), `gold_intern_performance` (61), `gold_mentor_workload` (49)

---

## ✅ Module 4: Streamlit Dashboard (7 Tabs) — DONE
- [x] Tab 1: Data Quality Dashboard — layer row counts, real/synthetic split, null counts, outlier flags
- [x] Tab 2: EDA Dashboard — hours histogram, activity bars, weekly heatmap, course box plots, status stacked bars
- [x] Tab 3: Manager View — 61-intern leaderboard, top/bottom 5, activity breakdown, batch weekly trend
- [x] Tab 4: Mentor View — mentee progress heatmap, at-risk list, KC score heatmap, not-started list
- [x] Tab 5: Intern View — personal KPIs, hours per activity, course progress bars, test vs batch avg, weekly trend
- [x] Tab 6: ML Insights — PCA cluster scatter, cluster pie, regression scatter, cluster results table, live prediction widget
- [x] Tab 7: GenAI Chatbot — chat UI, history, source expansion, unavailable message if embeddings not run

> ⚡ **Running at:** http://localhost:8501

---

## ✅ Module 5: ML Models — DONE
- [x] Model 1: Random Forest Classifier — **100% accuracy** on real-only test rows
- [x] Model 2: K-Means Clustering (k=3) — 29 High / 22 Average / 10 At-Risk performers
- [x] Model 3: Ridge Regression — MAE 18.7% on test scores
- [x] All models saved to `ml/models/` as `.pkl` + `.parquet`

---

## ✅ Module 6: GenAI RAG Chatbot — DONE (pending embeddings)
- [x] `genai/embeddings.py` — Chunk generator + ChromaDB persistence (run once to activate)
- [x] `genai/chatbot.py` — LangChain RAG chain + Ollama/Llama3 with graceful fallback
- [ ] Run `python -m genai.embeddings` to activate vector store (needs sentence-transformers)

---

## ✅ Module 7: Docker Compose Deployment — DONE (files ready)
- [x] `Dockerfile` — Python 3.11-slim + Streamlit
- [x] `docker-compose.yml` — App + Ollama services with volume mounts
- [x] `README.md` — Setup instructions
- [ ] End-to-end `docker compose up --build` test (requires Docker on host)

---

## Definition of Done — Status
- [x] Streamlit app loads at `http://localhost:8501`
- [x] All 7 dashboard tabs render with real DuckDB data
- [x] ML models load and display predictions
- [ ] GenAI chatbot live (run `python -m genai.embeddings` + `ollama pull llama3`)
- [x] DuckDB persists on disk (`db/intern_platform.duckdb`)
- [x] All 10 Excel files fully processed, visible in Gold layer
