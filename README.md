# InternIQ Analytics Platform

> Intern Activity Intelligence & Productivity Analytics Platform

---

## Quick Start

### Option 1: Local (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ETL pipeline (builds DuckDB warehouse)
python -m etl.warehouse

# 3. Train ML models
python -m ml.train

# 4. (Optional) Build GenAI vector store — needs internet for model download
python -m genai.embeddings

# 5. Launch dashboard
streamlit run app/streamlit_app.py
# Visit http://localhost:8501
```

### Option 2: Docker (all services)

```bash
docker compose up --build
# Visit http://localhost:8501
```

---

## Project Structure

```
InternIQ/
├── dataset/
│   ├── raw/           ← 5 real Excel files
│   └── synthesize/    ← 5 synthetic Excel files
├── etl/
│   ├── ingest.py      ← Merge real + synthetic → Bronze layer
│   ├── clean.py       ← Parse dates, fractions, %, names
│   └── warehouse.py   ← Bronze → Silver → Gold (DuckDB)
├── ml/
│   ├── train.py       ← Train 3 ML models (RF/KMeans/Ridge)
│   ├── predict.py     ← Inference functions
│   └── models/        ← Saved .pkl + .parquet files
├── genai/
│   ├── embeddings.py  ← Chunk + embed Gold tables → ChromaDB
│   └── chatbot.py     ← LangChain RAG + Ollama (Llama3)
├── app/
│   └── streamlit_app.py  ← 7-tab dashboard
├── db/
│   └── intern_platform.duckdb
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── to_do.md           ← Progress tracker
```

---

## Architecture — Medallion Data Warehouse

```
Raw Excel (10 files: 5 real + 5 synthetic)
         ↓ etl/ingest.py
BRONZE   → raw_eod_activities (6,330 rows)
           raw_lms_{python,sql,numpy_pandas,pyspark} (60 rows each)
         ↓ etl/clean.py + warehouse.py
SILVER   → dim_intern (61), dim_course (4), dim_mentor (26),
           dim_activity (12), dim_date (72)
           fact_eod_log (6,330), fact_lms_progress (240)
         ↓ warehouse.py SQL aggregations
GOLD     → gold_weekly_hours, gold_activity_summary,
           gold_course_progress, gold_intern_performance,
           gold_mentor_workload
```

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Data Quality | Row counts per layer, real/synthetic split, nulls, outliers |
| 🔍 EDA | Hours histogram, activity breakdown, weekly heatmap |
| 👔 Manager | 61-intern leaderboard, top/bottom 5, activity breakdown |
| 👨‍🏫 Mentor | Mentee progress heatmap, at-risk list, KC heatmap |
| 🎯 Intern | Personal stats, course progress bars, test vs batch avg |
| 🤖 ML Insights | PCA cluster scatter, regression plot, live prediction widget |
| 💬 GenAI Chat | Natural language Q&A over intern data (requires Ollama) |

---

## ML Models

| Model | Algorithm | Target | Metric |
|-------|-----------|--------|--------|
| Performance Classifier | Random Forest | Overall Status | 100% accuracy (real rows) |
| Intern Clustering | K-Means (k=3) | Work pattern group | 29 High / 22 Avg / 10 At-Risk |
| Test Score Predictor | Ridge Regression | Test Score % | MAE: 18.7% |

---

## GenAI Chatbot — Sample Queries

- *"Who hasn't started PySpark yet?"*
- *"Which intern logged the most hours this week?"*
- *"Show me top performers in SQL"*
- *"Which interns are at risk of not completing NumPy?"*
- *"Summarize batch progress across all courses"*

> **Note:** Run `python -m genai.embeddings` once, and have Ollama running with `llama3` pulled.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data Warehouse | DuckDB (medallion architecture) |
| ETL | Python + Pandas |
| Dashboard | Streamlit + Plotly |
| ML | scikit-learn (RF, KMeans, Ridge) |
| Embeddings | sentence-transformers (MiniLM-L6) |
| Vector Store | ChromaDB |
| LLM | Ollama + Llama3 |
| Containerization | Docker Compose |
