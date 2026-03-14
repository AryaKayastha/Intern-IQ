# 📋 Intern Activity Intelligence & Productivity Analytics Platform
### Kenexai Hackathon 2k26 — CHARUSAT, Changa | 24-Hour Round 2

---

## 📌 Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset Description](#2-dataset-description)
3. [Approach to Solution](#3-approach-to-solution)
4. [System Architecture](#4-system-architecture)
5. [Tech Stack Decisions](#5-tech-stack-decisions)
6. [Module Breakdown & Todo List](#6-module-breakdown--todo-list)
7. [Data Warehouse Design](#7-data-warehouse-design)
8. [ML Use Cases](#8-ml-use-cases)
9. [GenAI Use Cases](#9-genai-use-cases)
10. [Personas & KPIs](#10-personas--kpis)
11. [Docker Deployment Plan](#11-docker-deployment-plan)
12. [24-Hour Execution Timeline](#12-24-hour-execution-timeline)

---

## 1. Problem Statement

### Background
Organizations running internship or training programs collect daily activity reports from interns through collaboration platforms or internal reporting systems. These reports contain:
- Tasks performed each day
- Technologies used (Python, SQL, PySpark, NumPy, Pandas, etc.)
- Time spent on different activities
- Progress recorded in learning management systems (LMS)
- Assignment submission and review status

### The Problem
This data is **unstructured, scattered across multiple files, and siloed** — making it nearly impossible for managers, mentors, and HR teams to:
- Track intern productivity at scale
- Identify underperforming or at-risk interns early
- Compare progress across cohorts and courses
- Get a unified view of activity + LMS performance
- Ask natural language questions about intern data

### What We Are Building
A **fully automated, end-to-end data + AI platform** that:
1. Ingests raw intern data (EOD activity logs + LMS course progress) from both real and synthetic sources
2. Cleans, transforms, and stores it in a structured data warehouse
3. Surfaces insights via persona-specific dashboards
4. Predicts intern performance using ML models trained on the merged dataset
5. Allows natural language querying via a GenAI RAG chatbot
6. Is fully containerized and deployable with a single command

---

## 2. Dataset Description

### Data Strategy: Real + Synthetic Merged

All 5 input files are provided in **two versions** — original (real) and synthetic — which are merged during ingestion:

| File | Real Rows | Synthetic Rows | Merged Total | Interns |
|------|-----------|----------------|--------------|---------|
| `intern_eod_last3months_random.xlsx` | 4,379 | 1,951 | **~6,330** | 40 |
| `assignment_submissions_progress_Basic_Python_Programming.xlsx` | 39 | 21 new | **60** | 60 |
| `assignment_submissions_progress_Basic_SQL.xlsx` | 39 | 21 new | **60** | 60 |
| `assignment_submissions_progress_Data_Processing_using_NumPy_Pa.xlsx` | 28 | 32 new | **60** | 60 |
| `assignment_submissions_progress_Data_Processing_using_Pyspark.xlsx` | 27 | 33 new | **60** | 60 |

> **Merge Rule for LMS files:** Original rows are always preserved as ground truth. Only interns not present in the original file are added from the synthetic file (`drop_duplicates(subset=['User Name'])` keeping first/original occurrence).
>
> **Merge Rule for EOD file:** Simple concatenation — all rows from both files are kept since each row is a unique daily activity entry.
>
> **data_source flag:** A `data_source` column (`'real'` / `'synthetic'`) is added at ingestion time to allow model validation splits that never leak synthetic labels into test evaluation.

### Source Files

#### File 1: `intern_eod_last3months_random.xlsx` + `intern_eod_synthetic.xlsx` — Daily EOD Activity Log

| Column | Type | Notes |
|--------|------|-------|
| Date | String (DD/MM/YYYY) | Needs parsing |
| First Name | String | No unique intern ID |
| Last Name | String | Combined = intern identifier |
| Activity | String | 12 distinct activity types |
| Hours | Float | Range: 0.5 – 4.0 hrs |

- **Real rows:** 4,379 | **Synthetic rows:** 1,951 | **Merged:** ~6,330
- **Interns:** 20 real → 40 merged | **Period:** Jan 2026 (real) + Feb–Mar 2026 (synthetic)
- **12 Activity Types:** PySpark Session, NumPy Practice, Power BI Dashboard Work, PL/SQL Concepts, SQL Revision, Data Engineering Course, Pandas Practice, Pandas Exam Preparation, PySpark LMS Learning, Advanced SQL Practice, Spark Architecture Study, Project Research

#### Files 2–5: `assignment_submissions_progress_*.xlsx` + `*_synthetic_*.xlsx` — LMS Course Progress
(One real + one synthetic file each for: Basic Python Programming, Basic SQL, NumPy & Pandas, PySpark)

| Column | Type | Notes |
|--------|------|-------|
| User Name | String | Full name, join key |
| Course Name | String | Course identifier |
| Start Date / End Date | String | Course duration |
| Mentor Name | String | Comma-separated list of mentors |
| Progress (%) | String | e.g., "43%" — needs parsing |
| Completed Assignment | String | e.g., "3/3" — fraction format |
| Reviewed / Submitted | String | e.g., "2/3" — fraction format |
| Overall Knowledge Check | String | e.g., "443/470" — score format |
| Overall Test | String | e.g., "25/40" — score format |
| Reviewed / Total Assignments | String | e.g., "2/3" |
| Overall Status | String | Completed / In Progress / Not started |

- **Rows per merged file:** 60 | **Courses:** 4 | **Interns:** 60

### Key Data Quality Issues Identified
| Issue | Field | Fix |
|-------|-------|-----|
| Date stored as string | Date | Parse with `%d/%m/%Y` |
| Progress stored as "43%" | Progress (%) | Strip %, cast to int |
| Scores stored as "443/470" | Knowledge Check, Test | Split on `/` → numerator + denominator |
| Assignments stored as "3/3" | Completed Assignment | Split on `/` → two columns |
| Mentor Name is a comma-separated blob | Mentor Name | Explode into separate dimension table |
| No surrogate intern ID | First Name + Last Name | Create `intern_id` via hashing or index |
| EOD has 20 real interns, LMS has 39 real interns | User Name vs First+Last Name | Normalize names for JOIN |
| Some scores have "-" for missing | Overall Test | Replace with NULL |
| Duplicate interns across real + synthetic LMS files | User Name | Deduplicate keeping real row |

---

## 3. Approach to Solution

### Philosophy
Build **module by module**, validate at each step, and keep everything simple enough to work reliably in 24 hours. Prefer working software over perfect software.

### Step-by-Step Approach

```
Step 1: Understand & Profile the Data          ✅ Done
Step 1b: Synthesize & Merge Data               ✅ Done
Step 2: Design the Data Warehouse Schema       → Todo
Step 3: Build ETL Pipeline                     → Todo
Step 4: Simulate Streaming Ingestion           → Todo
Step 5: Build Dashboards (Quality + EDA + KPI) → Todo
Step 6: Train & Serve ML Models                → Todo
Step 7: Build GenAI RAG Chatbot                → Todo
Step 8: Containerize with Docker Compose       → Todo
```

### Data Merge Strategy (implemented in `etl/ingest.py`)

```python
import pandas as pd

# EOD — simple concat, every row is a unique activity entry
orig_eod  = pd.read_excel('data/raw/intern_eod_last3months_random.xlsx')
synth_eod = pd.read_excel('data/raw/intern_eod_synthetic.xlsx')
orig_eod['data_source']  = 'real'
synth_eod['data_source'] = 'synthetic'
eod_full = pd.concat([orig_eod, synth_eod], ignore_index=True)
# Result: ~6,330 rows, 40 interns, Jan–Mar 2026

# LMS — keep original rows, append only truly new interns
def merge_lms(real_path, synth_path):
    orig  = pd.read_excel(real_path);  orig['data_source']  = 'real'
    synth = pd.read_excel(synth_path); synth['data_source'] = 'synthetic'
    new_only = synth[~synth['User Name'].isin(orig['User Name'])]
    return pd.concat([orig, new_only], ignore_index=True)
# Result per course: 60 rows (39 real + 21 new synthetic)
```

### Decisions Made
- Use **DuckDB** as the data warehouse — no server needed, runs in-process, SQL-compatible
- Use **Streamlit** for all dashboards — fastest Python-native UI
- Use **Scikit-learn** for ML — no overhead, well-documented
- Use **LangChain + Ollama (Llama3)** for GenAI — fully local, no API cost
- Use **Docker Compose** — one file, all services
- **Skip** Kafka, Spark, Airflow, dbt, Snowflake — overkill for this dataset size

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                         │
│  intern_eod.xlsx (real)    │  intern_eod_synthetic.xlsx         │
│  *_lms.xlsx (real x4)      │  *_lms_synthetic.xlsx (x4)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          MODULE 1: DATA INGESTION, MERGE & SIMULATION           │
│  - Merge real + synthetic per file (see merge strategy above)   │
│  - Add data_source flag ('real' / 'synthetic') per row          │
│  - Simulate streaming: replay EOD rows every 10 seconds         │
│  - Python scheduler (no Kafka needed)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODULE 2: ETL PIPELINE                        │
│  - Parse & clean dates, percentages, fraction strings           │
│  - Normalize intern names → create intern_id                    │
│  - Deduplicate LMS on User Name (keep real rows)                │
│  - Explode mentor names → mentor dimension                      │
│  - Merge EOD + LMS data on intern name                          │
│  - Data quality checks (nulls, outliers, type validation)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             MODULE 3: DUCKDB DATA WAREHOUSE                     │
│                                                                 │
│  BRONZE LAYER (Raw)                                             │
│  ├── raw_eod_activities     (~6,330 rows, real+synthetic)       │
│  └── raw_lms_progress       (60 rows per course)                │
│                                                                 │
│  SILVER LAYER (Cleaned & Typed)                                 │
│  ├── dim_intern        (id, name, cohort, data_source)          │
│  ├── dim_course        (id, name, start, end)                   │
│  ├── dim_mentor        (id, name)                               │
│  ├── dim_activity      (id, activity_type, category)            │
│  ├── dim_date          (date, week, month)                      │
│  └── fact_eod_log      (intern, date, activity, hours,          │
│                          data_source)                           │
│                                                                 │
│  GOLD LAYER (Aggregated KPI Tables)                             │
│  ├── gold_intern_weekly_hours                                   │
│  ├── gold_intern_course_progress                                │
│  ├── gold_activity_distribution                                 │
│  └── gold_intern_performance_score                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
┌───────────────────────┐   ┌───────────────────────────────────┐
│   MODULE 4: STREAMLIT │   │         MODULE 5: ML MODELS       │
│   DASHBOARDS          │   │                                   │
│                       │   │  - Classifier: Predict status     │
│  Tab 1: Data Quality  │   │    trained on 60 labeled samples  │
│  Tab 2: EDA           │   │    per course (real + synthetic)  │
│  Tab 3: Manager View  │   │                                   │
│  Tab 4: Mentor View   │   │  - Clustering: Group 40 interns   │
│  Tab 5: Intern View   │   │    by work pattern (K-Means k=3)  │
│  Tab 6: ML Insights   │   │                                   │
│  Tab 7: GenAI Chat    │   │  - Regression: Predict test score │
│                       │   │    from activity hours            │
└───────────────────────┘   └───────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│               MODULE 6: GENAI RAG CHATBOT                       │
│  - Gold tables → text chunks → vector embeddings                │
│  - LangChain retrieval → Ollama (Llama3) LLM                    │
│  - Natural language queries over intern data                    │
│  Example: "Who hasn't started PySpark?"                         │
│  Example: "Which intern logged most hours this week?"           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               MODULE 7: DOCKER COMPOSE DEPLOYMENT              │
│  Services:                                                      │
│  ├── app (Streamlit — port 8501)                                │
│  ├── ollama (LLM server — port 11434)                           │
│  └── duckdb volume (persistent storage)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Tech Stack Decisions

| Category | Tool Chosen | Why | What We Skipped & Why |
|----------|-------------|-----|----------------------|
| Data Warehouse | DuckDB | Zero setup, in-process SQL, perfect for file-scale data | Snowflake, BigQuery (overkill + cost) |
| Dashboard | Streamlit | Python-native, fastest to build, all in one app | Apache Superset (heavy setup) |
| ETL | Python + Pandas | Already familiar, sufficient for this scale | Apache Spark, dbt (overkill) |
| Data Quality | Great Expectations (lite) or manual checks | Simple assertions on dataframes | Full GE suite (too heavy) |
| ML | Scikit-learn | Simple, no GPU needed, fast to train on merged dataset | TensorFlow, PyTorch (overkill) |
| GenAI / LLM | LangChain + Ollama + Llama3 | Fully local, no API cost, works offline | OpenAI API (cost), Hugging Face (complex) |
| Vector Store | ChromaDB | Lightweight, local, works with LangChain | Pinecone, Weaviate (cloud-only) |
| Scheduling | Python `schedule` library | Simple cron-like jobs in pure Python | Apache Airflow (too heavy for 24hrs) |
| Streaming Sim | Python time-based replay | Simulates live ingestion from merged Excel files | Apache Kafka (overkill) |
| Containerization | Docker Compose | Single file, all services, one command | Kubernetes (overkill) |

---

## 6. Module Breakdown & Todo List

### ✅ Module 0: Data Analysis & Synthesis (DONE)
- [x] Analyze all 5 original Excel files
- [x] Identify field types and data quality issues
- [x] Generate synthetic data (same schema, 20 new interns, Feb–Mar 2026)
- [x] Validate synthetic files match original column structure exactly
- [x] Decide architecture and tech stack

---

### 🔲 Module 1: Project Structure Setup
- [ ] Create project folder structure
- [ ] Create `requirements.txt`
- [ ] Create `.env` file for config
- [ ] Create `docker-compose.yml` skeleton

**Folder Structure:**
```
project/
├── data/
│   └── raw/
│       ├── real/             ← original Excel files (5 files)
│       └── synthetic/        ← synthetic Excel files (5 files)
├── etl/
│   ├── ingest.py             ← load + merge real & synthetic files
│   ├── clean.py              ← transformations
│   └── warehouse.py          ← write to DuckDB layers
├── ml/
│   ├── train.py              ← train all 3 models on merged data
│   └── predict.py            ← inference functions
├── genai/
│   ├── embeddings.py         ← chunk + embed gold tables
│   └── chatbot.py            ← RAG chain
├── app/
│   └── streamlit_app.py      ← main multi-tab UI
├── db/
│   └── intern_platform.duckdb
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

### 🔲 Module 2: ETL Pipeline
- [ ] **Ingest & Merge:** Load and merge real + synthetic files per the merge strategy (see Section 3)
  - [ ] Add `data_source` column (`'real'` / `'synthetic'`) to all ingested rows
  - [ ] Deduplicate LMS files on `User Name` — keep real row where both exist
- [ ] **Write Bronze layer:** raw merged tables into DuckDB
- [ ] **Clean EOD data:**
  - [ ] Parse `Date` string → proper date type
  - [ ] Create `intern_id` from First Name + Last Name
  - [ ] Create `full_name` column
  - [ ] Validate `Hours` range (0.5–4.0), flag outliers
- [ ] **Clean LMS data:**
  - [ ] Parse `Progress (%)` → integer (strip `%`)
  - [ ] Parse `Completed Assignment` "3/3" → `completed_count`, `total_count`, `completion_ratio`
  - [ ] Parse `Overall Knowledge Check` "443/470" → `kc_score`, `kc_max`, `kc_pct`
  - [ ] Parse `Overall Test` "25/40" → `test_score`, `test_max`, `test_pct` (handle "-" → NULL)
  - [ ] Parse `Start Date` / `End Date` → date type
  - [ ] Explode `Mentor Name` comma list → separate mentor dimension rows
  - [ ] Normalize `Overall Status` → enum (Completed / In Progress / Not Started)
- [ ] **Merge:** Join EOD + LMS on normalized intern full name
- [ ] **Write Silver layer:** dim tables + fact table (with `data_source` column preserved)
- [ ] **Write Gold layer:** aggregated KPI tables
- [ ] **Data Quality Report:** null counts, type errors, outlier flags, row counts per layer, real vs synthetic split

---

### 🔲 Module 3: Data Warehouse (DuckDB Schema)
*(See Section 7 for full schema)*
- [ ] Create Bronze tables (raw landing zone, real + synthetic merged)
- [ ] Create Silver dimension tables (dim_intern, dim_course, dim_mentor, dim_activity, dim_date)
- [ ] Create Silver fact table (fact_eod_log, fact_lms_progress) — both with `data_source` column
- [ ] Create Gold aggregation tables (weekly hours, course progress, performance scores)
- [ ] Write SQL views for each persona's KPI queries

---

### 🔲 Module 4: Streamlit Dashboard App
- [ ] **Tab 1 — Data Quality Dashboard:**
  - [ ] Show null counts per column (bar chart)
  - [ ] Show row counts per layer (Bronze → Silver → Gold)
  - [ ] Show real vs synthetic row split per table
  - [ ] Show outlier flags in EOD hours
  - [ ] Show data type validation results
- [ ] **Tab 2 — EDA Dashboard:**
  - [ ] Hours distribution (histogram)
  - [ ] Activity type breakdown (pie/bar)
  - [ ] Hours per intern per week (heatmap) — Jan through Mar 2026
  - [ ] Course progress distribution across 60 interns
  - [ ] Intern count per status per course
- [ ] **Tab 3 — Manager View (Persona Dashboard):**
  - [ ] Intern leaderboard (total hours, avg progress) — all 40 interns
  - [ ] Top 5 / Bottom 5 performers
  - [ ] Activity breakdown per intern
  - [ ] Weekly hours trend per intern
- [ ] **Tab 4 — Mentor View:**
  - [ ] Progress of each mentee per course
  - [ ] Assignment submission lag (submitted vs reviewed)
  - [ ] Interns who haven't started a course
  - [ ] Knowledge check score heatmap
- [ ] **Tab 5 — Intern View:**
  - [ ] Personal progress card (select intern from dropdown — all 40 interns)
  - [ ] Hours logged per activity (this week / all time)
  - [ ] Course progress bar per course
  - [ ] Test score vs batch average
- [ ] **Tab 6 — ML Insights:**
  - [ ] Cluster visualization (scatter plot, color = cluster, shape = real/synthetic)
  - [ ] Performance classification result per intern
  - [ ] Predicted test score vs actual
- [ ] **Tab 7 — GenAI Chatbot:**
  - [ ] Text input box
  - [ ] Chat history display
  - [ ] Response from RAG chain

---

### 🔲 Module 5: ML Models
*(See Section 8 for details)*
- [ ] **Model 1 — Classifier:**
  - [ ] Features: hours per activity, LMS progress %, assignment ratio
  - [ ] Target: Overall Status (Completed / In Progress / Not Started)
  - [ ] Algorithm: Random Forest Classifier
  - [ ] Training set: merged data (60 samples per course); test set: real-only rows
  - [ ] Evaluate: Accuracy, Confusion Matrix
- [ ] **Model 2 — Clustering:**
  - [ ] Features: total hours, activity diversity, avg progress across courses
  - [ ] Algorithm: K-Means (k=3: High / Medium / Low performer)
  - [ ] Input: all 40 interns from merged EOD + LMS
  - [ ] Visualize: 2D PCA scatter plot
- [ ] **Model 3 — Regression:**
  - [ ] Features: hours spent on related activities, knowledge check score
  - [ ] Target: Overall Test Score (percentage)
  - [ ] Algorithm: Linear Regression / Ridge
  - [ ] Training set: merged data; evaluate on real-only rows
  - [ ] Evaluate: R², MAE

---

### 🔲 Module 6: GenAI RAG Chatbot
*(See Section 9 for details)*
- [ ] Convert Gold layer tables → readable text chunks (all 40 interns, all 60 LMS rows per course)
- [ ] Embed chunks using `sentence-transformers`
- [ ] Store in ChromaDB vector store
- [ ] Build LangChain retrieval chain
- [ ] Connect to Ollama (Llama3) as LLM
- [ ] Expose chatbot in Streamlit Tab 7
- [ ] Test sample queries

---

### 🔲 Module 7: Docker Compose Deployment
*(See Section 11 for details)*
- [ ] Write `Dockerfile` for Streamlit app
- [ ] Write `docker-compose.yml` with all services
- [ ] Mount DuckDB as persistent volume
- [ ] Mount `data/raw/real/` and `data/raw/synthetic/` folders for all Excel files
- [ ] Add `ollama` service with Llama3 model pull on startup
- [ ] Test full `docker compose up` end-to-end
- [ ] Write `README.md` with setup instructions

---

## 7. Data Warehouse Design

### Medallion Architecture

#### Bronze Layer — Raw (merged real + synthetic, as-is)
```sql
raw_eod_activities     -- merged load of intern_eod real + synthetic (~6,330 rows)
raw_lms_python         -- merged load of python LMS real + synthetic (60 rows)
raw_lms_sql            -- merged load of sql LMS real + synthetic (60 rows)
raw_lms_numpy_pandas   -- merged load of numpy/pandas LMS real + synthetic (60 rows)
raw_lms_pyspark        -- merged load of pyspark LMS real + synthetic (60 rows)
```

#### Silver Layer — Cleaned & Typed (Star Schema)

**Dimension Tables:**
```sql
dim_intern    (intern_id PK, first_name, last_name, full_name, data_source)
dim_course    (course_id PK, course_name, start_date, end_date)
dim_mentor    (mentor_id PK, mentor_name)
dim_activity  (activity_id PK, activity_name, activity_category)
dim_date      (date_id PK, date, day_of_week, week_number, month)
```

**Bridge Table (Intern ↔ Mentor — many-to-many):**
```sql
bridge_intern_mentor (intern_id FK, course_id FK, mentor_id FK)
```

**Fact Tables:**
```sql
fact_eod_log (
  log_id PK, intern_id FK, date_id FK, activity_id FK,
  hours FLOAT, data_source VARCHAR
)

fact_lms_progress (
  progress_id PK, intern_id FK, course_id FK,
  progress_pct INT,
  completed_assignments INT, total_assignments INT,
  kc_score FLOAT, kc_max FLOAT, kc_pct FLOAT,
  test_score FLOAT, test_max FLOAT, test_pct FLOAT,
  overall_status VARCHAR, data_source VARCHAR
)
```

#### Gold Layer — Aggregated KPI Tables
```sql
gold_weekly_hours          -- intern_id, week, total_hours, avg_hours_per_day
gold_activity_summary      -- intern_id, activity_name, total_hours, activity_count
gold_course_progress       -- intern_id, course_name, progress_pct, status, test_pct
gold_intern_performance    -- intern_id, avg_progress, avg_test_pct, total_hours, cluster_label
gold_mentor_workload       -- mentor_id, intern_count, avg_mentee_progress
```

---

## 8. ML Use Cases

### Model 1: Performance Classifier
- **Problem:** Given an intern's activity hours and LMS data, predict if they will Complete, be In Progress, or Not Start a course
- **Type:** Multi-class Classification
- **Algorithm:** Random Forest (handles mixed features well, interpretable)
- **Input Features:** total hours on related activities, days since course start, knowledge check %, previous course completion status
- **Output:** Completed / In Progress / Not Started
- **Training Data:** 60 labeled samples per course from merged dataset (real + synthetic)
- **Validation:** Evaluate on real-only rows (`data_source == 'real'`) to avoid synthetic label leakage
- **Use:** Manager can see at-risk interns early

### Model 2: Intern Clustering
- **Problem:** Identify natural groups of interns based on behavior patterns
- **Type:** Unsupervised Clustering
- **Algorithm:** K-Means with k=3
- **Input Features:** total hours logged, number of distinct activities, avg course progress %, assignment completion ratio
- **Output:** Cluster 0 (High Performer), Cluster 1 (Average), Cluster 2 (At Risk)
- **Training Data:** All 40 interns from merged EOD + LMS data
- **Use:** Personalize mentoring strategy per cluster

### Model 3: Test Score Predictor
- **Problem:** Can we predict a student's test score from their activity patterns?
- **Type:** Regression
- **Algorithm:** Ridge Regression
- **Input Features:** hours on course-related activities, knowledge check score, assignment completion ratio
- **Output:** Predicted test score (%)
- **Training Data:** ~60 samples with test scores from merged LMS files
- **Validation:** Evaluate on real-only rows to measure true generalization
- **Use:** Early warning — predict score before the test

---

## 9. GenAI Use Cases

### RAG Chatbot Architecture
```
User Question (natural language)
      ↓
Embed question → Search ChromaDB (similarity search)
      ↓
Retrieve top-k relevant data chunks from Gold tables
(chunks built from all 40 interns across Jan–Mar 2026)
      ↓
Prompt = [System prompt] + [Retrieved context] + [User question]
      ↓
Ollama (Llama3) generates answer
      ↓
Display in Streamlit chat UI
```

### Sample Queries the Chatbot Should Handle
| Query | Expected Data Source |
|-------|---------------------|
| "Who hasn't started PySpark yet?" | gold_course_progress |
| "Which intern logged the most hours this week?" | gold_weekly_hours |
| "Show me top performers in SQL" | gold_intern_performance |
| "Which interns are at risk of not completing NumPy?" | gold_course_progress + classifier output |
| "How many assignments has Akshat submitted?" | fact_lms_progress |
| "Summarize batch progress across all courses" | gold_course_progress |

### Data Chunking Strategy
- Each intern's summary = one chunk (name + all course progress + total hours) — 40 chunks total
- Each course's summary = one chunk (all 60 interns in that course + their status)
- Gold table rows = converted to natural language sentences before embedding

---

## 10. Personas & KPIs

### Persona 1: Manager
**Goal:** Understand overall batch performance and identify top/bottom interns

| KPI | Calculation |
|-----|-------------|
| Average Hours / Week per Intern | SUM(hours) / weeks — across all 40 interns, Jan–Mar 2026 |
| Overall Batch Progress | AVG(progress_pct) across all courses, 60 interns |
| Top 5 Performers | Ranked by avg_progress + total_hours |
| At-Risk Interns | Status = Not Started OR progress < 30% |
| Activity Distribution | Hours by activity type across batch |

### Persona 2: Mentor
**Goal:** Track mentee progress and assignment review lag

| KPI | Calculation |
|-----|-------------|
| Mentee Progress per Course | progress_pct per intern (60 interns per course) |
| Assignment Review Lag | reviewed / submitted ratio |
| Knowledge Check Avg | AVG(kc_pct) across mentees |
| Interns Needing Attention | progress_pct < 50% AND status ≠ Completed |

### Persona 3: HR
**Goal:** Batch-level attendance and dropout risk patterns

| KPI | Calculation |
|-----|-------------|
| Daily Active Interns | COUNT(distinct intern_id) per date — Jan through Mar 2026 |
| Course Completion Rate | COUNT(status=Completed) / total interns (60 per course) |
| Batch Comparison | Progress by cohort/course over time |
| Dropout Risk | Not Started count per course |

### Persona 4: Intern (Self-View)
**Goal:** See personal progress, hours logged, and performance vs peers

| KPI | Calculation |
|-----|-------------|
| Hours Logged This Week | SUM(hours) for current week |
| My Progress per Course | progress_pct for self |
| My Test Score vs Avg | test_pct vs AVG(test_pct) across all 60 interns |
| My Activity Breakdown | Hours per activity type |

---

## 11. Docker Deployment Plan

### Services in `docker-compose.yml`
```yaml
services:
  app:
    build: .
    ports: ["8501:8501"]
    volumes:
      - ./data:/app/data        # real/ and synthetic/ Excel files
      - ./db:/app/db            # DuckDB persistent file
      - ./genai:/app/genai      # ChromaDB vector store
    depends_on: [ollama]
    environment:
      - OLLAMA_HOST=http://ollama:11434

  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes:
      - ollama_data:/root/.ollama
    entrypoint: ["/bin/sh", "-c", "ollama serve & sleep 5 && ollama pull llama3 && wait"]

volumes:
  ollama_data:
```

### `Dockerfile` for App
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### One-Command Startup
```bash
docker compose up --build
# Access at: http://localhost:8501
```

---

## 12. 24-Hour Execution Timeline

| Hour Range | Task | Module |
|------------|------|--------|
| 00:00 – 01:00 | Project structure setup, requirements.txt, folder creation (`data/raw/real/` + `data/raw/synthetic/`) | M1 |
| 01:00 – 03:00 | ETL: Merge real + synthetic → Bronze layer ingestion + all cleaning transformations | M2 |
| 03:00 – 05:00 | DuckDB: Silver schema + fact/dim tables (with `data_source` column) | M3 |
| 05:00 – 06:00 | DuckDB: Gold aggregation tables | M3 |
| 06:00 – 08:00 | Streamlit: Data Quality + EDA tabs (real vs synthetic split visible) | M4 |
| 08:00 – 11:00 | Streamlit: Manager + Mentor + Intern persona dashboards (all 40 interns) | M4 |
| 11:00 – 13:00 | ML: Train classifier + clustering + regression on merged data, validate on real-only | M5 |
| 13:00 – 14:00 | Streamlit: ML Insights tab | M4+M5 |
| 14:00 – 17:00 | GenAI: Chunking (40 intern summaries) + embedding + ChromaDB + LangChain RAG | M6 |
| 17:00 – 18:00 | Streamlit: GenAI Chatbot tab | M4+M6 |
| 18:00 – 20:00 | Docker: Dockerfile + docker-compose.yml + full build test | M7 |
| 20:00 – 22:00 | End-to-end testing, bug fixes | All |
| 22:00 – 24:00 | README, demo prep, final polish | All |

---

## ✅ Definition of Done (What "Working" Means)

- [ ] `docker compose up --build` starts all services without errors
- [ ] Streamlit app loads at `http://localhost:8501`
- [ ] All 7 dashboard tabs render with real data from DuckDB (merged dataset)
- [ ] ML models load and display predictions (trained on merged ~60-sample dataset)
- [ ] GenAI chatbot responds to at least 3 sample queries correctly (using 40-intern vector store)
- [ ] DuckDB persists data across container restarts (volume mount works)
- [ ] All 10 Excel files (5 real + 5 synthetic) are fully processed and visible in Gold layer

---

*Last Updated: Data strategy updated to include real + synthetic merged dataset (40 interns EOD, 60 interns per LMS course, Jan–Mar 2026)*
*Team: Building for CHARUSAT, Changa — 24hr Round 2*
