<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/DuckDB-Warehouse-FFC107?logo=duckdb&logoColor=black" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq%20%7C%20Ollama-GenAI-8A2BE2?logo=meta&logoColor=white" />
</p>

# 🎓 InternIQ — Intern Analytics & Intelligence Platform

**InternIQ** is an end-to-end analytics platform that ingests raw intern activity logs and LMS course data, builds a **Medallion Data Warehouse** (Bronze → Silver → Gold) in DuckDB, trains **ML models** for performance prediction & clustering, and serves everything through an interactive **Streamlit dashboard** with a **GenAI Text-to-SQL chatbot**.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Medallion Architecture** | Bronze → Silver → Gold data layers in DuckDB with automated ETL |
| **Role-Based Dashboard** | Separate Manager, Mentor, and Intern views with authentication |
| **ML Models** | Random Forest classifier, K-Means clustering (PCA), Ridge regression |
| **GenAI Chatbot** | Natural language → SQL using Groq / HuggingFace / Ollama (auto-fallback) |
| **Live Predictions** | Interactive widget to predict intern performance in real-time |
| **Data Quality Monitoring** | Null counts, outlier flags, layer-wise row counts |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) [Groq API key](https://console.groq.com) for the chatbot (free tier available)

### Installation & Run

```bash
# 1. Clone the repo
git clone https://github.com/AryaKayastha/Intern-IQ.git
cd Intern-IQ

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
#    Copy .env.example to .env and add your API keys (optional for chatbot)

# 5. Run ETL pipeline (builds DuckDB warehouse)
python -m etl.ingest
python -m etl.warehouse

# 6. Train ML models
python -m ml.train

# 7. Launch the dashboard
streamlit run app/streamlit_app.py
# Visit http://localhost:8501
```

### 🐳 Docker

```bash
docker compose up --build
# Visit http://localhost:8501
```

---

## 🏗️ Project Structure

```
Intern-IQ/
├── dataset/
│   ├── raw/                  # Original Excel files (EOD + LMS)
│   └── synthesize/           # Synthetic data for augmentation
├── etl/
│   ├── ingest.py             # Merge real + synthetic → Bronze layer
│   ├── clean.py              # Parse dates, fractions, %, names
│   └── warehouse.py          # Bronze → Silver → Gold (DuckDB)
├── ml/
│   ├── train.py              # Train 3 ML models (RF / KMeans / Ridge)
│   ├── predict.py            # Inference functions for live predictions
│   └── models/               # Saved .pkl + .parquet artifacts
├── genai/
│   └── chatbot.py            # Text-to-SQL chatbot (Groq / HF / Ollama)
├── app/
│   └── streamlit_app.py      # Multi-tab Streamlit dashboard
├── db/
│   └── intern_platform.duckdb
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env                      # API keys (not committed)
```

---

## 🏛️ Data Architecture — Medallion Warehouse

```
Raw Excel Files (EOD Activity Logs + LMS Course Data)
         │
         ▼  etl/ingest.py
┌─────────────────────────────────────────────────────┐
│  BRONZE  │  raw_eod_activities, raw_lms_*           │
│          │  (Verbatim ingestion, source tracking)    │
└─────────────────────────────────────────────────────┘
         │
         ▼  etl/clean.py + warehouse.py
┌─────────────────────────────────────────────────────┐
│  SILVER  │  dim_intern, dim_course, dim_mentor,     │
│          │  dim_activity, dim_date                   │
│          │  fact_eod_log, fact_lms_progress          │
│          │  bridge_intern_mentor                     │
└─────────────────────────────────────────────────────┘
         │
         ▼  warehouse.py (SQL aggregations)
┌─────────────────────────────────────────────────────┐
│  GOLD    │  gold_weekly_hours                       │
│          │  gold_activity_summary                   │
│          │  gold_course_progress                    │
│          │  gold_intern_performance                 │
│          │  gold_mentor_workload                    │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Dashboard Tabs

The dashboard uses **role-based access control** with three roles:

### 🔑 Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Manager | `admin` | `admin@123` |
| Mentor | `firstnamelastname` (e.g. `johndoe`) | `username@123` |
| Intern | `firstnamelastname` (e.g. `janedoe`) | `username@123` |

### Manager View
| Module | Description |
|--------|-------------|
| **Manager Dashboard** | 61-intern leaderboard, top/bottom 5 performers, activity breakdown, weekly batch trends |
| **EDA** | Hours distribution, activity breakdown, weekly comparison, course progress box plots, status stacked bars |
| **ML Insights** | PCA cluster scatter, regression plot, classification table, live prediction widget |
| **Chatbox** | Natural language Q&A over intern data |
| **Warehouse Analysis** | Row counts per medallion layer, null counts, outlier flags |

### Mentor View
| Module | Description |
|--------|-------------|
| **Mentor Dashboard** | Mentee progress treemap, at-risk alerts, KC score heatmaps, workload summary |
| **ML Insights** | Same as Manager |
| **Chatbox** | Same as Manager |

### Intern View
| Module | Description |
|--------|-------------|
| **Intern Self-View** | Personal stats, course progress bars, test vs batch average, weekly hours trend |

---

## 🤖 ML Models

| Model | Algorithm | Purpose | Details |
|-------|-----------|---------|---------|
| **Performance Classifier** | Random Forest | Predict intern overall status | Trained on course-level features |
| **Intern Clustering** | K-Means (k=3) + PCA | Group interns by work patterns | Labels: High Performer / Average / At Risk |
| **Test Score Predictor** | Ridge Regression | Predict test score % | Uses progress, KC score, assignment ratio |

### Live Prediction Widget
Enter custom intern stats (progress %, KC score, domain hours, etc.) and get real-time predictions for:
- Performance classification (Completed / In Progress / At Risk)
- Predicted test score
- Cluster assignment

---

## 💬 GenAI Chatbot

The chatbot converts natural language questions into SQL queries and executes them against the DuckDB warehouse.

**LLM Backend Priority:**
1. **Groq** (cloud, fastest ~1-2s, free tier) 
2. **HuggingFace Inference API** (cloud fallback)
3. **Ollama** (local, no internet needed)

### Sample Queries
- *"Who are the top 5 performers in SQL?"*
- *"Which interns haven't started PySpark yet?"*
- *"Show me the average test score per course"*
- *"Who logged the most hours this week?"*
- *"List interns at risk of not completing NumPy"*

### Setup (Optional — needed only for Chatbot)
Add your API key(s) to `.env`:
```env
GROQ_API_KEY=your_groq_key_here          # Recommended (free at console.groq.com)
HUGGINGFACEHUB_API_TOKEN=your_hf_token   # Alternative
OLLAMA_MODEL=llama3                       # For local Ollama
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Data Warehouse** | DuckDB (Medallion Architecture) |
| **ETL Pipeline** | Python, Pandas, OpenPyXL |
| **Dashboard** | Streamlit, Plotly |
| **Machine Learning** | scikit-learn (Random Forest, K-Means, Ridge) |
| **GenAI / LLM** | Groq API, HuggingFace Inference, Ollama |
| **Containerization** | Docker, Docker Compose |

---

## 📁 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DUCKDB_PATH` | Path to DuckDB file | No (default: `db/intern_platform.duckdb`) |
| `GROQ_API_KEY` | Groq API key for chatbot | No (chatbot only) |
| `GROQ_MODEL` | Groq model name | No (default: `llama-3.1-8b-instant`) |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace token | No (fallback for chatbot) |
| `OLLAMA_HOST` | Ollama server URL | No (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | Ollama model name | No (default: `llama3`) |

---

## 📄 License

This project is for educational and demonstration purposes.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/AryaKayastha">Arya Kayastha</a>
</p>
