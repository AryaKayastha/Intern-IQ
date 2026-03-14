"""
genai/chatbot.py
----------------
Text-to-SQL chatbot using direct HTTP calls to LLM backends.
No LangChain version conflicts — uses requests for both HuggingFace and Ollama.

PRIMARY:  HuggingFace Inference API (cloud GPU, fast)
FALLBACK: Local Ollama (when HF token absent or API fails)

Only 1 LLM call per query (SQL generation). Results returned as structured
rows + columns so Streamlit renders a native interactive dataframe.
"""

import os
import re
import json
import duckdb
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "db", "intern_platform.duckdb")

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
HF_TOKEN     = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
HF_MODEL     = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ── LLM callers (pure HTTP, no LangChain) ────────────────────────────────────

def _call_groq(prompt: str) -> str:
    """
    Groq Cloud API — OpenAI-compatible, free, ~1-2 s/query.
    Supports: llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.
    Get a free key at https://console.groq.com
    """
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.0,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def _call_huggingface(prompt: str) -> str:
    """Call HuggingFace Inference API. Returns the generated text."""
    url     = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.01,
            "return_full_text": False,   # only return the completion, not the prompt
            "do_sample": True,
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # HF returns a list: [{"generated_text": "..."}]
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip()
    raise ValueError(f"Unexpected HuggingFace response: {data}")


def _call_ollama(prompt: str) -> str:
    """Call local Ollama /api/generate endpoint."""
    url     = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    resp    = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def call_llm(prompt: str) -> tuple:
    """
    Try backends in priority order: Groq → HuggingFace → Ollama.
    Returns (generated_text: str, source: str).
    """
    # 1. Groq (fastest — free cloud LPU, ~1-2 s)
    if GROQ_KEY and GROQ_KEY != "your_groq_key_here":
        try:
            text = _call_groq(prompt)
            return text, "groq"
        except Exception as e:
            print(f"[GenAI] Groq failed ({e}). Trying HuggingFace.")

    # 2. HuggingFace Inference API
    if HF_TOKEN and HF_TOKEN != "your_hf_token_here":
        try:
            text = _call_huggingface(prompt)
            return text, "huggingface"
        except Exception as e:
            print(f"[GenAI] HuggingFace failed ({e}). Falling back to Ollama.")

    # 3. Local Ollama (no internet needed)
    try:
        text = _call_ollama(prompt)
        return text, "ollama"
    except Exception as e:
        raise RuntimeError(f"All LLM backends unavailable. Ollama error: {e}")


# ── Annotated schema (hardcoded for reliable table routing) ──────────────────
SCHEMA = """
-- TABLE 1: gold_course_progress
-- USE FOR: specific course questions (SQL/Python/PySpark/NumPy/Big Data),
--          top performers per course, completion status, assignment/KC/test scores.
CREATE TABLE gold_course_progress (
    intern_id        VARCHAR,   -- unique intern id
    full_name        VARCHAR,   -- intern full name
    course_name      VARCHAR,   -- 'Basic SQL' | 'Basic Python Programming' |
                                -- 'Data Processing using PySpark' |
                                -- 'Data Processing using NumPy & Pandas' |
                                -- 'Introduction to Data Engineering & Big Data'
    progress_pct     BIGINT,    -- course progress 0-100
    overall_status   VARCHAR,   -- 'Completed' | 'In Progress' | 'Not Started'
    assignment_ratio DOUBLE,    -- assignments submitted ratio 0.0-1.0
    kc_pct           DOUBLE,    -- knowledge-check score 0-100
    test_pct         DOUBLE,    -- test score 0-100
    completed_count  DOUBLE,
    total_assignments DOUBLE,
    data_source      VARCHAR    -- 'real' | 'synthetic'
);

-- TABLE 2: gold_intern_performance
-- USE FOR: overall intern leaderboard, total hours across all activities,
--          average scores across all courses, courses completed count.
CREATE TABLE gold_intern_performance (
    intern_id              VARCHAR,
    full_name              VARCHAR,
    data_source            VARCHAR,  -- 'real' | 'synthetic'
    total_hours            DOUBLE,
    total_activity_entries BIGINT,
    distinct_activities    BIGINT,
    avg_progress_pct       DOUBLE,
    avg_kc_pct             DOUBLE,
    avg_test_pct           DOUBLE,
    courses_completed      BIGINT,
    avg_assignment_ratio   DOUBLE
);

-- TABLE 3: gold_weekly_hours
-- USE FOR: hours per week, weekly trends, most/least active weeks.
CREATE TABLE gold_weekly_hours (
    intern_id              VARCHAR,
    full_name              VARCHAR,
    year                   BIGINT,
    week_number            BIGINT,
    month_name             VARCHAR,
    activity_count         BIGINT,
    total_hours            DOUBLE,
    avg_hours_per_activity DOUBLE,
    active_days            BIGINT
);
"""

SQL_PROMPT_TEMPLATE = """You are an expert DuckDB SQL assistant for 'InternIQ', an intern analytics platform.
Write ONE DuckDB SQL SELECT query to answer the question.

STRICT RULES:
1. Return ONLY the raw SQL — no markdown, no backticks, no explanation at all.
2. Choose the table based on the routing comments in the schema.
3. ALWAYS include full_name in the SELECT list so results are human-readable.
4. NEVER select intern_id as a standalone output column — it is an internal key, not meaningful to users.
5. Use ILIKE '%term%' for ALL string comparisons to handle typos.
6. Course synonyms:
   - "SQL"           → course_name ILIKE '%SQL%'
   - "Python"        → course_name ILIKE '%Python%'
   - "PySpark"       → course_name ILIKE '%PySpark%'
   - "NumPy/Pandas"  → course_name ILIKE '%NumPy%'
   - "Big Data"      → course_name ILIKE '%Big Data%' OR course_name ILIKE '%Engineering%'
7. Status synonyms:
   - "completed"    → overall_status = 'Completed'
   - "in progress"  → overall_status = 'In Progress'
   - "at risk" / "not started" → overall_status = 'Not Started'
8. "Top performers" per course → ORDER BY progress_pct DESC
   "Top performers" overall   → ORDER BY total_hours DESC
9. NEVER filter by data_source unless user explicitly says real or synthetic.
10. Always end with LIMIT 50.
11. Today is {today_date}. Convert relative dates to exact SQL date filters.
12. If the question is unrelated to interns, courses, or performance, return exactly: OUT_OF_CONTEXT

Schema:
{schema}

Question: {question}
SQL:"""

NARRATIVE_PROMPT_TEMPLATE = """You are a helpful analytics assistant for 'InternIQ', an intern training platform.
A user asked a question and the following data was retrieved from the database.
Write a concise, friendly, conversational paragraph (3-5 sentences) summarising the key insights from the data.
Do NOT just list the raw numbers — interpret them meaningfully.
Do NOT use markdown headers or bullet points — plain paragraph prose only.
Focus on the most interesting or actionable findings.

User question: {question}
SQL result (first 10 rows shown):
{result_preview}

Insight:"""


# ── Input validation ──────────────────────────────────────────────────────────
_BLOCKED = re.compile(
    r"(ignore previous|system prompt|drop table|delete from|update\s+\w|"
    r"insert into|truncate\s+\w|--\s|/\*)",
    re.IGNORECASE,
)

def validate_input(q: str) -> bool:
    return not bool(_BLOCKED.search(q))


# ── Safe SQL execution ────────────────────────────────────────────────────────
def run_sql(sql: str):
    """Single-execute, read_only DuckDB connection. Returns (cols, rows, error)."""
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        cur = con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        con.close()
        return cols, rows, None
    except Exception as e:
        return [], [], str(e)


# ── Public API ────────────────────────────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Returns:
        {
            "answer":     str,
            "query":      str | None,
            "columns":    list[str],
            "rows":       list[tuple],
            "llm_source": str,   # "huggingface" | "ollama"
        }
    """
    base = {"answer": "", "query": None, "columns": [], "rows": [], "llm_source": "unknown"}

    if not validate_input(question):
        base["answer"] = "Security alert: prompt contains disallowed keywords. Please rephrase."
        return base

    today  = datetime.now().strftime("%Y-%m-%d")
    prompt = SQL_PROMPT_TEMPLATE.format(schema=SCHEMA, today_date=today, question=question)

    # ── 1. Generate SQL ───────────────────────────────────────────────────────
    try:
        raw_sql, source = call_llm(prompt)
    except RuntimeError as e:
        base["answer"] = str(e)
        return base

    base["llm_source"] = source
    raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

    # Extract only first SQL statement if LLM returned explanation text after
    lines = raw_sql.splitlines()
    sql_lines = []
    for line in lines:
        if line.strip().upper().startswith("OUT_OF_CONTEXT"):
            base["answer"] = "I can only answer questions about InternIQ intern data."
            return base
        sql_lines.append(line)
        if line.strip().endswith(";"):
            break
    raw_sql = "\n".join(sql_lines).rstrip(";").strip()

    # ── 2. Execute with one auto-repair attempt ───────────────────────────────
    current_sql = raw_sql
    cols, rows, err = run_sql(current_sql)

    if err:
        repair = (
            f"This DuckDB query failed:\nError: {err}\nQuery: {current_sql}\n\n"
            f"Return ONLY the corrected raw SQL. No markdown, no explanation."
        )
        try:
            fixed, _ = call_llm(repair)
            current_sql = fixed.replace("```sql", "").replace("```", "").strip()
        except Exception:
            base.update({"answer": f"Query failed: {err}", "query": current_sql})
            return base
        cols, rows, err = run_sql(current_sql)
        if err:
            base.update({"answer": f"Query failed after retry: {err}", "query": current_sql})
            return base

    base["query"]   = current_sql
    base["columns"] = cols
    base["rows"]    = rows

    # ── 3. Narrative answer via LLM (fast with Groq, ~1-2s) ──────────────────
    n = len(rows)
    if n == 0:
        base["answer"] = "No matching records found for your query."
    else:
        # Build a readable preview of the first 10 rows for the LLM to summarise
        header = " | ".join(cols)
        preview_rows = "\n".join(" | ".join(str(v) for v in r) for r in rows[:10])
        result_preview = f"{header}\n{preview_rows}"
        if n > 10:
            result_preview += f"\n... and {n - 10} more rows."

        narrative_prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            question=question,
            result_preview=result_preview,
        )
        try:
            narrative, _ = call_llm(narrative_prompt)
            base["answer"] = narrative.strip()
        except Exception:
            # Graceful fallback if narrative call fails
            noun = "record" if n == 1 else "records"
            base["answer"] = f"Found **{n} {noun}**. See the table below for details."

    return base
