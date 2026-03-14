"""
genai/embeddings.py
-------------------
Convert Gold layer tables into natural-language text chunks,
embed with sentence-transformers, and store in ChromaDB.

Run once (or whenever Gold data is refreshed):
  python -m genai.embeddings
"""

import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, os.getenv("DUCKDB_PATH", "db/intern_platform.duckdb"))
CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "genai/chroma_store"))


# ============================================================================
# Chunk builders
# ============================================================================

def build_intern_chunks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One chunk per intern: name + hours + course progress summary."""
    perf = con.execute("SELECT * FROM gold_intern_performance").df()
    prog = con.execute("SELECT * FROM gold_course_progress").df()

    chunks = []
    for _, row in perf.iterrows():
        name        = row["full_name"]
        intern_id   = row["intern_id"]
        total_hours = row.get("total_hours", 0)
        avg_prog    = row.get("avg_progress_pct", 0)
        avg_test    = row.get("avg_test_pct", 0)
        n_courses   = row.get("courses_completed", 0)

        # Get per-course detail
        my_courses = prog[prog["intern_id"] == intern_id]
        course_lines = []
        for _, cr in my_courses.iterrows():
            status  = cr.get("overall_status", "Unknown")
            cp      = cr.get("progress_pct", 0)
            tp      = cr.get("test_pct", 0)
            cname   = cr.get("course_name", "")
            course_lines.append(
                f"  - {cname}: {status}, {cp:.0f}% progress, test score {tp:.1f}%"
            )
        course_text = "\n".join(course_lines) if course_lines else "  No course data"

        text = (
            f"Intern: {name}\n"
            f"Total hours logged: {total_hours:.1f}\n"
            f"Average progress across courses: {avg_prog:.1f}%\n"
            f"Average test score: {avg_test:.1f}%\n"
            f"Courses completed: {int(n_courses)}\n"
            f"Course breakdown:\n{course_text}\n"
        )
        chunks.append({
            "id":       f"intern_{intern_id}",
            "text":     text,
            "metadata": {"type": "intern", "name": name, "intern_id": intern_id},
        })
    return chunks


def build_course_chunks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One chunk per course: all 60 intern statuses."""
    prog = con.execute("SELECT * FROM gold_course_progress ORDER BY course_name, overall_status").df()

    chunks = []
    for course_name, group in prog.groupby("course_name"):
        status_summary = group["overall_status"].value_counts().to_dict()
        completed  = status_summary.get("Completed",   0)
        in_prog    = status_summary.get("In Progress", 0)
        not_start  = status_summary.get("Not Started", 0)

        not_started_names = group[group["overall_status"] == "Not Started"]["full_name"].tolist()
        top_scorers       = group.nlargest(5, "test_pct")[["full_name", "test_pct"]]

        top_text = "\n".join([
            f"  - {r['full_name']}: {r['test_pct']:.1f}%"
            for _, r in top_scorers.iterrows()
            if pd.notna(r["test_pct"])
        ])
        not_start_text = ", ".join(not_started_names[:10]) + ("…" if len(not_started_names) > 10 else "")

        text = (
            f"Course: {course_name}\n"
            f"Total interns: {len(group)}\n"
            f"Completed: {completed}, In Progress: {in_prog}, Not Started: {not_start}\n"
            f"Average progress: {group['progress_pct'].mean():.1f}%\n"
            f"Average test score: {group['test_pct'].mean():.1f}%\n"
            f"Interns not started: {not_start_text or 'None'}\n"
            f"Top scorers:\n{top_text or '  No test data'}\n"
        )
        chunks.append({
            "id":       f"course_{course_name.replace(' ', '_').lower()}",
            "text":     text,
            "metadata": {"type": "course", "course_name": course_name},
        })
    return chunks


def build_weekly_chunks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Summary chunks for top hourly performers per week."""
    wh = con.execute("""
        SELECT full_name, week_number, year, total_hours
        FROM gold_weekly_hours
        ORDER BY year, week_number, total_hours DESC
    """).df()

    chunks = []
    for (year, week), group in wh.groupby(["year", "week_number"]):
        top_n = group.head(5)
        lines = "\n".join([
            f"  - {r['full_name']}: {r['total_hours']:.1f} hrs"
            for _, r in top_n.iterrows()
        ])
        text = (
            f"Weekly hours summary – Year {int(year)}, Week {int(week)}\n"
            f"Top performers:\n{lines}\n"
            f"Total active interns this week: {len(group)}\n"
        )
        chunks.append({
            "id":       f"week_{year}_{week}",
            "text":     text,
            "metadata": {"type": "weekly", "year": int(year), "week": int(week)},
        })
    return chunks


# ============================================================================
# Embed and store in ChromaDB
# ============================================================================

def build_and_store_chunks() -> None:
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"  [WARN] GenAI dependencies not installed: {e}")
        print("  Skipping embedding step — chatbot will be unavailable.")
        return

    con = duckdb.connect(DB_PATH, read_only=True)
    print("\n=== Generating text chunks from Gold layer ===")

    intern_chunks = build_intern_chunks(con)
    course_chunks = build_course_chunks(con)
    weekly_chunks = build_weekly_chunks(con)
    all_chunks    = intern_chunks + course_chunks + weekly_chunks
    con.close()

    print(f"  {len(intern_chunks)} intern chunks + "
          f"{len(course_chunks)} course chunks + "
          f"{len(weekly_chunks)} weekly chunks = {len(all_chunks)} total")

    # Embed
    print("  Loading embedding model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    print("  Generating embeddings…")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # Store in ChromaDB
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name="intern_iq",
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert all chunks
    collection.upsert(
        ids        =[c["id"]       for c in all_chunks],
        documents  =[c["text"]     for c in all_chunks],
        embeddings =embeddings,
        metadatas  =[c["metadata"] for c in all_chunks],
    )
    print(f"  Stored {collection.count()} chunks in ChromaDB at {CHROMA_DIR}")
    print("\n✅ Embedding complete.\n")


if __name__ == "__main__":
    build_and_store_chunks()
