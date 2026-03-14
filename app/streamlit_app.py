"""
app/streamlit_app.py
--------------------
InternIQ — 7-tab Streamlit Dashboard
Tabs: Data Quality | EDA | Manager | Mentor | Intern | ML Insights | GenAI Chat
"""

import os
import sys

# Add project root to path so etl/ml/genai packages are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(ROOT, os.getenv("DUCKDB_PATH", "db/intern_platform.duckdb"))

st.set_page_config(
    page_title="InternIQ Analytics Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Cards */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px;
}
.kpi-number { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.kpi-label  { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

/* Tab headers */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.9rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    color: white !important;
    border-radius: 8px 8px 0 0;
}

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    padding: 10px 0;
}
.sub-header {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ── DB query execution (cached, auto-closing to avoid locks) ───────────────────
@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    with duckdb.connect(DB_PATH, read_only=True) as con:
        return con.execute(sql).df()


def q(sql: str) -> pd.DataFrame:
    return query(sql)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎓 InternIQ Analytics Platform</div>', unsafe_allow_html=True)

# ── Authentication ────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.real_name = None

if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.subheader("Welcome to InternIQ — Please Login")
        st.info("💡 Hint: Username is FirstnameLastname (e.g. johndoe). Password is username@123. For Managers, use `admin` / `admin@123`")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["Manager", "Mentor", "Intern"])
            submitted = st.form_submit_button("Login")
            
            if submitted:
                def validate_login(u, p, r):
                    u = u.strip().lower().replace(" ", "")
                    if r == "Manager" and u == "admin" and p == "admin@123":
                        return True, "Admin User"
                    if p != f"{u}@123":
                        return False, None
                        
                    if r == "Mentor":
                        names = sorted(x for x in set(q("SELECT DISTINCT mentor_name FROM dim_mentor")["mentor_name"]) if x)
                        for n in names:
                            if n.replace(" ", "").lower() == u:
                                return True, n
                    elif r == "Intern":
                        names = sorted(x for x in set(q("SELECT DISTINCT full_name FROM dim_intern")["full_name"]) if x)
                        for n in names:
                            if n.replace(" ", "").lower() == u:
                                return True, n
                    return False, None

                ok, real_name = validate_login(username, password, role)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username.strip().lower().replace(" ", "")
                    st.session_state.real_name = real_name
                    st.session_state.role = role
                    st.rerun()
                else:
                    st.error("Invalid username, password, or role")
    st.stop()  # Halt execution of the dashboard until logged in

# ── Global KPI bar ────────────────────────────────────────────────────────────
r = st.session_state.role
n = st.session_state.real_name

c1, c2, c3, c4, c5 = st.columns(5)

if r == "Manager":
    total_interns  = q("SELECT COUNT(DISTINCT intern_id) AS n FROM dim_intern")["n"].iloc[0]
    total_hours    = q("SELECT COALESCE(ROUND(SUM(hours),1), 0) AS n FROM fact_eod_log")["n"].iloc[0]
    total_eod      = q("SELECT COUNT(*) AS n FROM fact_eod_log")["n"].iloc[0]
    avg_progress   = q("SELECT COALESCE(ROUND(AVG(progress_pct),1), 0) AS n FROM fact_lms_progress")["n"].iloc[0]
    completions    = q("SELECT COUNT(*) AS n FROM fact_lms_progress WHERE overall_status='Completed'")["n"].iloc[0]
elif r == "Mentor":
    mentees_query = f"""
        SELECT intern_id 
        FROM bridge_intern_mentor b 
        JOIN dim_mentor m ON b.mentor_id = m.mentor_id 
        WHERE m.mentor_name = '{n}'
    """
    total_interns  = q(f"SELECT COUNT(DISTINCT intern_id) AS n FROM ({mentees_query})")["n"].iloc[0]
    total_hours    = q(f"SELECT COALESCE(ROUND(SUM(hours),1), 0) AS n FROM fact_eod_log WHERE intern_id IN ({mentees_query})")["n"].iloc[0]
    total_eod      = q(f"SELECT COUNT(*) AS n FROM fact_eod_log WHERE intern_id IN ({mentees_query})")["n"].iloc[0]
    avg_progress   = q(f"SELECT COALESCE(ROUND(AVG(progress_pct),1), 0) AS n FROM fact_lms_progress WHERE intern_id IN ({mentees_query})")["n"].iloc[0]
    completions    = q(f"SELECT COUNT(*) AS n FROM fact_lms_progress WHERE intern_id IN ({mentees_query}) AND overall_status='Completed'")["n"].iloc[0]
elif r == "Intern":
    total_interns  = 1
    total_hours    = q(f"SELECT COALESCE(ROUND(SUM(hours),1), 0) AS n FROM fact_eod_log f JOIN dim_intern d ON f.intern_id = d.intern_id WHERE d.full_name='{n}'")["n"].iloc[0]
    total_eod      = q(f"SELECT COUNT(*) AS n FROM fact_eod_log f JOIN dim_intern d ON f.intern_id = d.intern_id WHERE d.full_name='{n}'")["n"].iloc[0]
    avg_progress   = q(f"SELECT COALESCE(ROUND(AVG(progress_pct),1), 0) AS n FROM fact_lms_progress f JOIN dim_intern d ON f.intern_id = d.intern_id WHERE d.full_name='{n}'")["n"].iloc[0]
    completions    = q(f"SELECT COUNT(*) AS n FROM fact_lms_progress f JOIN dim_intern d ON f.intern_id = d.intern_id WHERE d.full_name='{n}' AND f.overall_status='Completed'")["n"].iloc[0]
else:
    total_interns, total_hours, total_eod, avg_progress, completions = 0, 0, 0, 0, 0

for col, val, label in [
    (c1, total_interns,  "Total Interns"),
    (c2, f"{total_hours:,}", "Total Hours Logged"),
    (c3, f"{total_eod:,}", "EOD Activity Entries"),
    (c4, f"{avg_progress}%", "Avg LMS Progress"),
    (c5, completions, "Course Completions"),
]:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-number">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Tabs Setup ───────────────────────────────────────────────────────────────────
role = st.session_state.role

role_tabs = {
    "Manager": ["👔 Manager Dashboard", "🔍 EDA", "🤖 ML Insights", "💬 GenAI Chat", "📊 Warehouse Analysis"],
    "Mentor":  ["👨‍🏫 Mentor Dashboard", "🤖 ML Insights", "💬 GenAI Chat"],
    "Intern":  ["🎯 Intern Self-View", "💬 GenAI Chat"]
}

allowed_tab_names = role_tabs.get(role, [])

if "current_nav_manager" not in st.session_state:
    st.session_state.current_nav_manager = "👔 Manager Dashboard"

if role == "Manager":
    st.sidebar.markdown("### Manager Modules")
    import re
    def strip_emoji(text):
        return re.sub(r'^[^\w\s]+\s*', '', text).strip()
    
    for t in allowed_tab_names:
        clean_name = strip_emoji(t)
        is_active = (st.session_state.current_nav_manager == t)
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(clean_name, key=f"nav_{t}", type=btn_type, use_container_width=True):
            st.session_state.current_nav_manager = t
            st.rerun()
            
    selected_tab = st.session_state.current_nav_manager
    tab_dict = {selected_tab: st.container()}
else:
    rendered_tabs = st.tabs(allowed_tab_names)
    tab_dict = {name: tab for name, tab in zip(allowed_tab_names, rendered_tabs)}


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Data Quality Dashboard
# ════════════════════════════════════════════════════════════════════════════
def render_tab_data_quality():
    st.header("📊 Warehouse Analysis")

    # Row counts per layer
    st.subheader("Row Counts — Medallion Architecture")
    layers = {
        "Bronze": [
            ("raw_eod_activities", "EOD Activities"),
            ("raw_lms_python",     "LMS Python"),
            ("raw_lms_sql",        "LMS SQL"),
            ("raw_lms_numpy_pandas","LMS NumPy/Pandas"),
            ("raw_lms_pyspark",    "LMS PySpark"),
        ],
        "Silver": [
            ("dim_intern",          "dim_intern"),
            ("dim_course",          "dim_course"),
            ("dim_mentor",          "dim_mentor"),
            ("dim_activity",        "dim_activity"),
            ("fact_eod_log",        "fact_eod_log"),
            ("fact_lms_progress",   "fact_lms_progress"),
        ],
        "Gold": [
            ("gold_weekly_hours",       "gold_weekly_hours"),
            ("gold_activity_summary",   "gold_activity_summary"),
            ("gold_course_progress",    "gold_course_progress"),
            ("gold_intern_performance", "gold_intern_performance"),
            ("gold_mentor_workload",    "gold_mentor_workload"),
        ],
    }
    for layer_name, tables in layers.items():
        rows, labels = [], []
        for tbl, label in tables:
            try:
                n = get_conn().execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            except Exception:
                n = 0
            rows.append(n)
            labels.append(label)
        layer_df = pd.DataFrame({"Table": labels, "Row Count": rows})
        colors   = {"Bronze": "#cd7f32", "Silver": "#94a3b8", "Gold": "#f59e0b"}
        fig = px.bar(layer_df, x="Table", y="Row Count",
                     title=f"{layer_name} Layer", color_discrete_sequence=[colors[layer_name]])
        fig.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)



    # Null counts in key LMS columns
    st.subheader("Null Counts in LMS Fact Table")
    null_df = q("""
        SELECT
            COUNT(*) - COUNT(test_pct)       AS test_pct_nulls,
            COUNT(*) - COUNT(kc_pct)         AS kc_pct_nulls,
            COUNT(*) - COUNT(assignment_ratio) AS assignment_ratio_nulls,
            COUNT(*) - COUNT(progress_pct)   AS progress_pct_nulls
        FROM fact_lms_progress
    """).T.reset_index()
    null_df.columns = ["Column", "Null Count"]
    fig = px.bar(null_df, x="Column", y="Null Count",
                 color="Null Count", color_continuous_scale="Reds",
                 title="Null counts per column in fact_lms_progress")
    st.plotly_chart(fig, use_container_width=True)

    # Outlier flags
    n_ok  = q("SELECT COUNT(*) AS n FROM fact_eod_log WHERE hours_outlier_flag=0")["n"].iloc[0]
    n_bad = q("SELECT COUNT(*) AS n FROM fact_eod_log WHERE hours_outlier_flag=1")["n"].iloc[0]
    st.subheader("EOD Hours — Outlier Flags")
    out_df = pd.DataFrame({"Type": ["Normal (0.5–4.0 hrs)", "Outlier"], "Count": [n_ok, n_bad]})
    fig = px.bar(out_df, x="Type", y="Count", color="Type",
                 color_discrete_sequence=["#22c55e", "#ef4444"])
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — EDA Dashboard
# ════════════════════════════════════════════════════════════════════════════
def render_tab_eda():
    st.header("🔍 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Hours distribution
        hours_df = q("SELECT hours FROM fact_eod_log WHERE hours IS NOT NULL")
        fig = px.histogram(hours_df, x="hours", nbins=30,
                           title="Distribution of Hours per Activity Entry",
                           color_discrete_sequence=["#0ea5e9"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Activity type breakdown
        act_df = q("""
            SELECT a.activity_name, SUM(f.hours) AS total_hours
            FROM fact_eod_log f
            JOIN dim_activity a ON f.activity_id = a.activity_id
            GROUP BY a.activity_name
            ORDER BY total_hours DESC
        """)
        fig = px.bar(act_df, x="total_hours", y="activity_name",
                     orientation="h", title="Total Hours by Activity Type",
                     color="total_hours", color_continuous_scale="Blues")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Weekly hours heatmap (intern × week)
    st.subheader("Weekly Hours Heatmap (Intern × Week)")
    heat_df = q("""
        SELECT full_name, year || '-W' || LPAD(CAST(week_number AS VARCHAR), 2, '0') AS week_label,
               total_hours
        FROM gold_weekly_hours
        ORDER BY week_label, full_name
    """)
    if not heat_df.empty:
        pivot = heat_df.pivot_table(index="full_name", columns="week_label",
                                    values="total_hours", fill_value=0)
        fig = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto",
                        title="Weekly Hours per Intern (Jan–Mar 2026)")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Course progress distribution
        prog_df = q("SELECT course_name, progress_pct FROM gold_course_progress")
        fig = px.box(prog_df, x="course_name", y="progress_pct",
                     title="Progress % Distribution per Course",
                     color="course_name")
        fig.update_xaxes(tickangle=15)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Intern count per status per course
        status_df = q("""
            SELECT course_name, overall_status, COUNT(*) AS intern_count
            FROM gold_course_progress
            GROUP BY course_name, overall_status
        """)
        fig = px.bar(status_df, x="course_name", y="intern_count",
                     color="overall_status", barmode="stack",
                     title="Intern Status per Course",
                     color_discrete_map={
                         "Completed":   "#22c55e",
                         "In Progress": "#f59e0b",
                         "Not Started": "#ef4444",
                     })
        fig.update_xaxes(tickangle=15)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Manager View
# ════════════════════════════════════════════════════════════════════════════
def render_tab_manager():
    st.header("👔 Manager Dashboard")

    # Intern leaderboard
    lb_df = q("""
        SELECT full_name,
               ROUND(total_hours, 1)       AS total_hours,
               ROUND(avg_progress_pct, 1)  AS avg_progress_pct,
               ROUND(avg_test_pct, 1)      AS avg_test_pct,
               courses_completed,
               data_source
        FROM gold_intern_performance
        ORDER BY total_hours DESC, avg_progress_pct DESC
    """)

    st.subheader(f"Intern Leaderboard — {len(lb_df)} Interns")
    st.dataframe(lb_df, use_container_width=True, height=350)

    col1, col2 = st.columns(2)
    with col1:
        top5 = lb_df.head(5)
        fig = px.bar(top5, x="full_name", y="total_hours",
                     title="Top 5 Performers (by Hours)", color="total_hours",
                     color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bot5 = lb_df.tail(5)
        fig = px.bar(bot5, x="full_name", y="total_hours",
                     title="Bottom 5 Performers (by Hours)", color="total_hours",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    # Activity distribution per intern (top 10 interns)
    st.subheader("Activity Breakdown (Top 10 Interns by Hours)")
    top10_names = lb_df.head(10)["full_name"].tolist()
    act_intern = q("""
        SELECT full_name, activity_name, total_hours
        FROM gold_activity_summary
    """)
    act_intern_top = act_intern[act_intern["full_name"].isin(top10_names)]
    fig = px.bar(act_intern_top, x="full_name", y="total_hours",
                 color="activity_name", barmode="stack",
                 title="Activity Hours per Intern (Top 10)")
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    # Weekly trend
    st.subheader("Weekly Hours Trend (Batch Total)")
    trend_df = q("""
        SELECT year || '-W' || LPAD(CAST(week_number AS VARCHAR), 2, '0') AS week_label,
               SUM(total_hours) AS batch_hours
        FROM gold_weekly_hours
        GROUP BY week_label
        ORDER BY week_label
    """)
    fig = px.line(trend_df, x="week_label", y="batch_hours",
                  markers=True, title="Total Batch Hours per Week",
                  color_discrete_sequence=["#0ea5e9"])
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 4 — Mentor View
# ════════════════════════════════════════════════════════════════════════════
def render_tab_mentor():
    st.header("👨‍🏫 Mentor Dashboard")

    if st.session_state.role == "Mentor":
        selected_mentor = st.session_state.real_name
    else:
        mentor_names = q("SELECT DISTINCT mentor_name FROM dim_mentor ORDER BY mentor_name")["mentor_name"].tolist()
        selected_mentor = st.selectbox("Select Mentor", mentor_names)

    if selected_mentor:
        mentee_ids = q(f"""
            SELECT DISTINCT b.intern_id
            FROM bridge_intern_mentor b
            JOIN dim_mentor m ON b.mentor_id = m.mentor_id
            WHERE m.mentor_name = '{selected_mentor}'
        """)["intern_id"].tolist()

        if mentee_ids:
            ids_str = "','".join(mentee_ids)
            mentee_prog = q(f"""
                SELECT full_name, course_name, progress_pct, overall_status,
                       kc_pct, test_pct, assignment_ratio
                FROM gold_course_progress
                WHERE intern_id IN ('{ids_str}')
                ORDER BY course_name, progress_pct DESC
            """)

            col1, col2, col3 = st.columns(3)
            col1.metric("Mentees", len(set(mentee_prog["full_name"])))
            col2.metric("Avg Progress", f"{mentee_prog['progress_pct'].mean():.1f}%")
            col3.metric("Avg KC Score", f"{mentee_prog['kc_pct'].mean():.1f}%")

            # Progress heatmap per mentee × course
            st.subheader("Mentee Progress Heatmap")
            pivot = mentee_prog.pivot_table(index="full_name", columns="course_name",
                                            values="progress_pct", fill_value=0)
            fig = px.imshow(pivot, color_continuous_scale="RdYlGn",
                            title=f"Progress % — {selected_mentor}'s Mentees",
                            zmin=0, zmax=100)
            st.plotly_chart(fig, use_container_width=True)

            # Interns needing attention
            at_risk = mentee_prog[
                (mentee_prog["progress_pct"] < 50) &
                (mentee_prog["overall_status"] != "Completed")
            ][["full_name", "course_name", "progress_pct", "overall_status"]].drop_duplicates()

            st.subheader("⚠️ Interns Needing Attention (Progress < 50%)")
            if at_risk.empty:
                st.success("All mentees are on track! 🎉")
            else:
                st.dataframe(at_risk, use_container_width=True)

            # Knowledge Check score heatmap
            st.subheader("Knowledge Check Score Heatmap")
            kc_pivot = mentee_prog.pivot_table(index="full_name", columns="course_name",
                                               values="kc_pct", fill_value=0)
            fig = px.imshow(kc_pivot, color_continuous_scale="Blues",
                            title="KC Score % per Mentee × Course")
            st.plotly_chart(fig, use_container_width=True)

            # Not started
            not_started = mentee_prog[mentee_prog["overall_status"] == "Not Started"]
            st.subheader("Interns Who Haven't Started a Course")
            if not_started.empty:
                st.success("All mentees have started at least one course.")
            else:
                st.dataframe(not_started[["full_name", "course_name"]], use_container_width=True)
        else:
            st.info("No mentee data for this mentor.")

    # Overall mentor workload table
    st.subheader("All Mentor Workload Summary")
    mw = q("SELECT * FROM gold_mentor_workload ORDER BY intern_count DESC")
    st.dataframe(mw, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 5 — Intern View
# ════════════════════════════════════════════════════════════════════════════
def render_tab_intern():
    st.header("🎯 Intern Self-View Dashboard")

    if st.session_state.role == "Intern":
        selected_intern = st.session_state.real_name
    else:
        intern_names = q("SELECT DISTINCT full_name FROM dim_intern ORDER BY full_name")["full_name"].tolist()
        selected_intern = st.selectbox("Select Intern", intern_names)

    if selected_intern:
        intern_row = q(f"""
            SELECT * FROM gold_intern_performance
            WHERE full_name = '{selected_intern}'
        """)
        if not intern_row.empty:
            row = intern_row.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Hours",     f"{row['total_hours']:.1f}")
            col2.metric("Avg Progress",    f"{row['avg_progress_pct']:.1f}%")
            col3.metric("Avg Test Score",  f"{row['avg_test_pct']:.1f}%")
            col4.metric("Courses Completed", int(row["courses_completed"]))

            col1, col2 = st.columns(2)
            with col1:
                # Hours per activity
                my_acts = q(f"""
                    SELECT activity_name, total_hours
                    FROM gold_activity_summary
                    WHERE full_name = '{selected_intern}'
                    ORDER BY total_hours DESC
                """)
                if not my_acts.empty:
                    fig = px.bar(my_acts, x="activity_name", y="total_hours",
                                 title="Hours per Activity",
                                 color="total_hours", color_continuous_scale="Blues")
                    fig.update_xaxes(tickangle=30)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Course progress bars
                my_courses = q(f"""
                    SELECT course_name, progress_pct, overall_status
                    FROM gold_course_progress
                    WHERE full_name = '{selected_intern}'
                """)
                if not my_courses.empty:
                    fig = px.bar(my_courses, x="progress_pct", y="course_name",
                                 orientation="h", title="Course Progress",
                                 color="overall_status",
                                 color_discrete_map={
                                     "Completed":   "#22c55e",
                                     "In Progress": "#f59e0b",
                                     "Not Started": "#ef4444",
                                 })
                    fig.update_xaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)

            # Test score vs batch average
            st.subheader("Test Score vs Batch Average")
            test_compare = q(f"""
                SELECT
                    g.course_name,
                    COALESCE(my.test_pct, 0) AS my_score,
                    COALESCE(g.avg_test_pct, 0) AS batch_avg
                FROM (
                    SELECT course_name, AVG(test_pct) AS avg_test_pct
                    FROM gold_course_progress
                    GROUP BY course_name
                ) g
                LEFT JOIN (
                    SELECT course_name, test_pct
                    FROM gold_course_progress
                    WHERE full_name = '{selected_intern}'
                ) my ON g.course_name = my.course_name
                ORDER BY g.course_name
            """)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="My Score",    x=test_compare["course_name"], y=test_compare["my_score"],   marker_color="#0ea5e9"))
            fig.add_trace(go.Bar(name="Batch Avg",   x=test_compare["course_name"], y=test_compare["batch_avg"],  marker_color="#94a3b8"))
            fig.update_layout(barmode="group", title="Test Score vs Batch Average per Course")
            st.plotly_chart(fig, use_container_width=True)

            # Weekly hours
            st.subheader("Weekly Hours Trend")
            my_weeks = q(f"""
                SELECT year || '-W' || LPAD(CAST(week_number AS VARCHAR), 2, '0') AS week_label,
                       total_hours
                FROM gold_weekly_hours
                WHERE full_name = '{selected_intern}'
                ORDER BY week_label
            """)
            if not my_weeks.empty:
                fig = px.line(my_weeks, x="week_label", y="total_hours",
                              markers=True, color_discrete_sequence=["#6366f1"])
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 6 — ML Insights
# ════════════════════════════════════════════════════════════════════════════
def render_tab_ml():
    st.header("🤖 ML Insights")

    MODELS_DIR = os.path.join(ROOT, "ml", "models")

    col1, col2 = st.columns(2)

    # Cluster visualization (PCA scatter)
    with col1:
        st.subheader("Intern Clustering — PCA Scatter (K=3)")
        pca_path = os.path.join(MODELS_DIR, "pca_results.parquet")
        if os.path.exists(pca_path):
            import pandas as pd
            pca_df = pd.read_parquet(pca_path)
            fig = px.scatter(
                pca_df, x="pca_x", y="pca_y",
                color="cluster_label",
                symbol="data_source",
                hover_data=["full_name"],
                title="Intern Clusters (PCA 2D) — shape: real / synthetic",
                color_discrete_map={
                    "High Performer":    "#22c55e",
                    "Average Performer": "#f59e0b",
                    "At Risk":           "#ef4444",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster distribution
            dist = pca_df["cluster_label"].value_counts().reset_index()
            dist.columns = ["Cluster", "Count"]
            fig2 = px.pie(dist, names="Cluster", values="Count",
                          title="Cluster Distribution",
                          color_discrete_map={
                              "High Performer":    "#22c55e",
                              "Average Performer": "#f59e0b",
                              "At Risk":           "#ef4444",
                          })
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Run `python -m ml.train` to generate ML results.")

    # Regression: predicted vs actual test score
    with col2:
        st.subheader("Predicted vs Actual Test Score (Ridge Regression)")
        reg_path = os.path.join(MODELS_DIR, "regression_results.parquet")
        if os.path.exists(reg_path):
            reg_df = pd.read_parquet(reg_path)
            fig = px.scatter(
                reg_df, x="test_pct", y="predicted_test_pct",
                color="data_source", hover_data=["full_name", "course_name"],
                title="Actual vs Predicted Test Score (%)",
                labels={"test_pct": "Actual Test %", "predicted_test_pct": "Predicted Test %"},
                color_discrete_map={"real": "#0ea5e9", "synthetic": "#f59e0b"},
            )
            # Diagonal reference line
            mn = reg_df[["test_pct", "predicted_test_pct"]].min().min()
            mx = reg_df[["test_pct", "predicted_test_pct"]].max().max()
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                     mode="lines", name="Perfect Fit",
                                     line=dict(dash="dash", color="gray")))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Run `python -m ml.train` to generate regression results.")

    # Classification results table
    st.subheader("Performance Classification per Intern (All Courses)")
    clust_path = os.path.join(MODELS_DIR, "cluster_results.parquet")
    if os.path.exists(clust_path):
        clust_df = pd.read_parquet(clust_path)
        show_cols = ["full_name", "cluster_label", "total_hours",
                     "avg_progress_pct", "avg_test_pct", "courses_completed", "data_source"]
        existing   = [c for c in show_cols if c in clust_df.columns]
        st.dataframe(clust_df[existing].sort_values("total_hours", ascending=False),
                     use_container_width=True)

    # Live prediction widget
    st.subheader("🔮 Live Prediction — Enter Intern Stats")
    with st.form("predict_form"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        prog_pct = fc1.slider("Progress %",         0, 100, 60)
        assign_r = fc2.slider("Assignment Ratio %",  0, 100, 75)
        kc_pct   = fc3.slider("KC Score %",          0, 100, 70)
        test_pct_in = fc4.slider("Test Score %",     0, 100, 65)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            from ml.predict import classify_intern_status, predict_test_score
            status_result = classify_intern_status(prog_pct, assign_r / 100, kc_pct, test_pct_in)
            pred_score    = predict_test_score(prog_pct, kc_pct, assign_r / 100)
            rc1, rc2 = st.columns(2)
            rc1.success(f"Predicted Status: **{status_result['status']}**")
            rc2.info(f"Predicted Test Score: **{pred_score:.1f}%**")
            # Probabilities bar
            prob_df = pd.DataFrame(list(status_result["probabilities"].items()),
                                   columns=["Status", "Probability"])
            fig = px.bar(prob_df, x="Status", y="Probability",
                         title="Classification Confidence",
                         color="Status",
                         color_discrete_map={
                             "Completed":   "#22c55e",
                             "In Progress": "#f59e0b",
                             "Not Started": "#ef4444",
                         })
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# Tab 7 — GenAI Chatbot
# ════════════════════════════════════════════════════════════════════════════
def render_tab_chat():
    st.header("💬 GenAI RAG Chatbot")
    st.markdown("""
    **Ask natural language questions about intern data.**  
    *Examples:*
    - *"Who hasn't started PySpark yet?"*
    - *"Which intern logged the most hours this week?"*
    - *"Show me top performers in SQL"*
    - *"Which interns are at risk of not completing NumPy?"*
    - *"Summarize batch progress across all courses"*
    """)

    CHROMA_DIR = os.path.join(ROOT, os.getenv("CHROMA_PERSIST_DIR", "genai/chroma_store"))

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_q = st.chat_input("Ask a question about intern performance…")

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                if not os.path.exists(CHROMA_DIR):
                    answer = (
                        "⚠️ Vector store not found. "
                        "Please run `python -m genai.embeddings` first, "
                        "then restart the app."
                    )
                    sources = []
                    mode    = "unavailable"
                else:
                    try:
                        from genai.chatbot import ask
                        result  = ask(user_q)
                        answer  = result["answer"]
                        sources = result.get("sources", [])
                        mode    = result.get("mode", "unknown")
                    except Exception as e:
                        answer  = f"Error during retrieval: {e}"
                        sources = []
                        mode    = "error"

            st.markdown(answer)
            if mode != "unavailable":
                st.caption(f"Mode: `{mode}` | Sources retrieved: {len(sources)}")
            if sources:
                with st.expander("📚 Source chunks"):
                    for s in sources:
                        st.markdown(f"- {s[:200]}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── Render Allowed Tabs ───────────────────────────────────────────────────────
if "📊 Warehouse Analysis" in tab_dict:
    with tab_dict["📊 Warehouse Analysis"]: render_tab_data_quality()
if "🔍 EDA" in tab_dict:
    with tab_dict["🔍 EDA"]: render_tab_eda()
if "👔 Manager Dashboard" in tab_dict:
    with tab_dict["👔 Manager Dashboard"]: render_tab_manager()
if "👨‍🏫 Mentor Dashboard" in tab_dict:
    with tab_dict["👨‍🏫 Mentor Dashboard"]: render_tab_mentor()
if "🎯 Intern Self-View" in tab_dict:
    with tab_dict["🎯 Intern Self-View"]: render_tab_intern()
if "🤖 ML Insights" in tab_dict:
    with tab_dict["🤖 ML Insights"]: render_tab_ml()
if "💬 GenAI Chat" in tab_dict:
    with tab_dict["💬 GenAI Chat"]: render_tab_chat()


# ── Sidebar Bottom Details ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("## 🎓 InternIQ")
    st.markdown(f"👤 **{st.session_state.real_name}** ({st.session_state.role})")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Logout", key="logout_btn", type="tertiary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.real_name = None
            st.session_state.role = None
            st.rerun()
    with col2:
        if st.button("Refresh", key="refresh_btn", type="tertiary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

