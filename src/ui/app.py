"""
TicketMind - Simple & Smart Support Ticket Analytics
"""

import pandas as pd
import streamlit as st

from src import config
from src.anomalies.rules import detect_anomalies
from src.data.store import get_dataframe, init_store
from src.query.llm import call_llm
from src.query.parser import QueryParseError, parse_query_spec
from src.query.prompt import build_messages
from src.query.runner import run_query

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TicketMind — Support Ticket Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Clean Light Theme Styles & Icon Fix
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global Typography */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1E293B;
    }

    /* Page Background */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC !important;
    }

    /* Streamlit Material Icons font fix */
    [data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, [class*="material-symbols"] {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        line-height: 1;
        display: inline-block;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: normal;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }

    /* Hide Default Header & Footer */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Main Container Width */
    .block-container {
        max-width: 1180px !important;
        padding-top: 36px !important;
        padding-bottom: 70px !important;
        padding-left: 36px !important;
        padding-right: 36px !important;
    }

    /* Header Card */
    .tm-header-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .tm-brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .tm-brand-icon {
        width: 44px;
        height: 44px;
        background: #2563EB;
        color: #FFFFFF;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 700;
        flex-shrink: 0;
    }

    .tm-brand-title {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }

    .tm-brand-subtitle {
        font-size: 15px;
        color: #64748B;
        margin: 4px 0 0 0;
    }

    .tm-header-tags {
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
    }

    .tm-pill {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #334155;
        font-size: 13.5px;
        font-weight: 500;
        padding: 6px 14px;
        border-radius: 8px;
    }

    .tm-pill strong {
        color: #0F172A;
    }

    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #F1F5F9 !important;
        padding: 6px !important;
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        margin-bottom: 24px !important;
        width: fit-content !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        padding: 0 22px !important;
        border-radius: 8px !important;
        background-color: transparent !important;
        border: none !important;
        color: #64748B !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Input Field */
    div[data-testid="stTextInput"] label {
        font-size: 15.5px !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        padding: 12px 16px !important;
        font-size: 15.5px !important;
        height: 48px !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #94A3B8 !important;
        font-size: 15px !important;
    }

    /* Primary Search Button */
    button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15.5px !important;
        height: 48px !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
    }

    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
    }

    /* Quick Question Prompt Buttons */
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 8px 14px !important;
        height: auto !important;
        min-height: 38px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #EFF6FF !important;
        border-color: #BFDBFE !important;
        color: #1D4ED8 !important;
    }

    /* Result Insight Card */
    .tm-answer-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 22px 26px;
        margin-top: 24px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .tm-answer-badge {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #2563EB;
        background: #EFF6FF;
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 10px;
    }

    .tm-answer-text {
        font-size: 18px;
        font-weight: 600;
        color: #0F172A;
        line-height: 1.5;
    }

    /* KPI Summary Cards */
    .tm-kpi-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        margin-bottom: 28px;
    }

    .tm-kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .tm-kpi-label {
        font-size: 14px;
        font-weight: 500;
        color: #64748B;
        margin-bottom: 6px;
    }

    .tm-kpi-number {
        font-size: 30px;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }

    .tm-kpi-desc {
        font-size: 12.5px;
        color: #94A3B8;
        margin-top: 4px;
    }

    .tm-section-heading {
        font-size: 16px;
        font-weight: 600;
        color: #0F172A;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    /* Custom Pure Light Mode Scrollable HTML Table */
    .tm-table-container {
        width: 100%;
        max-height: 420px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        margin-bottom: 24px;
    }

    .tm-table-container::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    .tm-table-container::-webkit-scrollbar-track {
        background: #F8FAFC;
        border-radius: 8px;
    }
    .tm-table-container::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 8px;
    }
    .tm-table-container::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }

    .tm-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13.5px;
        text-align: left;
        background: #FFFFFF;
    }

    .tm-table thead th {
        position: sticky;
        top: 0;
        background: #F8FAFC;
        color: #475569;
        font-weight: 600;
        padding: 12px 16px;
        border-bottom: 1px solid #E2E8F0;
        white-space: nowrap;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        z-index: 5;
    }

    .tm-table tbody td {
        padding: 12px 16px;
        color: #0F172A;
        border-bottom: 1px solid #F1F5F9;
        white-space: nowrap;
    }

    .tm-table tbody tr:last-child td {
        border-bottom: none;
    }

    .tm-table tbody tr:hover {
        background-color: #F8FAFC;
    }

    /* Badges in Table Cells */
    .badge-priority-critical {
        background: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-priority-high {
        background: #FFEDD5;
        color: #9A3412;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-priority-medium {
        background: #FEF3C7;
        color: #92400E;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-priority-low {
        background: #F1F5F9;
        color: #475569;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-status-resolved {
        background: #DCFCE7;
        color: #166534;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-status-open {
        background: #EFF6FF;
        color: #1E40AF;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .badge-status-escalated {
        background: #F3E8FF;
        color: #6B21A8;
        font-weight: 600;
        font-size: 11.5px;
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        margin-top: 18px !important;
    }

    div[data-testid="stExpander"] summary {
        color: #334155 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HTML Table Renderer (Guaranteed Pure Light Theme & Clear Text)
# -----------------------------------------------------------------------------
def render_light_table(data: list[dict] | pd.DataFrame) -> None:
    """Render a clean, responsive, pure light HTML table with badges."""
    if isinstance(data, list):
        df_display = pd.DataFrame(data)
    else:
        df_display = data.copy()

    if df_display.empty:
        st.info("No records to display.")
        return

    columns = list(df_display.columns)

    header_html = "".join([f"<th>{col.replace('_', ' ')}</th>" for col in columns])
    rows_html = []

    for _, row in df_display.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if pd.isna(val) or val is None:
                cell_content = "<span style='color:#94A3B8;'>—</span>"
            else:
                str_val = str(val)
                # Format Priority Badges
                if col.lower() == "priority":
                    p_class = f"badge-priority-{str_val.lower()}"
                    cell_content = f"<span class='{p_class}'>{str_val}</span>"
                # Format Status Badges
                elif col.lower() == "status":
                    s_class = f"badge-status-{str_val.lower()}"
                    cell_content = f"<span class='{s_class}'>{str_val}</span>"
                # Format Agent Badges
                elif col.lower() == "agent_id":
                    cell_content = f"<strong style='color:#1E293B;'>{str_val}</strong>"
                # Customer rating
                elif col.lower() == "customer_rating":
                    cell_content = f"⭐ <strong>{str_val}</strong>/5"
                else:
                    cell_content = str_val

            cells.append(f"<td>{cell_content}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f"""
    <div class="tm-table-container">
        <table class="tm-table">
            <thead>
                <tr>{header_html}</tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def _load_data() -> None:
    config.validate()
    init_store(config.CSV_PATH)

_load_data()
df = get_dataframe()

max_date_str = str(df["created_at"].max()).split(" ")[0]

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="tm-header-card">
        <div class="tm-brand-group">
            <div class="tm-brand-icon">⚡</div>
            <div>
                <h1 class="tm-brand-title">TicketMind</h1>
                <p class="tm-brand-subtitle">Ask questions in plain English and track delayed or unresolved support tickets</p>
            </div>
        </div>
        <div class="tm-header-tags">
            <span class="tm-pill">Total Tickets: <strong>{len(df):,}</strong></span>
            <span class="tm-pill">Latest Date: <strong>{max_date_str}</strong></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Tabs
# -----------------------------------------------------------------------------
tab_query, tab_anomalies = st.tabs(["Ask a Question", "Delayed & Urgent Tickets"])

# -----------------------------------------------------------------------------
# TAB 1: Ask Questions
# -----------------------------------------------------------------------------
with tab_query:
    if "user_question_text" not in st.session_state:
        st.session_state["user_question_text"] = ""

    def select_prompt(text: str) -> None:
        st.session_state["user_question_text"] = text

    # Search Bar Row
    col_input, col_button = st.columns([5, 1], vertical_alignment="bottom")
    with col_input:
        entered_question = st.text_input(
            "Type your question below:",
            value=st.session_state["user_question_text"],
            placeholder="e.g. Which agent has the lowest customer rating?",
            key="input_search_box",
        )
    with col_button:
        btn_clicked = st.button("Search", type="primary", use_container_width=True)

    # Example question chips
    st.markdown("<p style='font-size: 13.5px; font-weight: 600; color: #64748B; margin: 14px 0 8px 0;'>Example questions you can click to try:</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Lowest rated agent?", key="chip_agent", use_container_width=True):
            select_prompt("Which agent has the lowest average customer rating?")
            st.rerun()
    with c2:
        if st.button("How many tickets are open?", key="chip_open", use_container_width=True):
            select_prompt("How many tickets are currently open?")
            st.rerun()
    with c3:
        if st.button("Avg technical rating?", key="chip_rating", use_container_width=True):
            select_prompt("What is the average customer rating for Technical category tickets?")
            st.rerun()
    with c4:
        if st.button("Critical unresolved tickets?", key="chip_crit", use_container_width=True):
            select_prompt("Show me all Critical tickets not resolved within 12 hours.")
            st.rerun()

    # Query Execution
    question_to_run = entered_question.strip()
    if btn_clicked and question_to_run:
        with st.spinner("Finding answer..."):
            reference_date = str(df["created_at"].max())
            try:
                messages = build_messages(question_to_run, reference_date)
                raw_output = call_llm(messages)
                spec = parse_query_spec(raw_output)
                result = run_query(df, spec)

                # Big Clear Answer Box
                st.markdown(
                    f"""
                    <div class="tm-answer-card">
                        <div class="tm-answer-badge">Answer</div>
                        <div class="tm-answer-text">{result["answer"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show table if there are multiple records
                if isinstance(result["result"], list) and result["result"]:
                    st.markdown("<div class='tm-section-heading'>Matching Tickets</div>", unsafe_allow_html=True)
                    render_light_table(result["result"])

                # Simple technical explanation
                with st.expander("How this answer was calculated (Details)", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown(f"**Calculation Type:** `{spec.metric}`")
                    col_b.markdown(f"**Calculated Column:** `{spec.metric_column or 'None'}`")
                    col_c.markdown(f"**Grouped By:** `{spec.group_by or 'None'}`")

                    if spec.filters:
                        st.markdown("**Applied Filters:**")
                        st.json([f.model_dump() for f in spec.filters])

            except QueryParseError as exc:
                st.error(f"Could not understand the question. Please try phrasing it simply: {exc}")
            except RuntimeError as exc:
                st.error(f"Error connecting to AI service: {exc}")

# -----------------------------------------------------------------------------
# TAB 2: Delayed & Urgent Tickets
# -----------------------------------------------------------------------------
with tab_anomalies:
    report = detect_anomalies(
        df,
        resolution_percentile=config.ANOMALY_RESOLUTION_PERCENTILE,
        stale_hours=config.ANOMALY_STALE_HOURS,
    )

    long_res_list = report.get("long_resolution_time", [])
    stale_list = report.get("stale_high_priority", [])
    cutoff_hours = report.get("resolution_time_threshold_hrs", 0)

    # 3 Plain-English Summary Cards
    st.markdown(
        f"""
        <div class="tm-kpi-row">
            <div class="tm-kpi-card">
                <div class="tm-kpi-label">Resolution Time Limit</div>
                <div class="tm-kpi-number">{cutoff_hours} hrs</div>
                <div class="tm-kpi-desc">Cutoff for slowest 10% resolved tickets</div>
            </div>
            <div class="tm-kpi-card">
                <div class="tm-kpi-label">Unusually Slow Tickets</div>
                <div class="tm-kpi-number" style="color: #D97706;">{len(long_res_list)}</div>
                <div class="tm-kpi-desc">Took more than {cutoff_hours} hrs to resolve</div>
            </div>
            <div class="tm-kpi-card">
                <div class="tm-kpi-label">Stuck Urgent Tickets</div>
                <div class="tm-kpi-number" style="color: #DC2626;">{len(stale_list)}</div>
                <div class="tm-kpi-desc">High/Critical & open for over {config.ANOMALY_STALE_HOURS} hrs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 1: Slow resolved tickets
    st.markdown(f"<div class='tm-section-heading'>Tickets That Took Unusually Long to Resolve ({len(long_res_list)})</div>", unsafe_allow_html=True)
    if long_res_list:
        render_light_table(long_res_list)
    else:
        st.info("Great news! No tickets exceeded the resolution time limit.")

    # Section 2: Stuck urgent tickets
    st.markdown(f"<div class='tm-section-heading'>Urgent Tickets Waiting for More Than {config.ANOMALY_STALE_HOURS} Hours ({len(stale_list)})</div>", unsafe_allow_html=True)
    if stale_list:
        render_light_table(stale_list)
    else:
        st.info("Great news! No urgent tickets are currently stuck.")
