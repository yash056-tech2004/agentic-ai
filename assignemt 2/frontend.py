from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
OUTPUTS_DIR = ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import run_research, save_report  # noqa: E402

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.25
AUTHOR = "Ravi Deo (System ID: 2023458249)"

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1219;
    --panel: rgba(16, 24, 32, 0.78);
    --accent: #38bdf8;
    --accent-2: #f59e0b;
    --muted: #9fb4c7;
    --border: rgba(56, 189, 248, 0.25);
    --glow: rgba(56, 189, 248, 0.18);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: #e6edf5;
}

.stApp {
    background:
        radial-gradient(900px 500px at 15% 15%, rgba(56, 189, 248, 0.12), transparent),
        radial-gradient(800px 500px at 80% 10%, rgba(245, 158, 11, 0.10), transparent),
        linear-gradient(170deg, #050b11 0%, #0b1219 50%, #0a1823 100%);
}

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    position: relative;
    border-radius: 22px;
    padding: 2.2rem 2rem 2rem;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.16), rgba(245, 158, 11, 0.12));
    border: 1px solid rgba(56, 189, 248, 0.25);
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
    overflow: hidden;
    margin-bottom: 1.2rem;
}

.hero::after {
    content: '';
    position: absolute;
    inset: -30% auto auto -30%;
    width: 65%;
    height: 65%;
    background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.18), transparent 55%);
    filter: blur(28px);
    opacity: 0.85;
}

.hero h1 {
    position: relative;
    margin: 0 0 0.4rem 0;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    letter-spacing: -0.4px;
    color: #e2f3ff;
}

.hero p {
    position: relative;
    margin: 0 0 0.9rem 0;
    color: var(--muted);
    line-height: 1.55;
    font-size: 1.02rem;
}

.credit {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: #0b1219;
    background: linear-gradient(135deg, #38bdf8, #f59e0b);
    padding: 0.55rem 1.1rem;
    border-radius: 999px;
    font-size: 0.92rem;
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(56, 189, 248, 0.35);
}

.panel {
    border-radius: 18px;
    padding: 1.3rem;
    margin-top: 1rem;
    background: var(--panel);
    border: 1px solid var(--border);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.7rem;
}

.section-header .icon {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(245, 158, 11, 0.16));
    border: 1px solid var(--border);
}

.section-header h3 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e6edf5;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid var(--border);
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    color: #c6d5e7;
}

[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label {
    color: #d7e8f8;
    font-weight: 600;
    font-size: 0.94rem;
}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: rgba(12, 18, 27, 0.9);
    color: #e6edf5;
    border: 1px solid var(--border);
    border-radius: 12px;
    font-size: 0.96rem;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(245, 158, 11, 0.65);
    box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.18);
}

[data-testid="stButton"] button {
    border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.55);
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: #031018;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 0.2px;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.35);
}

[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    transform: translateY(-1px);
}

[data-testid="stDownloadButton"] button {
    border-radius: 12px;
    border: 1px solid rgba(245, 158, 11, 0.55);
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    color: #0b1219;
    font-weight: 800;
    font-size: 0.95rem;
    box-shadow: 0 8px 20px rgba(245, 158, 11, 0.3);
}

.delete-btn button {
    border-radius: 12px;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18), rgba(239, 68, 68, 0.12)) !important;
    color: #fecdd3 !important;
}

.report-card {
    border-radius: 18px;
    padding: 1.2rem;
    margin-top: 0.8rem;
    margin-bottom: 1rem;
    background: linear-gradient(150deg, rgba(13, 23, 32, 0.9), rgba(9, 16, 24, 0.85));
    border: 1px solid var(--border);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.stDivider {
    margin: 0.9rem 0 1.1rem;
    border-color: rgba(56, 189, 248, 0.16) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

load_dotenv(ROOT / ".env")


def list_recent_reports() -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        [p for p in OUTPUTS_DIR.glob("report_*.md") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def hydrate_state() -> None:
    if "report_paths" not in st.session_state:
        st.session_state["report_paths"] = list_recent_reports()
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    if "report" not in st.session_state:
        st.session_state["report"] = ""
    if "output_path" not in st.session_state:
        st.session_state["output_path"] = ""
    if "scroll_top" not in st.session_state:
        st.session_state["scroll_top"] = False


hydrate_state()

if st.session_state["scroll_top"]:
    components.html("<script>window.parent.scrollTo({top: 0, behavior: 'smooth'});</script>", height=0)
    st.session_state["scroll_top"] = False

# Hero
st.markdown(
    """
<div class="hero">
  <h1>🧭 Autonomous Research Agent</h1>
  <p>Generate compact, source-aware research briefs in a single click. Give us a topic, we do the digging.</p>
  <div class="credit">Built by Ravi Deo — 2023458249</div>
</div>
""",
    unsafe_allow_html=True,
)

paths: list[Path] = st.session_state["report_paths"]
report_names = [p.name for p in paths] if paths else []

if report_names and "selected_report_name" not in st.session_state:
    st.session_state["selected_report_name"] = report_names[0]

selected_name = st.session_state.get("selected_report_name", report_names[0] if report_names else "")
selected_index = report_names.index(selected_name) if (report_names and selected_name in report_names) else 0

num_reports = len(paths)
st.markdown(
    f"""
<div style="display:flex;gap:0.7rem;flex-wrap:wrap;margin-bottom:1rem;">
  <span class="stat-pill">📄 {num_reports} saved</span>
  <span class="stat-pill">⚡ Groq + DuckDuckGo + Wikipedia</span>
  <span class="stat-pill">🛠 ReAct agent</span>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# Create Report
st.markdown(
    '<div class="section-header"><div class="icon">✍️</div><h3>Create New Report</h3></div>',
    unsafe_allow_html=True,
)

topic = st.text_input(
    "Research Topic",
    value="Future of renewable energy storage",
    placeholder="e.g. Quantum networks, Climate-resilient crops, Edge AI for robotics",
)

col_gen, col_info = st.columns([1, 2])
with col_gen:
    run_clicked = st.button("🚀 Generate Report", use_container_width=True, type="primary")
with col_info:
    st.caption("A new report will appear at the top when ready.")

if run_clicked:
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("🔎 Researching and drafting report..."):
            try:
                report = run_research(
                    topic=topic.strip(),
                    provider=DEFAULT_PROVIDER,
                    model=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                )
                output_path = save_report(topic.strip(), report, OUTPUTS_DIR)

                st.session_state["report"] = report
                st.session_state["output_path"] = str(output_path)
                st.session_state["report_paths"] = list_recent_reports()
                st.session_state["selected_report_name"] = output_path.name
                st.session_state["scroll_top"] = True
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not generate the report right now. Please try again. ({exc})")

st.divider()

# Current Report
if paths:
    selected_path = paths[selected_index]
    if selected_path.exists():
        st.session_state["report"] = selected_path.read_text(encoding="utf-8")
        st.session_state["output_path"] = str(selected_path)

if st.session_state["report"]:
    st.markdown(
        '<div class="section-header"><div class="icon">📋</div><h3>Current Report</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="report-card">' + "</div>", unsafe_allow_html=True)
    st.markdown(st.session_state["report"])
    st.download_button(
        label="⬇️ Download Report (.md)",
        data=st.session_state["report"],
        file_name=Path(st.session_state["output_path"]).name if st.session_state["output_path"] else "research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()

# Recent Reports
st.markdown(
    '<div class="section-header"><div class="icon">📚</div><h3>Recent Reports</h3></div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 1])
with col1:
    selected_name = st.selectbox(
        "Recent Reports",
        options=report_names if report_names else ["No reports yet"],
        index=selected_index if report_names else 0,
        disabled=not report_names,
        key="selected_report_name",
        on_change=lambda: st.session_state.update({"scroll_top": True}),
    )
with col2:
    st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
    delete_clicked = st.button("🗑️ Delete", use_container_width=True, disabled=not report_names, key="delete_btn")
    st.markdown("</div>", unsafe_allow_html=True)

if report_names and delete_clicked:
    to_delete = paths[selected_index]
    try:
        to_delete.unlink(missing_ok=True)
        st.session_state["report_paths"] = list_recent_reports()
        updated_paths = st.session_state["report_paths"]
        if not updated_paths:
            st.session_state["report"] = ""
            st.session_state["output_path"] = ""
            st.session_state["selected_report_name"] = ""
        else:
            next_index = min(selected_index, len(updated_paths) - 1)
            st.session_state["selected_report_name"] = updated_paths[next_index].name
            st.session_state["report"] = updated_paths[next_index].read_text(encoding="utf-8")
            st.session_state["output_path"] = str(updated_paths[next_index])
        st.session_state["scroll_top"] = True
        st.rerun()
    except OSError:
        st.error("Could not delete the selected report.")
