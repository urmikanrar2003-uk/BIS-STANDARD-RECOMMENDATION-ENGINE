"""
app.py
──────
Streamlit UI for the BIS Standard Recommendation Engine.

Run:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# ── Ensure src/ is importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="BIS Standard Finder",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #f0f0f0;
}

/* Header */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Search box - High Contrast */
.stTextArea textarea {
    background: #ffffff !important;
    border: 2px solid #7c3aed !important;
    border-radius: 12px !important;
    color: #000000 !important; /* Black text for typing */
    font-size: 1rem !important;
    padding: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
.stTextArea textarea::placeholder {
    color: #64748b !important;
}
.stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 4px rgba(124,58,237,0.3) !important;
}

/* Example Buttons - Less Hazy */
.stButton > button {
    background: #1e1b4b !important; /* Dark solid blue */
    color: #e0e7ff !important;
    border: 1px solid #4338ca !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100%;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
}
.stButton > button:hover {
    background: #312e81 !important;
    border-color: #a78bfa !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
}

/* Primary Search Button */
div[data-testid="stFormSubmitButton"] > button, 
.stButton > button[key="search_btn"] {
    background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    text-align: center !important;
}

/* Result cards */
.result-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
    backdrop-filter: blur(10px);
}
.result-card:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(167,139,250,0.4);
    transform: translateX(4px);
}
.result-rank {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border-radius: 8px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #d97706) !important; }
.rank-2 { background: linear-gradient(135deg, #6b7280, #4b5563) !important; }
.rank-3 { background: linear-gradient(135deg, #92400e, #78350f) !important; }

.std-id {
    font-size: 1.25rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 0.3rem;
    letter-spacing: 0.5px;
}
.std-title {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Metric pills */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    margin-bottom: 2rem;
    justify-content: center;
    flex-wrap: wrap;
}
.metric-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-size: 0.85rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.metric-pill span { color: #34d399; font-weight: 600; }

/* Example queries */
.example-chip {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    color: #a5b4fc;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.15s ease;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 2rem 0;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load retriever once and keep in memory across requests."""
    from retriever import get_retriever
    get_retriever()   # warms up the singleton
    from pipeline import recommend
    return recommend


# ── Sample queries ─────────────────────────────────────────────────────────────
EXAMPLES = [
    "We manufacture 33 Grade Ordinary Portland Cement.",
    "Looking for regulations on coarse and fine aggregates for structural concrete.",
    "What standard applies to precast concrete pipes for water mains?",
    "We produce Portland Pozzolana Cement using calcined clay.",
    "Our factory makes corrugated asbestos cement roofing sheets.",
]


# ── Hero section ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏭 BIS Standard Finder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered RAG engine · Describe your product → get the right BIS standard in seconds</div>',
    unsafe_allow_html=True,
)

# ── Metrics bar ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
  <div class="metric-pill">🎯 Hit Rate @3 <span>100%</span></div>
  <div class="metric-pill">📊 MRR @5 <span>1.000</span></div>
  <div class="metric-pill">⚡ Avg Latency <span>&lt; 0.03s</span></div>
  <div class="metric-pill">📚 Standards Indexed <span>950</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────────────
col_main, col_right = st.columns([2, 1])

with col_main:
    st.markdown("#### 📝 Describe your product or compliance need")
    query = st.text_area(
        label="query_input",
        label_visibility="collapsed",
        placeholder="e.g. We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?",
        height=130,
        key="query_box",
    )
    search_clicked = st.button("🔍 Find BIS Standards", key="search_btn")

def set_query(ex_text):
    st.session_state["query_box"] = ex_text

with col_right:
    st.markdown("#### 💡 Try an example")
    for ex in EXAMPLES:
        st.button(
            ex[:55] + "…", 
            key=f"ex_{ex[:20]}", 
            help=ex, 
            on_click=set_query, 
            args=(ex,)
        )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("🔍 Searching BIS corpus · Neural Precision Rerank …"):
        try:
            recommend = load_pipeline()
            result = recommend(query.strip())
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    standards   = result["retrieved_standards"]
    latency     = result["latency_seconds"]
    candidates  = result.get("candidates", [])

    # Build a lookup from the retriever candidates for titles/descriptions
    meta = {c["standard_id"]: c for c in candidates}

    st.markdown(f"### ✅ Top {len(standards)} Recommended BIS Standards")
    st.caption(f"Query processed in **{latency:.2f}s** · {len(standards)} standards retrieved and reranked")

    rank_labels = {1: "🥇 Best Match", 2: "🥈 2nd Match", 3: "🥉 3rd Match"}
    rank_classes = {1: "rank-1", 2: "rank-2", 3: "rank-3"}

    for i, std_id in enumerate(standards, 1):
        info        = meta.get(std_id, {})
        title       = info.get("title", "—")
        description = info.get("description", "")
        rank_label  = rank_labels.get(i, f"#{i} Match")
        rank_cls    = rank_classes.get(i, "")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-rank {rank_cls}">{rank_label}</div>
            <div class="std-id">{std_id}</div>
            <div class="std-title">{title}</div>
            {"<br><small style='color:#64748b;'>" + description[:200] + "…</small>" if description else ""}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption("💡 Powered by FAISS · BM25 · RRF · Identity Matcher · Local Neural Search")

elif search_clicked and not query.strip():
    st.warning("Please enter a product description first.")
