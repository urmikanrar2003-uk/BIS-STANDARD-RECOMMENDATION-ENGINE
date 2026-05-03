"""
config.py
─────────
Central configuration for the BIS Standard Recommendation Engine.
All paths, model names, and API keys are defined here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = _root
DATA_DIR       = BASE_DIR / "data"
SRC_DIR        = BASE_DIR / "src"

PDF_PATH       = DATA_DIR / "dataset.pdf"
STANDARDS_JSON = DATA_DIR / "bis_standards.json"
FAISS_INDEX_DIR= DATA_DIR / "faiss_index"
RESULTS_DIR    = DATA_DIR / "results"

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE = 20   # candidates pulled before LLM reranking
TOP_K_FINAL    = 5    # final recommendations returned

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o-mini"
