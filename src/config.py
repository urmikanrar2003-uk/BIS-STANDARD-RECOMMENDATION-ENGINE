"""
config.py
─────────
Central configuration for the BIS Standard Recommendation Engine.
All paths and model names are defined here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_root = Path(__file__).parent.parent
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
TOP_K_RETRIEVE = 50   # Wider pool to ensure precision matching finds the winner
TOP_K_FINAL    = 5    # final recommendations returned
