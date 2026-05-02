"""
embedder.py
───────────
Builds a FAISS vector index from bis_standards.json.

Outputs to data/faiss_index/:
  • standards.index  — FAISS flat inner-product index
  • metadata.pkl     — list of standard dicts (id, title, description, keywords)
  • chunks.pkl       — list of text strings used for embedding
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import STANDARDS_JSON, FAISS_INDEX_DIR, EMBEDDING_MODEL


def _build_chunk(std: dict) -> str:
    """
    Combine standard_id + title + description into one rich text chunk.
    The embedding model encodes this to capture full semantic meaning.
    """
    return f"{std['standard_id']} | {std['title']} | {std['description']}"


def build_index(
    standards_path: Path = STANDARDS_JSON,
    index_dir:      Path = FAISS_INDEX_DIR,
    model_name:     str  = EMBEDDING_MODEL,
    batch_size:     int  = 64,
) -> None:
    """Build and save the FAISS index from the standards JSON."""

    # ── 1. Load standards ──────────────────────────────────────────────────────
    if not standards_path.exists():
        raise FileNotFoundError(
            f"Standards JSON not found at '{standards_path}'. "
            "Run 'python src/data_ingestion.py' first."
        )
    with open(standards_path, encoding="utf-8") as f:
        standards: list[dict] = json.load(f)
    print(f"[1/4] Loaded {len(standards)} standards from '{standards_path}'.")

    # ── 2. Build text chunks ───────────────────────────────────────────────────
    chunks = [_build_chunk(s) for s in standards]
    print(f"[2/4] Built {len(chunks)} text chunks.")

    # ── 3. Encode with SentenceTransformer ────────────────────────────────────
    print(f"[3/4] Loading embedding model: '{model_name}' …")
    model = SentenceTransformer(model_name)

    print(f"      Encoding {len(chunks)} chunks in batches of {batch_size} …")
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize 
    ).astype("float32")

    # ── 4. Build and save FAISS index ─────────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after L2-norm
    index.add(embeddings)
    print(f"[4/4] FAISS index built: {index.ntotal} vectors, dim={dim}")

    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "standards.index"))
    with open(index_dir / "metadata.pkl", "wb") as f:
        pickle.dump(standards, f)
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n✅ Index saved → {index_dir}/")
    print(f"   Files: standards.index | metadata.pkl | chunks.pkl")


if __name__ == "__main__":
    build_index()
