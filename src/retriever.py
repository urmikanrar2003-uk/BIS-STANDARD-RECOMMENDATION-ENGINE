"""
retriever.py
────────────
Hybrid retriever: dense FAISS cosine search + sparse BM25, fused via
Reciprocal Rank Fusion (RRF).

This two-signal approach significantly boosts Hit Rate @3 and MRR @5
compared to either method alone:
  • FAISS  → catches semantic matches ("portland cement" ↔ "OPC")
  • BM25   → catches exact keyword hits ("IS 269", "masonry")
  • RRF    → merges both ranked lists without needing score calibration
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import FAISS_INDEX_DIR, EMBEDDING_MODEL, TOP_K_RETRIEVE


class HybridRetriever:
    """
    Dense (FAISS) + Sparse (BM25) retriever with Reciprocal Rank Fusion.

    Usage:
        r = HybridRetriever()
        results = r.retrieve("portland cement 33 grade")
        # → list of dicts: {standard_id, title, description, keywords, retrieval_score}
    """

    def __init__(
        self,
        index_dir:  Path = FAISS_INDEX_DIR,
        model_name: str  = EMBEDDING_MODEL,
        top_k:      int  = TOP_K_RETRIEVE,
    ):
        if not (index_dir / "standards.index").exists():
            raise FileNotFoundError(
                f"FAISS index not found in '{index_dir}'. "
                "Run 'python src/embedder.py' first."
            )

        self.top_k = top_k

        # ── Load FAISS index ───────────────────────────────────────────────────
        self.index = faiss.read_index(str(index_dir / "standards.index"))

        # ── Load metadata (standard dicts) and text chunks ────────────────────
        with open(index_dir / "metadata.pkl", "rb") as f:
            self.standards: list[dict] = pickle.load(f)
        with open(index_dir / "chunks.pkl", "rb") as f:
            chunks: list[str] = pickle.load(f)

        # ── Embedding model ────────────────────────────────────────────────────
        self.model = SentenceTransformer(model_name)

        # ── BM25 on tokenized chunks ───────────────────────────────────────────
        tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

        print(f"[Retriever] Ready — {len(self.standards)} standards indexed.")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _dense_search(self, query: str) -> list[tuple[int, float]]:
        """Encode query, search FAISS, return (doc_idx, score) pairs."""
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(q_emb, self.top_k)
        return [
            (int(idx), float(scores[0][i]))
            for i, idx in enumerate(indices[0])
            if idx >= 0
        ]

    def _sparse_search(self, query: str) -> list[tuple[int, float]]:
        """BM25 over tokenized chunks, return (doc_idx, score) pairs."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][: self.top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    @staticmethod
    def _rrf_fusion(
        *ranked_lists: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion — score = Σ 1/(k + rank_i).
        Works across arbitrarily many ranked lists without score calibration.
        """
        rrf: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked, start=1):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Return ranked list of BIS standard dicts for the given query.

        Args:
            query:  Natural-language product description.
            top_k:  Override the default number of results.

        Returns:
            List of dicts, each with keys:
                standard_id, title, description, keywords, retrieval_score
        """
        k = top_k or self.top_k

        dense_results  = self._dense_search(query)
        sparse_results = self._sparse_search(query)
        fused          = self._rrf_fusion(dense_results, sparse_results)

        results = []
        for doc_idx, rrf_score in fused[:k]:
            entry = dict(self.standards[doc_idx])   # shallow copy
            entry["retrieval_score"] = round(rrf_score, 6)
            results.append(entry)

        return results


# ── Singleton (loaded once per process) ───────────────────────────────────────
_retriever_instance: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return the module-level singleton HybridRetriever (lazy init)."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "33 Grade Ordinary Portland Cement"
    print(f"\nQuery: {query}\n")
    r = HybridRetriever()
    for i, std in enumerate(r.retrieve(query), 1):
        print(f"  [{i}] {std['standard_id']:30s}  score={std['retrieval_score']:.5f}")
        print(f"       {std['title'][:80]}")
