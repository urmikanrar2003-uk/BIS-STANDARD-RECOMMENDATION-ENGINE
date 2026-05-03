"""
Usage:
    from pipeline import recommend
    result = recommend("We manufacture 33 Grade Ordinary Portland Cement.")
    print(result["retrieved_standards"])   # ["IS 269: 1989", ...]
    print(result["latency_seconds"])       # 1.24
"""

import time

from retriever    import get_retriever
from config       import TOP_K_RETRIEVE, TOP_K_FINAL


def recommend(query: str, top_k: int = TOP_K_FINAL) -> dict:
    """
    Full RAG pipeline: query → BIS standard recommendations.
    Optimized for competition latency (target < 0.02s).
    """
    t_start = time.perf_counter()

    # ── Step 1: Hybrid retrieval (FAISS + BM25) ───────────────────────────────
    # This is extremely fast (<20ms) and highly accurate for this dataset.
    retriever  = get_retriever()
    candidates = retriever.retrieve(query, top_k=top_k)

    # ── Step 2: Final Rank ────────────────────────────────────────────────────
    # We use the retrieval_score (RRF-fused) for final ranking.
    # LLM reranking is removed to achieve sub-0.02s latency.
    ranked_ids = [c["standard_id"] for c in candidates]

    t_end   = time.perf_counter()
    latency = round(t_end - t_start, 4)

    return {
        "retrieved_standards": ranked_ids[:top_k],
        "latency_seconds":     latency,
        "candidates":          candidates,
    }
