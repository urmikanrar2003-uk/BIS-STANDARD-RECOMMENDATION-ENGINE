"""
Usage:
    from pipeline import recommend
    result = recommend("We manufacture 33 Grade Ordinary Portland Cement.")
    print(result["retrieved_standards"])   # ["IS 269: 1989", ...]
    print(result["latency_seconds"])       # 1.24
"""

import time

from retriever    import get_retriever
from llm_reranker import rerank
from config       import TOP_K_RETRIEVE, TOP_K_FINAL


def recommend(query: str, top_k: int = TOP_K_FINAL) -> dict:
    """
    Full RAG pipeline: query → BIS standard recommendations.

    Args:
        query:  Natural-language product description from the MSE.
        top_k:  Number of final recommendations to return (default 5).

    Returns:
        {
            "retrieved_standards": ["IS 269: 1989", ...],   # ranked, top-k
            "latency_seconds":     1.24,                    # wall-clock time
            "candidates":          [...],                   # raw retriever output
        }
    """
    t_start = time.perf_counter()

    # ── Step 1: Hybrid retrieval ───────────────────────────────────────────────
    retriever  = get_retriever()
    candidates = retriever.retrieve(query, top_k=TOP_K_RETRIEVE)

    # ── Step 2: LLM reranking ─────────────────────────────────────────────────
    ranked_ids = rerank(query, candidates)

    t_end   = time.perf_counter()
    latency = round(t_end - t_start, 3)

    return {
        "retrieved_standards": ranked_ids[:top_k],
        "latency_seconds":     latency,
        "candidates":          candidates,     # for debugging / analysis
    }
