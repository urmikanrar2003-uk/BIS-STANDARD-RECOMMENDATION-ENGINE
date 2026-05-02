"""
llm_reranker.py
───────────────
Uses OpenAI GPT-4o-mini to rerank the top retrieved BIS standards
for a given query and produce a final ranked list.

The LLM acts as a "smart filter":
  • Understands the business context from the natural-language query.
  • Reorders retrieved candidates by true relevance.
  • Strictly grounded — can ONLY output standard IDs from the
    provided candidate list (no hallucinated standards).

Fallback: if the API call fails, retrieval-score ordering is used.
"""

import re
import json

from config import OPENAI_API_KEY, OPENAI_MODEL, TOP_K_FINAL


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(query: str, candidates: list[dict]) -> str:
    """
    Build the reranking prompt.
    The LLM only sees standard IDs, titles, and short descriptions —
    never raw embeddings or retrieval scores.
    """
    candidate_block = ""
    for i, std in enumerate(candidates, 1):
        candidate_block += (
            f"\n[{i}] {std['standard_id']}\n"
            f"     Title      : {std['title'][:120]}\n"
            f"     Description: {std['description'][:200]}\n"
        )

    return f"""You are a senior BIS (Bureau of Indian Standards) compliance expert helping \
Indian Micro and Small Enterprises (MSEs) find the correct standards for their products.

A business has described their product or compliance need as follows:
<query>
{query}
</query>

Below are candidate BIS standards retrieved from the official database:
{candidate_block}

Your task:
1. Read the query carefully and understand the product / use-case.
2. Select the {TOP_K_FINAL} most relevant standards from the list above.
3. Rank them from MOST relevant (index 0) to LEAST relevant.
4. Return ONLY a JSON array of standard IDs in ranked order. Example:
   ["IS 269: 1989", "IS 455: 1989", "IS 8112: 1989"]
5. IMPORTANT: Only use standard IDs that appear in the candidate list above.
   Do NOT invent, modify, or combine standard IDs.
6. Return ONLY the JSON array — no explanation, no markdown, no extra text.

JSON array:"""


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_response(response_text: str, fallback: list[dict]) -> list[str]:
    """Extract ranked standard IDs from LLM response. Falls back to retrieval order."""
    try:
        match = re.search(r"\[.*?\]", response_text, re.DOTALL)
        if match:
            ids = json.loads(match.group(0))
            cleaned = [str(s).strip() for s in ids if isinstance(s, str)]
            if cleaned:
                return cleaned
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return [s["standard_id"] for s in fallback[:TOP_K_FINAL]]


# ── Public API ─────────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict]) -> list[str]:
    """
    Rerank the retrieved candidates using OpenAI GPT-4o-mini.

    Args:
        query:      Original product description query.
        candidates: List of standard dicts from the retriever.

    Returns:
        Ranked list of standard_id strings (most relevant first).
        Falls back to retrieval order on any failure.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = _build_prompt(query, candidates)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        return _parse_response(response.choices[0].message.content, candidates)

    except Exception as exc:
        print(f"[LLM Reranker] ⚠️  Error: {exc}. Using retrieval fallback.")
        return [s["standard_id"] for s in candidates[:TOP_K_FINAL]]
