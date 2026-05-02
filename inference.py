"""
inference.py  ← MANDATORY JUDGE ENTRY-POINT
────────────
Runs the BIS Standard Recommendation Engine on a JSON test set and
writes results in the exact format expected by eval_script.py.

Usage:
    python inference.py
    python inference.py --input  data/public_test_set.json \
                        --output data/results/output.json

Output JSON format (per item):
    {
        "id":                  "PUB-01",
        "query":               "We are a small enterprise …",
        "expected_standards":  ["IS 269: 1989"],
        "retrieved_standards": ["IS 269: 1989", "IS 8112: 1989", …],
        "latency_seconds":     1.24
    }
"""

import sys
import json
import argparse
from pathlib import Path

# ── Ensure src/ is importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import recommend  # noqa: E402 (after sys.path modification)


# ── Banner ─────────────────────────────────────────────────────────────────────
_BANNER = """
╔══════════════════════════════════════════════════════════╗
║        BIS Standard Recommendation Engine               ║
║        AI-powered RAG · Retriever → LLM → Output        ║
╚══════════════════════════════════════════════════════════╝
"""


def run_inference(input_path: str, output_path: str) -> None:
    """
    Load test queries from input_path, run the pipeline on each,
    and write the evaluation-ready JSON to output_path.
    """
    # ── Load test set ──────────────────────────────────────────────────────────
    in_path = Path(input_path)
    if not in_path.exists():
        print(f"❌  Input file not found: {in_path}")
        sys.exit(1)

    with open(in_path, encoding="utf-8") as f:
        queries: list[dict] = json.load(f)

    print(_BANNER)
    print(f"  Input  : {in_path}")
    print(f"  Queries: {len(queries)}")
    print(f"{'─'*60}\n")

    # ── Run pipeline per query ─────────────────────────────────────────────────
    results = []
    total_latency = 0.0

    for i, item in enumerate(queries, 1):
        qid   = item.get("id", f"Q-{i:02d}")
        query = item["query"]

        print(f"[{i:02d}/{len(queries)}]  {qid}")
        print(f"         Query: {query[:80]}{'…' if len(query) > 80 else ''}")

        output = recommend(query)

        total_latency += output["latency_seconds"]

        result = {
            "id":                  qid,
            "query":               query,
            "expected_standards":  item.get("expected_standards", []),
            "retrieved_standards": output["retrieved_standards"],
            "latency_seconds":     output["latency_seconds"],
        }
        results.append(result)

        print(f"         → {output['retrieved_standards']}")
        print(f"         Latency: {output['latency_seconds']}s\n")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_latency = total_latency / len(queries) if queries else 0.0

    print(f"{'─'*60}")
    print(f"✅  Done! Results saved → {out_path}")
    print(f"    Avg latency : {avg_latency:.2f}s  (target: <5s)")
    print(f"\n  Next step — run the evaluator:")
    print(f"  python eval_script.py --results {out_path}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BIS Standard Recommendation Engine — Inference Script"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/public_test_set.json",
        help="Path to input JSON (list of query objects)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/output.json",
        help="Path to write output JSON (eval_script.py compatible)",
    )
    args = parser.parse_args()

    run_inference(args.input, args.output)
