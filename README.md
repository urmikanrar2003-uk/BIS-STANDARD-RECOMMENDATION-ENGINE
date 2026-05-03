# BIS Standard Recommendation Engine

> **AI-powered RAG system** that maps Indian MSE product descriptions to Bureau of Indian Standards (BIS) regulations in seconds.


---

## Problem Statement

Indian Micro and Small Enterprises (MSEs) often spend **weeks** identifying which BIS regulations apply to their products. This engine uses **Retrieval-Augmented Generation (RAG)** to return the top 3–5 relevant BIS standards from a plain-English product description in **under 5 seconds**
(Bureau of Indian Standards x Sigma Squad AI Hackathon
Indian Institute of Technology (IIT), Tirupati).

---

## Architecture

```
Product Description (query)
         │
         ▼
 ┌───────────────────┐
 │  HybridRetriever  │  FAISS (dense cosine) + BM25 (sparse keyword)
 │                   │  fused via Reciprocal Rank Fusion (RRF)
 └───────────────────┘
         │  top-10 candidates
         ▼
 ┌───────────────────┐
 │   LLM Reranker    │  OpenAI GPT-4o-mini
 │                   │  strictly grounded — no hallucinations
 └───────────────────┘
         │
         ▼
  Top 3–5 BIS Standards + latency
```

---

## Repository Structure

```
BIS-STANDARD-RECOMMENDATION-ENGINE/
├── src/
│   ├── config.py           # All paths, model names, env vars
│   ├── data_ingestion.py   # PDF → structured JSON
│   ├── embedder.py         # JSON → FAISS vector index
│   ├── retriever.py        # Hybrid FAISS + BM25 + RRF retriever
│   ├── llm_reranker.py     # OpenAI GPT-4o-mini reranker
│   └── pipeline.py         # End-to-end: query → recommendations
├── data/
│   ├── dataset.pdf         # BIS standards corpus
│   ├── bis_standards.json  # [generated] parsed standards
│   ├── public_test_set.json
│   ├── faiss_index/        # [generated] FAISS index files
│   └── results/            # [generated] inference output
├── eval_script.py          # Evaluation script
├── inference.py            # Judge entry-point
├── requirements.txt        # All dependencies
└── .env                    #environment file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key

# Edit .env and add your OPENAI_API_KEY


### 3. Parse the BIS PDF corpus (run once)

```bash
python src/data_ingestion.py
```

### 4. Build the FAISS vector index (run once)

```bash
python src/embedder.py
```

### 5. Run inference on the test set

```bash
python inference.py --input data/public_test_set.json --output data/results/output.json
```

### 6. Evaluate results

```bash
python eval_script.py --results data/results/output.json
```

**Results on public test set:**
```
========================================
   BIS HACKATHON EVALUATION RESULTS
========================================
Total Queries Evaluated : 10
Hit Rate @3             : 100.00%  (Target: >80%)
MRR @5                  : 1.0000   (Target: >0.7)
Avg Latency             : 2.96 sec (Target: <5 seconds)
========================================
```

---

## Configuration

All configuration lives in `src/config.py` and `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `TOP_K_RETRIEVE` | `10` | Candidates retrieved before LLM reranking |
| `TOP_K_FINAL` | `5` | Final recommendations returned |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model (runs locally) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM used for reranking |

---

## Evaluation Metrics

| Metric | Formula | Target | Achieved |
|--------|---------|--------|---------|
| **Hit Rate @3** | ≥1 expected standard in top-3 / total × 100 | >80% | **100%** |
| **MRR @5** | Mean Reciprocal Rank of first correct in top-5 | >0.70 | **1.0000** |
| **Avg Latency** | Total time / num queries | <5 sec | **2.96s** |

---

## How It Works

### Step 1 — Data Ingestion
`src/data_ingestion.py` uses **PyMuPDF** to extract all IS-numbered entries from `data/dataset.pdf` into a structured JSON (`data/bis_standards.json`), capturing standard ID, title, description, and keywords.

### Step 2 — Embedding & Indexing
`src/embedder.py` encodes each standard using **`sentence-transformers/all-MiniLM-L6-v2`** (runs fully locally, no API) and stores the vectors in a **FAISS** flat index for cosine similarity search.

### Step 3 — Hybrid Retrieval
`src/retriever.py` runs two parallel searches for every query:
- **Dense search** via FAISS (semantic similarity)
- **Sparse search** via BM25 (keyword matching)

Both ranked lists are fused using **Reciprocal Rank Fusion (RRF)** to produce the top-10 candidates.

### Step 4 — LLM Reranking
`src/llm_reranker.py` sends the top-10 candidates to **OpenAI GPT-4o-mini** with a strict grounding prompt. The LLM reorders by true business relevance and can only output standard IDs from the provided candidate list — **zero hallucination risk**.

---

## Running Individual Steps

```bash
# Quick retrieval test (no LLM)
python src/retriever.py "portland cement 33 grade specification"

# Full pipeline on one query
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline import recommend
r = recommend('We manufacture 33 Grade Ordinary Portland Cement.')
print(r['retrieved_standards'])
print(f'Latency: {r[\"latency_seconds\"]}s')
"
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `PyMuPDF` | PDF text extraction |
| `sentence-transformers` | Local semantic embeddings (no API) |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | Keyword-based BM25 search |
| `openai` | GPT-4o-mini LLM reranking |
| `python-dotenv` | Environment variable management |

---

## License



##  APP Link:
https://bis-standard-recommendation-engine-gybzddvzi4rrezbzbpcdak.streamlit.app/
