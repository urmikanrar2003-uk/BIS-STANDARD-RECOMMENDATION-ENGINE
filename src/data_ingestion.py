"""
data_ingestion.py
─────────────────
Parses the BIS standards catalog PDF (data/dataset.pdf) and extracts
structured entries into data/bis_standards.json.
"""

import re
import json
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

from config import PDF_PATH, STANDARDS_JSON

# ── Patterns ───────────────────────────────────────────────────────────────────
# Matches IS numbers in the forms:
#   IS 269 : 1989
#   IS 2185 (Part 1) : 1979
#   IS 1489 (Part 2) : 1991
#   IS 9142 : 1979
IS_PATTERN = re.compile(
    r"IS\s+\d+\s*(?:\(\s*(?:Part|Pt\.?)\s*\d+\s*\))?\s*(?:\([^)]{1,30}\))?\s*:\s*\d{4}",
    re.IGNORECASE,
)

# Words to exclude from keyword extraction
_STOPWORDS = {
    "that", "with", "from", "this", "shall", "used", "which", "part",
    "test", "spec", "india", "bureau", "standard", "standards", "indian",
    "requirements", "specification", "general", "method", "methods",
    "code", "practice", "amendment", "reaffirmed", "edition", "section",
    "clause", "table", "figure", "note", "type", "class", "grade",
    "material", "product", "manufacture", "manufacturing",
}


def _normalize_std_id(raw: str) -> str:
    """Normalize IS id: collapse spaces, fix colon spacing."""
    normalized = re.sub(r"\s*:\s*", ": ", raw)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_full_text(pdf_path: Path) -> list[str]:
    """Return list of page texts from the PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in tqdm(doc, desc="  Reading PDF pages", unit="pg"):
        pages.append(page.get_text("text"))
    doc.close()
    return pages


def parse_standards(pages: list[str]) -> list[dict]:
    """
    Extract BIS standards from raw page texts.

    Strategy:
      1. Join all pages into one string.
      2. Find every IS-number occurrence via regex.
      3. Grab the text between consecutive IS numbers as the description.
      4. Extract title (first non-empty line after the IS number).
    """
    full_text = "\n".join(pages)

    # All IS matches in document order
    matches = list(IS_PATTERN.finditer(full_text))
    print(f"  Found {len(matches)} raw IS-number occurrences in PDF.")

    standards = []
    seen_ids: set[str] = set()

    for i, match in enumerate(tqdm(matches, desc="  Parsing standards", unit="std")):
        raw_id       = match.group(0).strip()
        standard_id  = _normalize_std_id(raw_id)

        if standard_id in seen_ids:
            continue
        seen_ids.add(standard_id)

        # Text window: from end of this match to start of next (max 600 chars)
        start = match.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else start + 600
        window = full_text[start:end].strip()

        # Title = first meaningful line
        lines  = [ln.strip() for ln in window.splitlines() if ln.strip()]
        title  = lines[0] if lines else standard_id
        # Cap title at 150 chars to avoid grabbing full paragraphs
        if len(title) > 150:
            title = title[:150].rsplit(" ", 1)[0] + "…"

        # Description = collapsed block, max 400 chars
        description = " ".join(window.split())[:400]

        # Keywords from id + title + description
        combined = (standard_id + " " + title + " " + description).lower()
        raw_words = re.findall(r"[a-z]{4,}", combined)
        keywords  = list({w for w in raw_words if w not in _STOPWORDS})[:20]

        standards.append({
            "standard_id": standard_id,
            "title":       title,
            "description": description,
            "keywords":    keywords,
        })

    return standards


def ingest(
    pdf_path:    Path = PDF_PATH,
    output_path: Path = STANDARDS_JSON,
) -> list[dict]:
    """Full ingestion pipeline: PDF → structured JSON."""
    print(f"\n[Step 1/3] Extracting text from '{pdf_path.name}' …")
    pages = extract_full_text(pdf_path)
    print(f"  Extracted {len(pages)} pages.")

    print(f"\n[Step 2/3] Parsing BIS standards …")
    standards = parse_standards(pages)
    print(f"  Parsed {len(standards)} unique standards.")

    print(f"\n[Step 3/3] Saving to '{output_path}' …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standards, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {len(standards)} standards → {output_path}")

    return standards


if __name__ == "__main__":
    standards = ingest()
    print(f"\nSample entry:\n{json.dumps(standards[0], indent=2)}")
    print(f"\nTotal unique standards extracted: {len(standards)}")
