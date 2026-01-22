import sys
from pathlib import Path
from typing import List, Dict, Any
import re

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.retrieve import build_retriever           # TF-IDF
from src.bm25 import build_bm25_retriever          # BM25
from src.answer import answer_with_citations


# --- Pick demo queries that behave well across manuals ---
QUERIES = [
"How do you eject a floppy disk? (yellow activity light)",
"List the items that should be in the box when you receive the Archimedes system.",
"Guidelines for Returning Condition Values on OpenVMS",
"What is the operating voltage range for the AlphaServer DS10?",
"What is the nominal voltage (VAC) for the AlphaServer DS20 power supply?",
]


def is_junk_chunk(text: str) -> bool:
    """
    Heuristic filter for TOC/Index/garbage chunks that often pollute retrieval.
    Keep it simple + safe (high precision).
    """
    t = (text or "").strip()
    if not t:
        return True

    low = t.lower()

    # Common TOC / Index markers
    if low.startswith("index"):
        return True
    if "table of contents" in low or low.startswith("contents"):
        return True
    if "....." in t:  # typical TOC dot leaders
        return True

    # Many short lines = likely index/TOC
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 18:
        avg_len = sum(len(ln) for ln in lines) / max(1, len(lines))
        # Lots of short lines -> looks like index entries
        if avg_len < 35:
            return True

    return False


def filter_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop junk chunks but preserve ordering/scores of the remaining hits."""
    out = []
    for h in hits:
        if not is_junk_chunk(h.get("text", "")):
            out.append(h)
    return out


def rerank_hits_by_keywords(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple heuristic re-ranker:
    If the query is about electrical specs, boost chunks containing those terms.
    """
    q = query.lower()
    boost_terms = []

    if any(w in q for w in ["voltage", "vac", "ac", "nominal", "operating voltage", "operating range"]):
        boost_terms = ["voltage", "vac", "ac", "nominal", "operating", "range", "hz"]

    def bonus(text: str) -> float:
        t = (text or "").lower()
        b = 0.0
        for term in boost_terms:
            if term in t:
                b += 1.0
        return b

    # copy + add small bonus to the original score
    new_hits = []
    for h in hits:
        hh = dict(h)
        hh["score"] = float(hh["score"]) + 0.50 * bonus(h.get("text", ""))
        new_hits.append(hh)

    new_hits.sort(key=lambda x: x["score"], reverse=True)
    return new_hits


def print_hits(hits: List[Dict[str, Any]], top_k: int, max_preview: int = 220):
    for i, h in enumerate(hits[:top_k], start=1):
        preview = (h.get("text", "") or "").replace("\n", " ")
        preview = (preview[:max_preview] + "...") if len(preview) > max_preview else preview
        print(f"{i}. score={h['score']:.4f}  {h['chunk_id']}  ({h['doc_id']})")
        print(f"   {preview}\n")


def run_one_retriever(name: str, retriever, queries: List[str], k: int):
    print("\n" + "=" * 90)
    print(f"RETRIEVER: {name.upper()}   (k={k})")
    print("=" * 90)

    for q in queries:
        hits = retriever.search(q, top_k=k)
        hits_f = filter_hits(hits)
        hits_f = rerank_hits_by_keywords(q, hits_f) if hits_f else hits
        hits_for_answer = hits_f if hits_f else hits

        out = answer_with_citations(q, hits_for_answer, max_sentences=3)

        print("\n" + "-" * 90)
        print("QUERY:", q)
        print(f"DEBUG: requested k = {k} returned hits = {len(hits)}")
        print(f"DEBUG: filtered hits = {len(hits_f)}")

        print("\nANSWER:")
        print(out["answer"])
        print("\nCITATIONS:", out["citations"])

        print("\nTOP RESULTS (raw):")
        print_hits(hits, top_k=k)

        if hits_f:
            print("\nTOP RESULTS (filtered):")
            print_hits(hits_f, top_k=min(k, len(hits_f)))


def main():
    chunks_path = Path("data_processed") / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    k = 5

    # Build retrievers once
    bm25 = build_bm25_retriever(str(chunks_path))
    tfidf = build_retriever(str(chunks_path))

    # Run both
    run_one_retriever("bm25", bm25, QUERIES, k=k)
    run_one_retriever("tfidf", tfidf, QUERIES, k=k)


if __name__ == "__main__":
    main()
