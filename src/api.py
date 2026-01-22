from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from .retrieve import build_retriever
from .bm25 import build_bm25_retriever
from .answer import answer_with_citations
import re

app = FastAPI(title="RAG POC Manuals")

def is_junk_chunk(h) -> bool:
    t = (h.get("text") or "").lower()
    head = t[:600]

    if "contents" in head or "index" in head:
        return True

    if "summary of" in head and "guidelines" in head:
        return True

    if t.count(",") > 25 and len(t) > 800:
        return True

    if len(re.findall(r"\b[A-Z]?\d+\-\d+\b", head)) >= 8:
        return True

    return False

class QueryIn(BaseModel):
    question: str
    top_k: int = 5
    retriever: str = "tfidf"   # or "bm25"

@lru_cache(maxsize=2)
def get_retriever(name: str):
    chunks_path = str(Path("data_processed") / "chunks.jsonl")
    return build_bm25_retriever(chunks_path) if name == "bm25" else build_retriever(chunks_path)

@app.post("/query")
def query(q: QueryIn):
    r = get_retriever(q.retriever)
    hits = r.search(q.question, top_k=q.top_k)

    filtered_hits = [h for h in hits if not is_junk_chunk(h)]

    out = answer_with_citations(q.question, filtered_hits, max_sentences=3)

    return {
        "question": q.question,
        "answer": out["answer"],
        "citations": out["citations"],
        "top_k": hits,              # raw retrieval (for debugging / transparency)
        "top_k_filtered": filtered_hits
    }
