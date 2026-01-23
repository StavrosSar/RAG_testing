#FastAPI backend
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import re

from fastapi.middleware.cors import CORSMiddleware

from .retrieve import build_retriever, build_retriever_from_records
from .bm25 import build_bm25_retriever, build_bm25_retriever_from_records
from .answer import answer_with_citations
from .chunks_mssql import load_chunks_from_mssql
from .llm_ollama import answer_with_llm


app = FastAPI(title="RAG POC Manuals")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    top_k: int = 10
    retriever: str = "bm25"      # "bm25", "tfidf", "bm25_sql", "tfidf_sql"
    mode: str = "llm"            # "llm" or "extractive"


@lru_cache(maxsize=4)
def get_retriever(name: str):
    chunks_path = str(Path("data_processed") / "chunks.jsonl")

    if name == "bm25":
        return build_bm25_retriever(chunks_path)

    if name == "tfidf":
        return build_retriever(chunks_path)

    # SQL-backed retrievers (SQL is source of truth)
    if name in ("tfidf_sql", "bm25_sql"):
        records = load_chunks_from_mssql()  # cached via lru_cache

        if name == "bm25_sql":
            return build_bm25_retriever_from_records(records)
        return build_retriever_from_records(records)

    # default fallback
    return build_retriever(chunks_path)


@app.post("/query")
def query(q: QueryIn):
    r = get_retriever(q.retriever)
    hits = r.search(q.question, top_k=q.top_k)

    filtered_hits = [h for h in hits if not is_junk_chunk(h)]

    answer_text = ""
    citations = []

    if q.mode.lower() == "llm":
        llm_hits = filtered_hits[:5]  # keep context small
        llm_out = answer_with_llm(q.question, llm_hits)

        # answer_with_llm might return str OR dict; handle both safely
        if isinstance(llm_out, dict):
            answer_text = llm_out.get("answer", "")
            citations = llm_out.get("citations", [h["chunk_id"] for h in llm_hits])
        else:
            answer_text = str(llm_out)
            citations = [h["chunk_id"] for h in llm_hits]

    else:
        out = answer_with_citations(q.question, filtered_hits, max_sentences=3)
        answer_text = out["answer"]
        citations = out.get("citations", [])

    return {
        "question": q.question,
        "answer": answer_text,
        "citations": citations,
        "top_k": hits,
        "top_k_filtered": filtered_hits,
    }

    

@app.get("/debug/env")
def debug_env():
    import os
    return {"has_mssql_conn_str": "MSSQL_CONN_STR" in os.environ}


@app.get("/debug/sqlcount")
def debug_sqlcount():
    import os, pyodbc
    conn = pyodbc.connect(os.environ["MSSQL_CONN_STR"])
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM dbo.rag_chunks")
    n = cur.fetchone()[0]
    cur.close()
    conn.close()
    return {"rows": n}
