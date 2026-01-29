import pytest
from pathlib import Path
from src.bm25 import build_bm25_retriever
from src.retrieve import build_retriever

@pytest.mark.slow
def test_load_chunks_and_query_returns_results():
    chunks_path = Path("data_processed") / "chunks.jsonl"
    assert chunks_path.exists()

    bm25 = build_bm25_retriever(str(chunks_path))
    tfidf = build_retriever(str(chunks_path))

    q = "AC voltage frequency environmental conditions"
    assert len(bm25.search(q, top_k=5)) > 0
    assert len(tfidf.search(q, top_k=5)) > 0