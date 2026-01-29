import json
from src.bm25 import build_bm25_retriever
from src.retrieve import build_retriever

def _write_chunks(tmp_path):
    p = tmp_path / "chunks.jsonl"
    rows = [
        {"doc_id": "Doc1", "chunk_id": "c1", "source": "s1", "text": "This chunk explains AC voltage range and power supply input voltage."},
        {"doc_id": "Doc2", "chunk_id": "c2", "source": "s2", "text": "This chunk describes environmental conditions for operating a server."},
        {"doc_id": "Doc3", "chunk_id": "c3", "source": "s3", "text": "This chunk is about floppy disk eject procedure and yellow activity light."},
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)

def test_bm25_returns_results_on_tiny_corpus(tmp_path):
    chunks_path = _write_chunks(tmp_path)
    r = build_bm25_retriever(chunks_path)
    hits = r.search("AC voltage input", top_k=3)
    assert len(hits) > 0
    assert hits[0]["chunk_id"]

def test_tfidf_returns_results_on_tiny_corpus(tmp_path):
    chunks_path = _write_chunks(tmp_path)
    r = build_retriever(chunks_path)
    hits = r.search("environmental conditions operating", top_k=3)
    assert len(hits) > 0
    assert hits[0]["chunk_id"]