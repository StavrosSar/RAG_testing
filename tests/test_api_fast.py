from fastapi.testclient import TestClient
import src.api as api

class DummyRetriever:
    def search(self, q: str, top_k: int = 5):
        return [
            {"doc_id": "Doc1", "chunk_id": "c1", "score": 1.0, "text": "AC voltage range is 100-240V.", "source": "s1"},
            {"doc_id": "Doc2", "chunk_id": "c2", "score": 0.9, "text": "Environmental conditions: 10-35C.", "source": "s2"},
        ][:top_k]

def test_query_endpoint_returns_200(monkeypatch):
    # force API to use dummy retriever (no file, no DB)
    monkeypatch.setattr(api, "get_retriever", lambda name: DummyRetriever())

    client = TestClient(api.app)
    payload = {"question": "What AC voltage range is supported?", "top_k": 2, "retriever": "bm25", "mode": "extractive"}
    resp = client.post("/query", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "top_k" in body
    assert len(body["top_k"]) > 0