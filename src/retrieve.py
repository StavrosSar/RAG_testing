#TF-IDF retriever 
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import PorterStemmer
from .query_utils import normalize_and_expand_query

stemmer = PorterStemmer()


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source: Optional[str] = None


class TfidfRetriever:
    def __init__(self, chunks: List[Chunk], ngram_range=(1, 2), max_features: int = 200_000):
        self.chunks = chunks
        # Use tokenizer (not analyzer) so sklearn can apply ngram_range.
        self.vectorizer = TfidfVectorizer(
            tokenizer=stem_analyzer,
            preprocessor=lambda x: x,
            token_pattern=None,
            ngram_range=ngram_range,
            max_features=max_features,
        )

        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    @staticmethod
    def load_chunks_jsonl(path: Path) -> List[Chunk]:
        out: List[Chunk] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if len(text) < 50:
                    continue
                out.append(
                    Chunk(
                        doc_id=str(obj.get("doc_id", "")),
                        chunk_id=str(obj.get("chunk_id", "")),
                        text=text,
                        source=obj.get("source"),
                    )
                )
        if not out:
            raise RuntimeError("No valid chunks loaded. Check chunks.jsonl.")
        return out


    @staticmethod
    def load_chunks_from_records(records) -> List[Chunk]:
        out: List[Chunk] = []
        for obj in records:
            text = (obj.get("text") or "").strip()
            if len(text) < 50:
                continue
            out.append(
                Chunk(
                    doc_id=str(obj.get("doc_id", "")),
                    chunk_id=str(obj.get("chunk_id", "")),
                    text=text,
                    source=obj.get("source"),
                )
            )
        if not out:
            raise RuntimeError("No valid chunks loaded from SQL.")
        return out

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = normalize_and_expand_query(query)
        if not q:
            return []
        qv = self.vectorizer.transform([q])
        scores = linear_kernel(qv, self.matrix).flatten()
        idxs = scores.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            c = self.chunks[int(i)]
            results.append(
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "score": float(scores[int(i)]),
                    "text": c.text,
                    "source": c.source,
                }
            )
        return results

def build_retriever_from_records(records) -> TfidfRetriever:
    chunks = TfidfRetriever.load_chunks_from_records(records)
    return TfidfRetriever(chunks)

def build_retriever(chunks_path: str) -> TfidfRetriever:
    path = Path(chunks_path)
    chunks = TfidfRetriever.load_chunks_jsonl(path)
    return TfidfRetriever(chunks)


def stem_analyzer(text: str):
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [stemmer.stem(t) for t in tokens]
