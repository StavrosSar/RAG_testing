import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ω0-9]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source: Optional[str] = None


class BM25Retriever:
    """
    Pure-Python BM25 (Okapi) retriever over chunks.jsonl.
    Good baseline for manuals/procedures.
    """

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        # per-doc term frequencies + doc lengths
        self.doc_tf: List[Counter] = []
        self.doc_len: List[int] = []

        # document frequency
        self.df: Counter = Counter()

        # inverted index: term -> list of (doc_idx, tf)
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for i, ch in enumerate(chunks):
            terms = tokenize(ch.text)
            tf = Counter(terms)
            self.doc_tf.append(tf)
            dl = sum(tf.values())
            self.doc_len.append(dl)

            for term in tf.keys():
                self.df[term] += 1

        self.N = len(chunks)
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

        # build postings
        for i, tf in enumerate(self.doc_tf):
            for term, f in tf.items():
                self.postings[term].append((i, f))

    @staticmethod
    def load_chunks_jsonl(path: Path) -> List[Chunk]:
        chunks: List[Chunk] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if len(text) < 50:
                    continue
                chunks.append(
                    Chunk(
                        doc_id=str(obj.get("doc_id", "")),
                        chunk_id=str(obj.get("chunk_id", "")),
                        text=text,
                        source=obj.get("source"),
                    )
                )
        if not chunks:
            raise RuntimeError("No valid chunks loaded. Check chunks.jsonl.")
        return chunks

    def _idf(self, term: str) -> float:
        """
        BM25 idf with +1 smoothing.
        """
        df = self.df.get(term, 0)
        # classic BM25 idf:
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []

        # accumulate scores only for docs that contain at least one query term
        scores = defaultdict(float)

        # BM25 scoring
        for term in q_terms:
            idf = self._idf(term)
            for doc_idx, tf in self.postings.get(term, []):
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                score = idf * (tf * (self.k1 + 1)) / (denom if denom != 0 else 1.0)
                scores[doc_idx] += score

        if not scores:
            return []

        # top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max(top_k, 1)]
        results = []
        for doc_idx, score in ranked:
            ch = self.chunks[doc_idx]
            results.append(
                {
                    "doc_id": ch.doc_id,
                    "chunk_id": ch.chunk_id,
                    "score": float(score),
                    "text": ch.text,
                    "source": ch.source,
                }
            )
        return results


def build_bm25_retriever(chunks_path: str, k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {path}")
    chunks = BM25Retriever.load_chunks_jsonl(path)
    return BM25Retriever(chunks, k1=k1, b=b)
