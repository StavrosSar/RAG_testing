import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from .bm25 import build_bm25_retriever
from .retrieve import build_retriever


@dataclass
class EvalItem:
    qid: str
    question: str
    gold_chunk_ids: List[str]


def load_eval_jsonl(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("qid", f"line_{line_num}"))
            q = str(obj["question"])
            gold = obj.get("gold_chunk_ids", [])
            if not isinstance(gold, list):
                raise ValueError(f"gold_chunk_ids must be a list on line {line_num}")
            items.append(EvalItem(qid=qid, question=q, gold_chunk_ids=[str(x) for x in gold]))
    if not items:
        raise RuntimeError("No eval items loaded.")
    return items


def precision_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    gold_set = set(gold)
    hits = sum(1 for cid in topk if cid in gold_set)
    return hits / float(k)


def hit_at_k(retrieved: List[str], gold: List[str], k: int) -> int:
    topk = retrieved[:k]
    gold_set = set(gold)
    return 1 if any(cid in gold_set for cid in topk) else 0


def classify_failure(retrieved: List[str], gold: List[str]) -> Tuple[str, List[str]]:
    """
    Simple failure taxonomy based on retrieval only.
    """
    if not gold:
        return ("NO_GOLD_LABELS", ["Add gold_chunk_ids for this question."])

    if not retrieved:
        return ("NO_RETRIEVAL_RESULTS", ["Check indexing; query may be empty or tokenizer too strict."])

    gold_set = set(gold)
    if not any(cid in gold_set for cid in retrieved[:5]):
        return (
            "LOW_CONTEXT_RELEVANCE",
            [
                "Try smaller chunks or add overlap.",
                "Consider BM25 (often better for manuals) or hybrid BM25+TFIDF.",
                "Improve PDF text cleaning (remove headers/garbage) to reduce noise.",
                "Add simple query expansion (synonyms) for common terms.",
            ],
        )

    # If we hit in top-5 but precision low, likely too many distractors
    p5 = precision_at_k(retrieved, gold, 5)
    if p5 < 0.2:
        return (
            "CONTEXT_TOO_NOISY",
            [
                "Remove boilerplate (page numbers, headers/footers) during ingestion.",
                "Use section-aware chunking (split by headings).",
                "Add reranking (even a simple heuristic) or increase k then filter.",
            ],
        )

    return ("OK", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["tfidf", "bm25"], default="bm25")
    args = parser.parse_args()

    root = Path(".")
    chunks_path = root / "data_processed" / "chunks.jsonl"
    eval_path = root / "eval" / "eval_questions.jsonl"
    report_csv = root / "eval" / f"report_{args.retriever}.csv"
    failures_jsonl = root / "eval" / f"failures_{args.retriever}.jsonl"

    if args.retriever == "bm25":
        retriever = build_bm25_retriever(str(chunks_path))
    else:
        retriever = build_retriever(str(chunks_path))
    
    items = load_eval_jsonl(eval_path)

    report_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []

    for item in items:
        hits = retriever.search(item.question, top_k=10)
        retrieved_ids = [h["chunk_id"] for h in hits]

        p3 = precision_at_k(retrieved_ids, item.gold_chunk_ids, 3)
        p5 = precision_at_k(retrieved_ids, item.gold_chunk_ids, 5)
        h5 = hit_at_k(retrieved_ids, item.gold_chunk_ids, 5)

        report_rows.append(
            {
                "qid": item.qid,
                "question": item.question,
                "precision@3": f"{p3:.4f}",
                "precision@5": f"{p5:.4f}",
                "hit@5": str(h5),
                "gold_chunk_ids": "|".join(item.gold_chunk_ids),
                "top5_chunk_ids": "|".join(retrieved_ids[:5]),
            }
        )

        failure_type, suggested_fixes = classify_failure(retrieved_ids, item.gold_chunk_ids)
        if failure_type != "OK":
            failure_rows.append(
                {
                    "qid": item.qid,
                    "failure_type": failure_type,
                    "question": item.question,
                    "gold_chunk_ids": item.gold_chunk_ids,
                    "top5_chunk_ids": retrieved_ids[:5],
                    "suggested_fixes": suggested_fixes,
                }
            )

    # write report.csv
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    # write failures.jsonl
    with open(failures_jsonl, "w", encoding="utf-8") as f:
        for row in failure_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # summary
    avg_p3 = sum(float(r["precision@3"]) for r in report_rows) / len(report_rows)
    avg_p5 = sum(float(r["precision@5"]) for r in report_rows) / len(report_rows)
    hit5_rate = sum(int(r["hit@5"]) for r in report_rows) / len(report_rows)

    print("✅ Wrote:", report_csv)
    print("✅ Wrote:", failures_jsonl)
    print(f"Avg precision@3 = {avg_p3:.4f}")
    print(f"Avg precision@5 = {avg_p5:.4f}")
    print(f"Hit@5 rate      = {hit5_rate:.4f}")


if __name__ == "__main__":
    main()
