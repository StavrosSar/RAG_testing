import argparse
import re
from pathlib import Path
from .answer import answer_with_citations
from .retrieve import build_retriever  # TF-IDF
from .bm25 import build_bm25_retriever  # BM25

def print_result(q: str, hits, out):
    print("\nQ:", q)
    print("\nAnswer:")
    print(out["answer"])
    print("\nCitations:", out["citations"])

    print("\nTop results:")
    for i, h in enumerate(hits, start=1):
        preview = h["text"].replace("\n", " ")[:240]
        print(f"{i}. score={h['score']:.4f} {h['chunk_id']} ({h['doc_id']})")
        print("   ", preview, "...\n")

# Argument Parsing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default=str(Path("data_processed") / "chunks.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", type=str, default=None, help="Run a single query and exit")
    parser.add_argument("--retriever", choices=["tfidf", "bm25"], default="bm25")
    args = parser.parse_args()

# Choosing the retriever
    if args.retriever == "bm25":
        r = build_bm25_retriever(args.chunks)
    else:
        r = build_retriever(args.chunks)

    if args.query:
        q = args.query.strip() #Stripping the query
        hits = r.search(q, top_k=args.k)
        print("DEBUG: requested k =", args.k, "returned hits =", len(hits))

        # filter junk hits BEFORE answering
        filtered_hits = [h for h in hits if not is_junk_chunk(h)]
        print("DEBUG: filtered hits =", len(filtered_hits))

        out = answer_with_citations(q, filtered_hits, max_sentences=3)
        print_result(q, hits, out)   
        return

    print(f"RAG baseline ready ({args.retriever}). Type 'exit' to quit.\n")
    while True:
        try:
            q = input("Q> ").strip()
        except EOFError:
            print("\n(EOF) No interactive input available. Tip: use --query \"...\"")
            break

        if q.lower() in {"exit", "quit"}:
            break

        hits = r.search(q, top_k=args.k)
        print("DEBUG: requested k =", args.k, "returned hits =", len(hits))

        # filter junk hits BEFORE answering
        filtered_hits = [h for h in hits if not is_junk_chunk(h)]
        print("DEBUG: filtered hits =", len(filtered_hits))

        out = answer_with_citations(q, filtered_hits, max_sentences=3)
        print_result(q, hits, out)   
 
def is_junk_chunk(h) -> bool:
    t = (h.get("text") or "").lower()
    head = t[:600]  # κοιτάμε μόνο την αρχή

    # obvious index/contents markers
    if "contents" in head or "index" in head:
        return True

    # "Summary of ..." / appendix index pages
    if "summary of" in head and "guidelines" in head:
        return True

    # looks like index: tons of commas in a long block
    if t.count(",") > 25 and len(t) > 800:
        return True

    # lots of page/section artifacts: many "A-5", "2-21", etc.
    # (simple proxy: many hyphen-digit patterns)
    if len(re.findall(r"\b[A-Z]?\d+\-\d+\b", head)) >= 8:
        return True

    return False

if __name__ == "__main__":
    main()