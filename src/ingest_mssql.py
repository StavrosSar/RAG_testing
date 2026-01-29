import os
import json
import argparse
from pathlib import Path

import pyodbc


# Idempotent ingest (Option C): MERGE upsert by chunk_id.
# Assumes a unique constraint/index exists on dbo.rag_chunks(chunk_id)
# (e.g. UX_rag_chunks_chunk_id).
MERGE_UPSERT_SQL = """
MERGE dbo.rag_chunks AS t
USING (VALUES (?,?,?,?)) AS s(doc_id, chunk_id, src_source, src_text)
ON t.chunk_id = s.chunk_id
WHEN MATCHED THEN
  UPDATE SET
    t.doc_id = s.doc_id,
    t.[source] = s.src_source,
    t.[text] = s.src_text
WHEN NOT MATCHED THEN
  INSERT (doc_id, chunk_id, [source], [text])
  VALUES (s.doc_id, s.chunk_id, s.src_source, s.src_text);
"""


def ingest_jsonl_to_mssql(jsonl_path: str, *, upsert: bool = True, batch_size: int = 500) -> None:
    """Load chunks from a JSONL file into MSSQL.

    - If upsert=True (default), ingestion is idempotent by chunk_id via MERGE.
    - If upsert=False, a plain INSERT is used and will fail on duplicate chunk_id.
    """
    conn_str = os.environ["MSSQL_CONN_STR"]
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path.resolve()}")

    conn = pyodbc.connect(conn_str)
    cur = conn.cursor()
    cur.fast_executemany = True

    rows = []
    processed = 0
    skipped_short = 0

    sql = MERGE_UPSERT_SQL if upsert else (
        """
        INSERT INTO dbo.rag_chunks (doc_id, chunk_id, [source], [text])
        VALUES (?,?,?,?)
        """
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            text = (obj.get("text") or "").strip()
            if len(text) < 20:
                skipped_short += 1
                continue

            doc_id = str(
                obj.get("doc_id")
                or obj.get("document_id")
                or obj.get("source")
                or "unknown"
            )

            chunk_id = str(
                obj.get("chunk_id")
                or obj.get("id")
                or f"{doc_id}:{obj.get('chunk_index', 0)}"
            )

            source = obj.get("source") or obj.get("file") or doc_id

            rows.append((doc_id, chunk_id, source, text))

            if len(rows) >= batch_size:
                cur.executemany(sql, rows)
                conn.commit()
                processed += len(rows)
                rows.clear()

    if rows:
        cur.executemany(sql, rows)
        conn.commit()
        processed += len(rows)

    cur.close()
    conn.close()

    mode = "UPSERT (MERGE)" if upsert else "INSERT ONLY"
    print(f"Mode: {mode} | Processed: {processed} | Skipped (too short): {skipped_short}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest chunks.jsonl into MSSQL (idempotent upsert by chunk_id).")
    p.add_argument("jsonl_path", nargs="?", default="data_processed/chunks.jsonl", help="Path to chunks.jsonl")
    p.add_argument(
        "--upsert",
        action="store_true",
        default=True,
        help="Use MERGE upsert by chunk_id (default, safe to re-run).",
    )
    p.add_argument(
        "--insert-only",
        action="store_true",
        help="Use INSERT only (NOT idempotent; will fail on duplicate chunk_id).",
    )
    p.add_argument("--batch-size", type=int, default=500, help="Executemany batch size (default: 500)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    ingest_jsonl_to_mssql(
        args.jsonl_path,
        upsert=(not args.insert_only),
        batch_size=args.batch_size,
    )
