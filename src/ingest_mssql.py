import os
import json
from pathlib import Path
import pyodbc

def ingest_jsonl_to_mssql(jsonl_path: str):
    conn_str = os.environ["MSSQL_CONN_STR"]
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path.resolve()}")

    conn = pyodbc.connect(conn_str)
    cur = conn.cursor()
    cur.fast_executemany = True

    rows = []
    inserted = 0
    skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # Adjust keys safely
            text = (obj.get("text") or "").strip()
            if len(text) < 20:
                skipped += 1
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

            if len(rows) >= 500:
                cur.executemany(
                    """
                    INSERT INTO dbo.rag_chunks (doc_id, chunk_id, source, text)
                    VALUES (?,?,?,?)
                    """,
                    rows
                )
                conn.commit()
                inserted += len(rows)
                rows.clear()

    if rows:
        cur.executemany(
            """
            INSERT INTO dbo.rag_chunks (doc_id, chunk_id, source, text)
            VALUES (?,?,?,?)
            """,
            rows
        )
        conn.commit()
        inserted += len(rows)

    cur.close()
    conn.close()
    print(f"Inserted: {inserted} | Skipped (too short): {skipped}")

if __name__ == "__main__":
    ingest_jsonl_to_mssql("data_processed/chunks.jsonl")
