#SQL loader
import os
from typing import List, Dict, Any
import pyodbc

def load_chunks_from_mssql(limit: int | None = None) -> List[Dict[str, Any]]:
    conn = pyodbc.connect(os.environ["MSSQL_CONN_STR"])
    cur = conn.cursor()

    sql = f"""
    SELECT {"TOP (" + str(limit) + ")" if limit else ""}
           doc_id, chunk_id, source, text
    FROM dbo.rag_chunks
    ORDER BY id;
    """

    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "doc_id": r.doc_id,
            "chunk_id": r.chunk_id,
            "source": r.source,
            "text": r.text,
        }
        for r in rows
    ]
