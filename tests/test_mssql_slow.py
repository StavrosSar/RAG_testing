import os
import pytest
from src.chunks_mssql import load_chunks_from_mssql

@pytest.mark.slow
def test_load_chunks_from_mssql_requires_env():
    conn_str = os.environ.get("MSSQL_CONN_STR")
    if not conn_str:
        pytest.skip("MSSQL_CONN_STR not set; skipping DB integration test")

    rows = load_chunks_from_mssql(limit=5)
    assert len(rows) > 0
    assert "chunk_id" in rows[0]