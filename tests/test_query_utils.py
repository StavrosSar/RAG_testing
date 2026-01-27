from src.query_utils import normalize_and_expand_query

def test_query_expansion_adds_synonyms():
    q = normalize_and_expand_query("AC voltage")
    assert "input voltage" in q or "vac" in q
