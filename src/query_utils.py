"""Small, deterministic query normalization + lightweight expansion.

The goal is not "semantic" rewriting, but to reduce brittleness in technical
manual queries (e.g., AC voltage vs input voltage, environmental conditions).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


_WS_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("a/c", "ac")
    s = s.replace("a.c.", "ac")
    s = s.replace("a.c", "ac")
    s = s.replace("v~", "vac")
    s = s.replace("v⎓", "vdc")
    # collapse whitespace
    s = _WS_RE.sub(" ", s)
    return s


# Pairs of (pattern, expansions).
# If pattern is found (case-insensitive) in the normalized query, append expansions.
_EXPANSIONS: List[Tuple[re.Pattern, List[str]]] = [
    (re.compile(r"\bac\s+voltage\b", re.I), ["input voltage", "vac"]),
    (re.compile(r"\binput\s+voltage\b", re.I), ["ac voltage", "vac"]),
    (re.compile(r"\bfrequency\b", re.I), ["hz", "operating frequency"]),
    (re.compile(r"\benvironmental\s+conditions\b", re.I), ["temperature", "humidity", "operating environment"]),
]


def normalize_and_expand_query(query: str) -> str:
    """Return a normalized query with a few safe expansions appended.

    This intentionally stays conservative so it doesn't drift away from
    the user's intent.
    """
    q = _normalize_text(query)
    if not q:
        return q

    additions: List[str] = []
    for pat, exps in _EXPANSIONS:
        if pat.search(q):
            additions.extend(exps)

    if additions:
        # de-dup while preserving order
        seen = set()
        deduped = []
        for a in additions:
            key = a.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)
        q = f"{q} {' '.join(deduped)}"

    return _WS_RE.sub(" ", q).strip()
