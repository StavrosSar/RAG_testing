#extractive answers
import re
from typing import List, Dict

print("LOADED: src/answer.py")

# ----------------------------
# helpers
# ----------------------------

TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ω0-9]+", re.UNICODE)
STOPWORDS = {
    # EN (mini list)
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "as", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "you", "your", "yours", "we", "our", "ours", "they", "their", "them",
    "how", "what", "when", "where", "why", "which",
    "it", "this", "that", "these", "those",
    "can", "could", "should", "would", "may", "might", "must",
}

HEADINGISH_RE = re.compile(
    r"""^(
        \d+(\.\d+)+\b            # 2.7.2
      | \d+\-\d+\b               # 2-20
      | (chapter|appendix)\s+\w+ # Chapter 2, Appendix A
      | (contents|index)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

FACT_PATTERNS = [
    r"\bwhat\b.*\b(range|voltage|frequency|size|capacity|version|limit)\b",
    r"\bhow (many|much)\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\bsupported\b",
]


def cap_words(text: str, max_words: int = 80) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def tokens(s: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(s) if t.lower() not in STOPWORDS]


def sent_score(query: str, sent: str) -> int:
    q = set(tokens(query))
    s = set(tokens(sent))
    return len(q.intersection(s))


def is_fact_question(q: str) -> bool:
    q = q.lower()
    return any(re.search(p, q) for p in FACT_PATTERNS)


def is_headingish(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # obvious section/page artifacts
    if HEADINGISH_RE.match(s):
        return True
    # “Designing Modular Procedures” (Title Case short lines) are headings too
    if len(s) < 60 and s == s.title():
        return True
    # too many digits/punct relative to letters
    letters = sum(ch.isalpha() for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    if letters < 8 and digits >= 2:
        return True
    return False


def fact_signal_score(sentence: str) -> int:
    """
    Boost sentences that look like factual specifications.
    """
    score = 0
    if re.search(r"\b\d+\s*(to|\-)\s*\d+\b", sentence):
        score += 3  # numeric ranges
    if re.search(r"\b\d+\s*(v|vac|hz|w|kw|mhz|ghz|mb|gb|tb)\b", sentence, re.I):
        score += 3  # number + unit
    if re.search(r"\b(range|maximum|minimum|operating|nominal|rated)\b", sentence, re.I):
        score += 2
    if ":" in sentence:
        score += 1  # table-like specs
    return score


def split_sentences(text: str) -> List[str]:
    # 1) break into lines, drop heading-ish lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not is_headingish(ln)]

    # 2) re-join and sentence split
    cleaned = " ".join(lines)
    sents = re.split(r"(?<=[.!?])\s+", cleaned)

    # keep substantial sentences only
    return [s.strip() for s in sents if len(s.strip()) >= 60]


def looks_like_noise(text: str) -> bool:
    if not text:
        return True
    sample = text[:800]
    symbol_ratio = sum(1 for ch in sample if ch in "-_*~|") / max(len(sample), 1)
    if symbol_ratio > 0.05:
        return True
    if sample.count("---") >= 8:
        return True
    return False


def extract_relevant_clause(query: str, sentence: str) -> str:
    """
    Extract the shortest clause containing query keywords.
    Splits on ';' and ':' and returns the highest-overlap, shortest clause.
    """
    q_tokens = set(tokens(query))
    parts = re.split(r"[;:]", sentence)

    scored = []
    for p in parts:
        score = len(q_tokens.intersection(tokens(p)))
        if score > 0:
            scored.append((score, p.strip()))

    if not scored:
        return sentence.strip()

    # highest score, then shortest clause
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return scored[0][1]


# ----------------------------
# main function
# ----------------------------

def answer_with_citations(
    query: str,
    hits: List[Dict],
    max_sentences: int = 3,
):
    """
    Extractive answer with citations.
    Selects the most query-relevant sentences from retrieved chunks.
    For fact-style questions, extracts a short clause and returns only 1 sentence.
    """
    fact_mode = is_fact_question(query)

    selected = []      # list of (sentence, chunk_id)
    citations = []     # ordered unique chunk_ids

    q_low = query.lower()

    for h in hits:
        chunk_id = h["chunk_id"]
        text = h["text"]

        if looks_like_noise(text):
            continue

        sents = split_sentences(text)
        if not sents:
            continue

        ranked = sorted(
            sents,
            key=lambda s: (
                sent_score(query, s)
                + (fact_signal_score(s) if fact_mode else 0)
            ),
            reverse=True,
        )


        for s in ranked:
            s_low = s.lower()

            # if query is about returning, avoid signaling sentences
            if "return" in q_low and "signal" in s_low:
                continue

            if sent_score(query, s) == 0:
                continue

            if fact_mode:
                s = extract_relevant_clause(query, s)

            selected.append((s, chunk_id))
            if chunk_id not in citations:
                citations.append(chunk_id)

            # Fact questions: stop after first good hit
            if fact_mode:
                break

            if len(selected) >= max_sentences:
                break

        if fact_mode and selected:
            break

        if len(selected) >= max_sentences:
            break

    # Fact mode: keep only the best single sentence
    if fact_mode and selected:
        selected = selected[:1]

    if not selected:
        return {
            "answer": "No relevant information found in the retrieved documents.",
            "citations": [],
        }

    answer_parts = []
    for sent, cid in selected:
        answer_parts.append(f"{sent} [{cid}]")

    answer = " ".join(answer_parts)
    answer = cap_words(answer, 80)

    return {
        "answer": answer,
        "citations": citations,
    }
