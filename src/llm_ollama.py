#calls Ollama
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.1:8b"

def build_context(hits, max_chars=3500):
    parts = []
    total = 0
    for h in hits:
        text = (h.get("text") or "").strip()
        cid = h.get("chunk_id", "chunk")
        if not text:
            continue
        snippet = f"[{cid}]\n{text}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)

def rag_prompt(question: str, context: str) -> str:
    return f"""You are a technical assistant.
    Answer the question using ONLY the information in the context.
    If the answer is not in the context, say: I don't know.

    Context:
    {context}

    Question: {question}

    Answer (be concise and factual):"""

def ask_ollama(prompt: str, model: str = MODEL) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={"model": model,
                "prompt": prompt, 
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200}},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"].strip()

def answer_with_llm(question: str, hits):
    context = build_context(hits)
    prompt = rag_prompt(question, context)
    answer = ask_ollama(prompt)

    citations = []
    for h in hits:
        cid = h.get("chunk_id")
        if cid and cid not in citations:
            citations.append(cid)

    return {"answer": answer, "citations": citations}
