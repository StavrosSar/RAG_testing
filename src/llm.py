# prompt building helpers and Ollama API calls
import os
import requests
from typing import List, Dict

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def build_rag_prompt(question: str, hits: List[Dict]) -> str:
    """
    Build a prompt for the LLM using retrieved chunks.
    `hits` are dicts like: {doc_id, chunk_id, score, text, source}
    """
    context_blocks = []
    for h in hits:
        chunk_id = h.get("chunk_id", "unknown_chunk")
        doc_id = h.get("doc_id", "unknown_doc")
        text = (h.get("text") or "").strip()
        if not text:
            continue
        context_blocks.append(
            f"[{chunk_id} | {doc_id}]\n{text}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""
        You are a technical assistant.

        Rules:
        - Use ONLY the context below.
        - If the answer is not explicitly stated, say: "I don't know."
        - If the question asks for a part number, voltage, size, or value:
        - Extract the EXACT value as written.
        - Do NOT guess or infer.
        - Prefer values from tables over narrative text.
        - Return the value first, then a short explanation if helpful.
        - Answer in 1–2 sentences maximum.

        Context:
        {context}

        Question:
        {question}

        Answer:

        """.strip()


    return prompt


def ask_ollama(prompt: str, model: str = None, timeout: int = 120) -> str:
    model = model or OLLAMA_MODEL
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def answer_with_llm(question: str, hits: List[Dict]) -> str:
    prompt = build_rag_prompt(question, hits)
    return ask_ollama(prompt)
