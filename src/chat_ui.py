# Streamlit chat UI (Ollama + RAG API)

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="RAG Chat (Ollama + SQL)", layout="wide")

DEFAULT_SYSTEM_PROMPT = """You are a friendly technical assistant.

Rules:
- Use ONLY the context below.
- If the answer is not in the context, say: "I don't know."
- Answer in 1–3 sentences.
- If the question asks for a routine/function, name it and briefly explain what it replaces/does.

Context:
{context}

Question:
{question}

Answer:
""".strip()

# ---- Session state init ----
if "prompt_template" not in st.session_state:
    st.session_state.prompt_template = DEFAULT_SYSTEM_PROMPT

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Sidebar ----
with st.sidebar:
    st.header("Prompt Settings")

    st.session_state.prompt_template = st.text_area(
        "Edit system prompt (keep {context} and {question})",
        value=st.session_state.prompt_template,
        height=260,
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Reset prompt"):
            st.session_state.prompt_template = DEFAULT_SYSTEM_PROMPT
            st.rerun()

    with colB:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.subheader("Query Settings")
    retriever = st.selectbox(
        "Retriever",
        ["bm25_sql", "tfidf_sql", "bm25", "tfidf"],
        index=0,
    )
    mode = st.selectbox("Mode", ["llm", "extractive"], index=0)
    top_k = st.slider("Top K", 1, 10, 5)

    show_sources = st.checkbox("Show sources (top_k_filtered)", value=False)

# ---- Main ----
st.title("RAG Chat (Ollama + SQL)")

# Render existing messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
user_msg = st.chat_input("Ask a question about the manuals...")

if user_msg:
    # show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    payload = {
        "question": user_msg,
        "top_k": top_k,
        "retriever": retriever,
        "mode": mode,
        # Optional: if your API supports it, you can pass prompt_template too.
        # Otherwise the API will use its own prompt internally.
        # "prompt_template": st.session_state.prompt_template,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        # Your API may return either:
        # - answer: "text"
        # - answer: {"answer": "text", "citations": [...]}
        raw_answer = data.get("answer", "")

        if isinstance(raw_answer, dict):
            answer_text = raw_answer.get("answer", "")
            citations = raw_answer.get("citations", data.get("citations", []))
        else:
            answer_text = raw_answer
            citations = data.get("citations", [])

        assistant_text = answer_text.strip() if answer_text else "I don't know."

        if citations:
            assistant_text += "\n\n**Citations:**\n" + "\n".join([f"- {c}" for c in citations])

        if show_sources:
            hits = data.get("top_k_filtered", [])
            if hits:
                assistant_text += "\n\n---\n### Retrieved context (top_k_filtered)\n"
                for h in hits[: min(len(hits), 5)]:
                    assistant_text += f"\n**{h.get('chunk_id','(no id)')}**\n\n{(h.get('text','')[:1200]).strip()}...\n"

    except Exception as e:
        assistant_text = f"❌ Error calling API: {e}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
