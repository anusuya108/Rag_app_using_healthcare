import os
import re
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────
# RERANKER — loaded once at import
# ─────────────────────────────────────────────
reranker = CrossEncoder("BAAI/bge-reranker-large")


# ─────────────────────────────────────────────
# GROQ CLIENT — lazy load (safe for Streamlit)
# ─────────────────────────────────────────────
def get_groq_client():
    """
    Load Groq client at call time — not at import time.
    Checks Streamlit secrets first, then env var.
    """
    api_key = None

    # 1. Streamlit Cloud secrets
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # 2. Local .env
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found.\n"
            "Local: add to .env file.\n"
            "Streamlit Cloud: App Settings → Secrets → add GROQ_API_KEY."
        )

    from groq import Groq
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────
# RETRIEVAL + RERANKING
# ─────────────────────────────────────────────
def retrieve_chunks(vector_store: Chroma, question: str) -> List[Document]:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 15,
            "fetch_k": 30,
            "lambda_mult": 0.7,
        },
    )

    docs = retriever.invoke(question)

    if not docs:
        return []

    # Rerank
    pairs  = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:5]]


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(.*?\d{4}.*?\)", "", text)
    text = re.sub(r"et al\.", "", text)
    return text.strip()


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────
def build_context(docs: List[Document]) -> str:
    seen = set()
    chunks = []

    for doc in docs:
        text = clean_text(doc.page_content)

        if not text or len(text) < 50:
            continue

        key = text[:120]
        if key in seen:
            continue
        seen.add(key)

        chunks.append(text)

    return "\n\n".join(chunks[:4])


# ─────────────────────────────────────────────
# LLM ANSWER
# ─────────────────────────────────────────────
def generate_answer(question: str, docs: List[Document]) -> str:
    if not docs:
        return "I don't know."

    context = build_context(docs)

    prompt = f"""You are a healthcare AI assistant.

Answer the question using ONLY the context below.

Rules:
- Start with a clear definition (1-2 lines)
- Then give 3-4 bullet points explaining it
- Do NOT include author names, citations, or paper titles
- Do NOT copy text directly
- Be concise and precise
- If the answer is not in the context, say "I don't know"

Context:
{context}

Question:
{question}

Answer:"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# FALLBACK (no LLM)
# ─────────────────────────────────────────────
def fallback_answer(question: str, docs: List[Document]) -> str:
    if not docs:
        return "No relevant content found."

    summary = []

    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        for s in sentences:
            s = clean_text(s)
            if len(s) > 80:
                summary.append(s)
            if len(summary) >= 4:
                break

    answer = summary[0] if summary else "No relevant content found."
    answer += "\n\nKey Points:\n"
    for s in summary[1:4]:
        answer += f"- {s}\n"

    return answer


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def answer_question(
    question: str,
    vector_store: Chroma = None,
) -> Tuple[str, List[Document]]:

    if vector_store is None:
        from ingestion import run_ingestion_pipeline
        vector_store = run_ingestion_pipeline("data")

    docs = retrieve_chunks(vector_store, question)

    try:
        answer = generate_answer(question, docs)
    except Exception as e:
        print(f"LLM failed → fallback: {e}")
        answer = fallback_answer(question, docs)

    return answer, docs