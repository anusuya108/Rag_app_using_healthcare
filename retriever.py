import os
import re
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────
# RERANKER
# ─────────────────────────────────────────────
reranker = CrossEncoder("BAAI/bge-reranker-large")


# ─────────────────────────────────────────────
# GROQ CLIENT — lazy load
# ─────────────────────────────────────────────
def get_groq_client():
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
        raise EnvironmentError("GROQ_API_KEY not found.")

    from groq import Groq
    return Groq(api_key=api_key)


# ─────────────────────────────────────────────
# RETRIEVAL + RERANKING
# FIX: fetch from MORE diverse sources
# ─────────────────────────────────────────────
def retrieve_chunks(vector_store: Chroma, question: str) -> List[Document]:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 20,           # fetch more candidates
            "fetch_k": 50,     # wider search pool
            "lambda_mult": 0.5, # FIX: lower = more diversity
        },
    )

    docs = retriever.invoke(question)

    if not docs:
        return []

    # Rerank
    pairs  = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # FIX: enforce source diversity — max 2 chunks per PDF
    seen_sources = {}
    diverse_docs = []
    for doc, score in ranked:
        source = doc.metadata.get("source", "unknown")
        count  = seen_sources.get(source, 0)
        if count < 2:
            diverse_docs.append(doc)
            seen_sources[source] = count + 1
        if len(diverse_docs) >= 6:
            break

    return diverse_docs


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[\d,\s]+\]", "", text)   # remove [1], [12, 14]
    text = re.sub(r"\(.*?\d{4}.*?\)", "", text) # remove (Author 2020)
    text = re.sub(r"et al\.", "", text)
    return text.strip()


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# FIX: use all 6 chunks for better coverage
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

        source = doc.metadata.get("source", "unknown")
        chunks.append(f"[Source: {source}]\n{text}")

    # FIX: use all 6 chunks, not just 4
    return "\n\n".join(chunks[:6])


# ─────────────────────────────────────────────
# LLM ANSWER
# FIX: better prompt for higher scores
# ─────────────────────────────────────────────
def generate_answer(question: str, docs: List[Document]) -> str:
    if not docs:
        return "I don't know."

    context = build_context(docs)

    prompt = f"""You are an expert healthcare AI assistant.

Answer the following question using ONLY the information provided in the context below.

STRICT RULES:
- Write a clear 2-3 sentence definition first
- Then write exactly 4 detailed bullet points that directly address the question
- Each bullet point must be at least 2 sentences long with specific details
- Use your own words — do NOT copy text from context
- Do NOT mention author names, citations, paper titles, or reference numbers
- Do NOT say "according to the context" or "based on the context"
- If the answer is genuinely not in the context, say exactly: "I don't know"

Context:
{context}

Question: {question}

Answer:"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800,   # FIX: allow longer answers → better coverage score
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# FALLBACK — show error clearly instead of
# silently returning chunk text
# ─────────────────────────────────────────────
def fallback_answer(question: str, docs: List[Document], error: str = "") -> str:
    if not docs:
        return "No relevant content found in the knowledge base."

    error_note = f"\n\n⚠️ LLM Error: {error}" if error else ""

    summary = []
    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)
        for s in sentences:
            s = clean_text(s)
            if len(s) > 80:
                summary.append(s)
            if len(summary) >= 5:
                break

    result = summary[0] if summary else "No relevant content found."
    result += "\n\nKey Points:\n"
    for s in summary[1:5]:
        result += f"- {s}\n"

    return result + error_note


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
        error_msg = str(e)
        print(f"LLM failed → fallback: {error_msg}")
        answer = fallback_answer(question, docs, error=error_msg)

    return answer, docs
