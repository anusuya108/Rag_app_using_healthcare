"""
app.py — Healthcare RAG System
Streamlit Cloud ready
"""

import streamlit as st

# ── Page config — must be first ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Healthcare RAG",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI in Healthcare RAG")
st.write("Ask questions from healthcare research papers with explainable AI.")


# ── Load & cache vector store (runs once per session) ────────────────────────
@st.cache_resource(show_spinner="📚 Loading and indexing research papers...")
def load_rag_pipeline():
    """
    Builds the in-memory vector store from PDFs in the data/ folder.
    Cached — only runs once when the app starts.
    """
    from ingestion import run_ingestion_pipeline
    return run_ingestion_pipeline("data")


# ── Build vector store at startup ────────────────────────────────────────────
try:
    vector_store = load_rag_pipeline()
    st.success("✅ Knowledge base loaded successfully.")
except FileNotFoundError:
    st.error(
        "❌ No `data/` folder found. "
        "Please create a `data/` folder in your repo and add PDF files to it."
    )
    st.stop()
except ValueError as e:
    st.error(f"❌ Ingestion error: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load knowledge base: {e}")
    st.stop()

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_input("Enter your question")

# ── Answer button ─────────────────────────────────────────────────────────────
if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    with st.spinner("Analyzing documents..."):

        try:
            from retriever import answer_question
            from evaluator import evaluate_answer

            answer, docs = answer_question(question, vector_store)

        except Exception as e:
            st.error(f"❌ Runtime Error: {e}")
            st.stop()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("### 💬 Answer")
    st.write(answer)

    # ── Sources ───────────────────────────────────────────────────────────────
    st.markdown("### 📄 Retrieved Sources")

    if docs:
        for i, doc in enumerate(docs):
            with st.expander(
                f"Chunk {i+1} | "
                f"{doc.metadata.get('source', 'unknown')} | "
                f"Page {doc.metadata.get('page', '?')}"
            ):
                st.write(doc.page_content[:800])
    else:
        st.warning("No relevant documents retrieved.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Evaluation Metrics")

    metrics = evaluate_answer(question, answer, docs)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Relevance", round(metrics.get("relevance",   0), 2))
    col2.metric("Coverage",  round(metrics.get("coverage",    0), 2))
    col3.metric("Diversity", round(metrics.get("diversity",   0), 2))
    col4.metric("Semantic",  round(metrics.get("semantic",    0), 2))
    col5.metric("Score",     f"{metrics.get('final_score',    0)}%")

    # ── Debug ─────────────────────────────────────────────────────────────────
    with st.expander("🛠 Debug: Raw Retrieved Chunks"):
        if docs:
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
        else:
            st.write("No chunks retrieved.")