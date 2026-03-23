"""
app.py — Healthcare RAG System
"""

import streamlit as st

st.set_page_config(
    page_title="AI Healthcare RAG",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI in Healthcare RAG")
st.write("Ask questions from healthcare research papers with explainable AI.")


# ── Load vector store once ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 Loading and indexing research papers...")
def load_rag_pipeline():
    from ingestion import run_ingestion_pipeline
    return run_ingestion_pipeline("data")


try:
    vector_store = load_rag_pipeline()
    st.success("✅ Knowledge base loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load knowledge base: {e}")
    st.stop()

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_input("Enter your question")

if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    with st.spinner("Analyzing documents..."):

        # ── Step 1: retrieve chunks ───────────────────────────────────────────
        try:
            from retriever import retrieve_chunks, generate_answer, fallback_answer, build_context
            docs = retrieve_chunks(vector_store, question)
        except Exception as e:
            st.error(f"❌ Retrieval Error: {e}")
            st.stop()

        # ── Step 2: generate answer with visible error handling ───────────────
        llm_error = None
        answer = None

        try:
            # Test API key is accessible
            api_key = None
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
            if not api_key:
                import os
                api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise EnvironmentError("GROQ_API_KEY not found in Streamlit secrets. Go to App Settings → Secrets and add it.")

            answer = generate_answer(question, docs)

        except Exception as e:
            llm_error = str(e)
            answer = fallback_answer(question, docs)

        # ── Step 3: evaluate ─────────────────────────────────────────────────
        try:
            from evaluator import evaluate_answer
            metrics = evaluate_answer(question, answer, docs)
        except Exception as e:
            metrics = {"relevance": 0, "coverage": 0, "diversity": 0, "semantic": 0, "final_score": 0}

    # ── Show LLM error prominently if it failed ───────────────────────────────
    if llm_error:
        st.error(f"⚠️ LLM failed — showing fallback answer. Error: {llm_error}")
        st.info("💡 Fix: Go to Streamlit Cloud → App Settings → Secrets → make sure GROQ_API_KEY is set correctly.")

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("### 💬 Answer")
    st.write(answer)

    # ── Sources ───────────────────────────────────────────────────────────────
    st.markdown("### 📄 Retrieved Sources")
    if docs:
        for i, doc in enumerate(docs):
            with st.expander(
                f"Chunk {i+1} | {doc.metadata.get('source', 'unknown')} | Page {doc.metadata.get('page', '?')}"
            ):
                st.write(doc.page_content[:800])
    else:
        st.warning("No relevant documents retrieved.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Evaluation Metrics")
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
                st.markdown(f"**Chunk {i+1}** — `{doc.metadata.get('source','?')}` p.{doc.metadata.get('page','?')}")
                st.write(doc.page_content)
        else:
            st.write("No chunks retrieved.")

    # ── Context sent to LLM ──────────────────────────────────────────────────
    with st.expander("🔍 Debug: Context sent to LLM"):
        st.code(build_context(docs))
