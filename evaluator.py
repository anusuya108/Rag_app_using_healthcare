from sentence_transformers import SentenceTransformer
import numpy as np

# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_answer(question, answer, docs):
    # ── 1. RELEVANCE (keyword overlap)
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    relevance = len(q_words & a_words) / max(len(q_words), 1)

    # ── 2. COVERAGE
    total_doc_length = sum(len(d.page_content.split()) for d in docs)
    answer_length = len(answer.split())
    coverage = min(answer_length / max(total_doc_length, 1), 1)

    # ── 3. DIVERSITY
    sources = set(d.metadata.get("source") for d in docs)
    diversity = len(sources) / max(len(docs), 1)

    # ── 4. SEMANTIC SIMILARITY ( NEW)
    q_emb = model.encode(question)
    a_emb = model.encode(answer)

    semantic_score = cosine_similarity(q_emb, a_emb)

    # ── FINAL SCORE (weighted)
    final_score = (
        0.25 * relevance +
        0.25 * coverage +
        0.2 * diversity +
        0.3 * semantic_score
    )

    return {
        "relevance": round(relevance, 2),
        "coverage": round(coverage, 2),
        "diversity": round(diversity, 2),
        "semantic": round(float(semantic_score), 2),
        "final_score": round(float(final_score * 100), 1)
    }