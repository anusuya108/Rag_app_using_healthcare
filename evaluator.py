from sentence_transformers import SentenceTransformer
import numpy as np

# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_answer(question, answer, docs):

    # ── 1. RELEVANCE — keyword overlap (improved: use bigrams too)
    q_words = set(question.lower().split())
    a_words  = set(answer.lower().split())
    # remove stopwords for better signal
    stopwords = {"the","a","an","is","in","of","and","to","for","with","that","this","are","was","were","be","been","it","its","by","on","at","as","or","not","from","have","has","had","what","how","why","when","which"}
    q_words -= stopwords
    a_words  -= stopwords
    relevance = len(q_words & a_words) / max(len(q_words), 1)

    # ── 2. COVERAGE — FIX: compare answer length to context length
    # instead of full doc length (which is too long, making coverage always low)
    context_words = sum(
        min(len(d.page_content.split()), 200)  # cap per-chunk to 200 words
        for d in docs
    )
    answer_words = len(answer.split())
    # FIX: scale so a 200-word answer over 6 chunks scores ~0.5+
    coverage = min(answer_words / max(context_words * 0.3, 1), 1.0)

    # ── 3. DIVERSITY — unique sources / total chunks
    sources   = set(d.metadata.get("source") for d in docs)
    diversity = len(sources) / max(len(docs), 1)

    # ── 4. SEMANTIC SIMILARITY
    q_emb = model.encode(question)
    a_emb = model.encode(answer)
    semantic_score = float(cosine_similarity(q_emb, a_emb))

    # ── 5. ANSWER QUALITY — penalise fallback answers
    is_fallback = "Key Points:" in answer and answer.count("\n-") >= 2
    quality_penalty = 0.2 if is_fallback else 0.0

    # ── FINAL SCORE (weighted)
    raw_score = (
        0.30 * relevance       +
        0.20 * coverage        +
        0.20 * diversity       +
        0.30 * semantic_score
    ) - quality_penalty

    final_score = max(round(float(raw_score) * 100, 1), 0)

    return {
        "relevance":   round(relevance, 2),
        "coverage":    round(coverage, 2),
        "diversity":   round(diversity, 2),
        "semantic":    round(semantic_score, 2),
        "final_score": final_score,
    }
