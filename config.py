import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_FOLDER = os.path.join(BASE_DIR, "data")
    CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

    # Embeddings
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

    # Retrieval
    RETRIEVAL_K = 5
    RETRIEVAL_FETCH_K = 20
    RETRIEVAL_LAMBDA = 0.6

config = Config()