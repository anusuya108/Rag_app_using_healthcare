import os
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings import get_embedding_model


# ─────────────────────────────────────────────
# CLEAN TEXT
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"doi:\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+\]", "", text)

    if "references" in text.lower():
        return ""

    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────
# LOAD PDFs
# ─────────────────────────────────────────────
def load_pdfs(folder_path: str) -> List[Document]:
    documents = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not files:
        raise ValueError(f"No PDFs found in '{folder_path}' folder")

    print(f"Found {len(files)} PDFs: {files}")

    for file in files:
        path = os.path.join(folder_path, file)
        loader = PyPDFLoader(path)

        try:
            pages = loader.load()
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

        for i, page in enumerate(pages):
            cleaned = clean_text(page.page_content)

            if len(cleaned) < 120:
                continue

            documents.append(
                Document(
                    page_content=cleaned,
                    metadata={"source": file, "page": i},
                )
            )

    print(f"Loaded {len(documents)} pages")
    return documents


# ─────────────────────────────────────────────
# SPLIT INTO CHUNKS
# ─────────────────────────────────────────────
def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(documents)

    clean_chunks = []
    seen = set()

    for i, chunk in enumerate(chunks):
        text = chunk.page_content.strip()

        if len(text) < 200:
            continue

        key = text[:150]
        if key in seen:
            continue
        seen.add(key)

        chunk.metadata["chunk_id"] = i
        clean_chunks.append(chunk)

    print(f"Created {len(clean_chunks)} chunks")
    return clean_chunks


# ─────────────────────────────────────────────
# BUILD VECTOR STORE — IN MEMORY (no persist)
# ─────────────────────────────────────────────
def build_vector_store(chunks: List[Document]) -> Chroma:
    embedding = get_embedding_model()

    print("Building in-memory vector store...")

    # NO persist_directory → stays in memory
    # safe for Streamlit Cloud
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
    )

    print(f"Vector store ready with {len(chunks)} chunks")
    return vector_store


# ─────────────────────────────────────────────
# FULL PIPELINE — returns vector store
# ─────────────────────────────────────────────
def run_ingestion_pipeline(folder_path: str = "data") -> Chroma:
    print("Starting ingestion pipeline...")

    docs = load_pdfs(folder_path)

    if not docs:
        raise ValueError("No documents loaded from PDFs")

    chunks = split_documents(docs)

    if not chunks:
        raise ValueError("No chunks created from documents")

    vector_store = build_vector_store(chunks)

    print("Ingestion complete!")
    return vector_store


# ─────────────────────────────────────────────
# LOCAL RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_ingestion_pipeline("data")