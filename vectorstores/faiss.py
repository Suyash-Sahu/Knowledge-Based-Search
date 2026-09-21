"""Build, save, and load a FAISS vector store, persisted locally to disk."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config


def build_faiss(documents: list[Document], embeddings: Embeddings) -> FAISS:
    """Embed documents into a new FAISS index and persist it to disk."""
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(config.FAISS_DIR)
    return store


def load_faiss(embeddings: Embeddings) -> FAISS:
    """Load an existing FAISS index from disk without re-embedding."""
    return FAISS.load_local(
        config.FAISS_DIR, embeddings, allow_dangerous_deserialization=True
    )


def faiss_index_exists() -> bool:
    """Whether a FAISS index has already been saved to disk."""
    return (Path(config.FAISS_DIR) / "index.faiss").exists()
