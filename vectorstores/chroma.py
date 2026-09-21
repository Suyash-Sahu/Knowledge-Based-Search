"""Build and load a Chroma vector store, persisted locally to disk."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import config


def build_chroma(documents: list[Document], embeddings: Embeddings) -> Chroma:
    """Embed documents and persist them into a local Chroma collection."""
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
    )


def load_chroma(embeddings: Embeddings) -> Chroma:
    """Load the existing persisted Chroma collection without re-embedding."""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )


def chroma_has_data(embeddings: Embeddings) -> bool:
    """Whether the persisted Chroma collection already contains chunks."""
    return len(load_chroma(embeddings).get()["ids"]) > 0
