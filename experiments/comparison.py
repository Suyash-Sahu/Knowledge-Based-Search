"""Compare retrieval configurations: FAISS vs Chroma, and chunk size variations.

Builds throwaway in-memory vector stores for each comparison run, so
experimenting never touches (or overwrites) the persisted knowledge base
under storage/.
"""

import time

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from processing.chunker import chunk_documents
from retrieval.search import search_with_scores


def compare_faiss_vs_chroma(
    documents: list[Document], embeddings: Embeddings, query: str, k: int = 4
) -> dict:
    """Build in-memory FAISS and Chroma stores from the same chunks and compare."""
    builders = {
        "FAISS": lambda: FAISS.from_documents(documents, embeddings),
        "Chroma": lambda: Chroma.from_documents(documents, embeddings),
    }
    results = {}
    for name, build_fn in builders.items():
        start = time.perf_counter()
        store = build_fn()
        build_seconds = time.perf_counter() - start

        start = time.perf_counter()
        hits = search_with_scores(store, query, k=k)
        search_seconds = time.perf_counter() - start

        results[name] = {
            "build_seconds": build_seconds,
            "search_seconds": search_seconds,
            "hits": hits,
        }
    return results


def compare_chunk_sizes(
    raw_documents: list[Document],
    embeddings: Embeddings,
    query: str,
    chunk_sizes: list[int],
    chunk_overlap: int = 200,
    k: int = 4,
) -> dict:
    """Chunk the same source documents at different sizes and compare search results."""
    results = {}
    for size in chunk_sizes:
        chunks = chunk_documents(raw_documents, chunk_size=size, chunk_overlap=chunk_overlap)
        store = FAISS.from_documents(chunks, embeddings)
        hits = search_with_scores(store, query, k=k)
        results[size] = {"num_chunks": len(chunks), "hits": hits}
    return results
