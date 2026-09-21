"""Query a vector store for the most relevant chunks.

Exposes the four retrieval methods named in the TRD as small, distinct
functions so each one can be understood (and compared) on its own:
- similarity_search: plain top-k lookup
- similarity_search_with_score: top-k plus a distance/similarity score
- similarity_search_by_vector: search from a vector you already computed
- as_retriever().invoke(): the standard LangChain retriever interface
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

import config


def search(vectorstore: VectorStore, query: str, k: int = config.DEFAULT_TOP_K) -> list[Document]:
    """Return the top-k most similar chunks to the query."""
    return vectorstore.similarity_search(query, k=k)


def search_with_scores(
    vectorstore: VectorStore, query: str, k: int = config.DEFAULT_TOP_K
) -> list[tuple[Document, float]]:
    """Return the top-k most similar chunks paired with a distance/similarity score."""
    return vectorstore.similarity_search_with_score(query, k=k)


def search_by_vector(
    vectorstore: VectorStore, embeddings: Embeddings, query: str, k: int = config.DEFAULT_TOP_K
) -> list[Document]:
    """Embed the query manually, then search using that vector directly."""
    query_vector = embeddings.embed_query(query)
    return vectorstore.similarity_search_by_vector(query_vector, k=k)


def search_as_retriever(vectorstore: VectorStore, query: str, k: int = config.DEFAULT_TOP_K) -> list[Document]:
    """Search via the retriever interface (the API LangChain chains expect)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
