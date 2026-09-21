"""Smoke test: all four retrieval methods rank the relevant chunk first."""

import shutil
import tempfile

from langchain_core.documents import Document

import config
from embeddings.provider import get_embeddings
from vectorstores.chroma import build_chroma
from retrieval.search import search, search_with_scores, search_by_vector, search_as_retriever

DOCS = [
    Document(page_content="The Eiffel Tower is located in Paris, France.", metadata={"source": "a"}),
    Document(page_content="Python is a popular programming language for data science.", metadata={"source": "b"}),
    Document(page_content="The Great Wall of China stretches thousands of kilometers.", metadata={"source": "c"}),
]

QUERY = "Where is the Eiffel Tower?"


def test_all_retrieval_methods():
    tmp_dir = tempfile.mkdtemp()
    original_dir = config.CHROMA_DIR
    config.CHROMA_DIR = tmp_dir
    try:
        embeddings = get_embeddings()
        store = build_chroma(DOCS, embeddings)

        results = search(store, QUERY, k=1)
        assert "Paris" in results[0].page_content

        scored = search_with_scores(store, QUERY, k=1)
        doc, score = scored[0]
        assert "Paris" in doc.page_content
        assert isinstance(score, float)

        by_vector = search_by_vector(store, embeddings, QUERY, k=1)
        assert "Paris" in by_vector[0].page_content

        via_retriever = search_as_retriever(store, QUERY, k=1)
        assert "Paris" in via_retriever[0].page_content
    finally:
        config.CHROMA_DIR = original_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_all_retrieval_methods()
    print("search: OK")
