"""Smoke test: experiment helpers run and don't touch persisted storage/."""

from pathlib import Path

from langchain_core.documents import Document

import config
from embeddings.provider import get_embeddings
from experiments.comparison import compare_faiss_vs_chroma, compare_chunk_sizes

DOCS = [
    Document(page_content="The Eiffel Tower is located in Paris, France.", metadata={"source": "a"}),
    Document(page_content="Python is a popular programming language for data science.", metadata={"source": "b"}),
]

LONG_DOC = [Document(page_content="The Eiffel Tower stands in Paris. " * 100, metadata={"source": "c"})]


def test_compare_faiss_vs_chroma_does_not_touch_storage():
    faiss_before = list(Path(config.FAISS_DIR).glob("*")) if Path(config.FAISS_DIR).exists() else []
    chroma_before = list(Path(config.CHROMA_DIR).glob("*")) if Path(config.CHROMA_DIR).exists() else []

    embeddings = get_embeddings()
    results = compare_faiss_vs_chroma(DOCS, embeddings, "Where is the Eiffel Tower?", k=1)

    assert set(results.keys()) == {"FAISS", "Chroma"}
    for name, r in results.items():
        assert "Paris" in r["hits"][0][0].page_content, name
        assert r["build_seconds"] >= 0
        assert r["search_seconds"] >= 0

    faiss_after = list(Path(config.FAISS_DIR).glob("*")) if Path(config.FAISS_DIR).exists() else []
    chroma_after = list(Path(config.CHROMA_DIR).glob("*")) if Path(config.CHROMA_DIR).exists() else []
    assert faiss_before == faiss_after
    assert chroma_before == chroma_after


def test_compare_chunk_sizes():
    embeddings = get_embeddings()
    results = compare_chunk_sizes(
        LONG_DOC, embeddings, "Where is the Eiffel Tower?", chunk_sizes=[100, 500], chunk_overlap=20
    )

    assert set(results.keys()) == {100, 500}
    assert results[100]["num_chunks"] > results[500]["num_chunks"]
    for size, r in results.items():
        assert "Paris" in r["hits"][0][0].page_content, size


if __name__ == "__main__":
    test_compare_faiss_vs_chroma_does_not_touch_storage()
    print("comparison (faiss vs chroma): OK")
    test_compare_chunk_sizes()
    print("comparison (chunk sizes): OK")
