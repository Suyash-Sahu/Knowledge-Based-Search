"""Smoke test: both vector stores persist to disk and reload without
re-embedding, then correctly retrieve the most relevant chunk for a query.

Uses temporary directories so this never touches the app's real storage/.
"""

import shutil
import tempfile

from langchain_core.documents import Document

import config
from embeddings.provider import get_embeddings
from vectorstores.chroma import build_chroma, load_chroma
from vectorstores.faiss import build_faiss, load_faiss

DOCS = [
    Document(page_content="The Eiffel Tower is located in Paris, France.", metadata={"source": "a"}),
    Document(page_content="Python is a popular programming language for data science.", metadata={"source": "b"}),
    Document(page_content="The Great Wall of China stretches thousands of kilometers.", metadata={"source": "c"}),
]


def test_chroma():
    tmp_dir = tempfile.mkdtemp()
    original_dir = config.CHROMA_DIR
    config.CHROMA_DIR = tmp_dir
    try:
        embeddings = get_embeddings()
        build_chroma(DOCS, embeddings)

        reloaded = load_chroma(embeddings)
        results = reloaded.similarity_search("Where is the Eiffel Tower?", k=1)

        assert len(results) == 1
        assert "Paris" in results[0].page_content
    finally:
        config.CHROMA_DIR = original_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_faiss():
    tmp_dir = tempfile.mkdtemp()
    original_dir = config.FAISS_DIR
    config.FAISS_DIR = tmp_dir
    try:
        embeddings = get_embeddings()
        build_faiss(DOCS, embeddings)

        reloaded = load_faiss(embeddings)
        results = reloaded.similarity_search("Where is the Eiffel Tower?", k=1)

        assert len(results) == 1
        assert "Paris" in results[0].page_content
    finally:
        config.FAISS_DIR = original_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_chroma()
    print("chroma vectorstore: OK")
    test_faiss()
    print("faiss vectorstore: OK")
