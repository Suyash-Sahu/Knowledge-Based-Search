"""Smoke test: chunking splits long text into overlapping, bounded pieces."""

from langchain_core.documents import Document

from processing.chunker import chunk_documents


def test_chunk_documents():
    long_text = "abcdefghij " * 200  # 2200 chars
    docs = [Document(page_content=long_text, metadata={"source": "test"})]

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all(c.metadata["source"] == "test" for c in chunks)


if __name__ == "__main__":
    test_chunk_documents()
    print("chunker: OK")
