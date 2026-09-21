"""Smoke test: each loader turns a fixture into LangChain Documents with text."""

from pathlib import Path

from ingestion.text import load_text
from ingestion.pdf import load_pdf

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_text():
    docs = load_text(str(FIXTURES / "sample.txt"))
    assert len(docs) == 1
    assert "Ollama" in docs[0].page_content
    assert docs[0].metadata["source"].endswith("sample.txt")


def test_load_pdf():
    docs = load_pdf(str(FIXTURES / "sample.pdf"))
    assert len(docs) == 1
    assert "Hello Knowledge Base" in docs[0].page_content


if __name__ == "__main__":
    test_load_text()
    test_load_pdf()
    print("ingestion: OK")
