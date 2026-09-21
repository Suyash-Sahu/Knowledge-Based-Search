"""Load .txt files into LangChain Documents."""

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_text(file_path: str) -> list[Document]:
    """Load a single .txt file into a list of LangChain Documents."""
    return TextLoader(file_path, encoding="utf-8").load()
