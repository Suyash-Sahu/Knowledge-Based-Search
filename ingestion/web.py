"""Load web pages into LangChain Documents."""

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_web(url: str) -> list[Document]:
    """Load a single web page into a list of LangChain Documents."""
    return WebBaseLoader(url).load()
