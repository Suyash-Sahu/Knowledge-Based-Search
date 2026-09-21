"""Load PDF files into LangChain Documents (one Document per page)."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """Load a single PDF file into a list of LangChain Documents, one per page."""
    return PyPDFLoader(file_path).load()
