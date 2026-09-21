"""Provide the embedding model used to turn text into vectors.

Uses Ollama to run nomic-embed-text locally: no API keys, no data leaving
the machine. Every chunk and every search query gets embedded with this
same model so their vectors live in the same space and can be compared.
"""

from langchain_ollama import OllamaEmbeddings

import config


def get_embeddings() -> OllamaEmbeddings:
    """Return the Ollama embedding client for the configured model."""
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
