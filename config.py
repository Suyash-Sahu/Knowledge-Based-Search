"""Central configuration for the Personal Knowledge Base Search app."""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# Embeddings (TRD: Ollama + nomic-embed-text)
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

# Chunking (TRD: RecursiveCharacterTextSplitter, configurable size/overlap)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Vector store persistence (TRD section 5: store locally, survive restarts)
CHROMA_DIR = str(BASE_DIR / "storage" / "chroma")
FAISS_DIR = str(BASE_DIR / "storage" / "faiss")
CHROMA_COLLECTION_NAME = "knowledge_base"

# Search
DEFAULT_TOP_K = 4
