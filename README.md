# 🧠 Personal Knowledge Base Search

A Privacy-first, Local Retrieval-Augmented Generation (RAG) and Search engine built with **Streamlit**, **LangChain**, **Ollama**, **ChromaDB**, and **FAISS**.

This application allows you to index personal documents (PDF, TXT) and web pages into local vector stores and perform fast semantic search and benchmarking without sending data to external APIs.

---

## ✨ Key Features

- **📄 Multi-Source Ingestion**:
  - **PDF Documents**: Parsed page-by-page via `pypdf`.
  - **Text Files**: Directly indexed via standard loaders.
  - **Web Pages**: Scraped and structured using `BeautifulSoup4` (`WebBaseLoader`).
- **🦙 Local Embeddings (100% Private)**:
  - Powered by **Ollama** running the `nomic-embed-text` embedding model locally on `http://localhost:11434`.
- **🗄️ Dual Vector Store Persistence**:
  - **FAISS**: In-memory similarity search with local disk serialization.
  - **ChromaDB**: Native persistent vector database storing metadata and document collections.
- **🧪 Retrieval Experiments & Benchmarking**:
  - **FAISS vs Chroma Comparison**: Measure build times, query latency, and similarity score distributions side-by-side.
  - **Chunk Size Impact**: Test different chunk sizes (`250`, `500`, `1000`, `2000`) on retrieval performance.
- **🖥️ Streamlit Web Interface**: Clean, tabbed dashboard for managing knowledge base indexing, running search queries, and conducting performance experiments.

---

## 🏗️ System Architecture

```text
               Streamlit Web Interface
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  Ingestion Module                  Search Module
  (PDF / TXT / Web)               (Query Input + Top-K)
          │                               │
          ▼                               ▼
  Document Chunking               Query Embedding
(RecursiveTextSplitter)          (Ollama / nomic-embed)
          │                               │
          ▼                         ┌─────┴─────┐
  Vector Embeddings                 ▼           ▼
(nomic-embed-text)                FAISS       Chroma
          │                         └─────┬─────┘
          └───────────────┬───────────────┘
                          ▼
                 Ranked Search Results
                          │
                          ▼
             Streamlit Result Cards & Metrics
```

---

## 🛠️ Tech Stack

- **Frontend / Framework**: [Streamlit](https://streamlit.io/)
- **LLM & RAG Orchestration**: [LangChain](https://www.langchain.com/) (`langchain-community`, `langchain-chroma`, `langchain-ollama`, `langchain-text-splitters`)
- **Local Embedding Server**: [Ollama](https://ollama.com/) (`nomic-embed-text`)
- **Vector Stores**: [FAISS (cpu)](https://github.com/facebookresearch/faiss), [ChromaDB](https://www.trychroma.com/)
- **Document Parsers**: `PyPDF`, `BeautifulSoup4`
- **Testing**: `pytest`

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+** installed.
- **Ollama** installed and running locally.
  - Download Ollama from [ollama.com](https://ollama.com/).
  - Ensure Ollama service is active at `http://localhost:11434`.
  - Pull the embedding model used by the application:
    ```bash
    ollama pull nomic-embed-text
    ```

### 2. Environment Setup

Clone the repository (or navigate to the project directory) and activate your virtual environment:

#### Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt):
```cmd
.\.venv\Scripts\activate.bat
```

#### macOS / Linux:
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚡ Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Open your browser and navigate to:
`http://localhost:8501`

---

## 🧪 Running Unit Tests

Run the test suite with `pytest`:

```bash
pytest
```

---

## 📂 Project Structure

```text
Personal Knowledge Based Search/
├── app.py                   # Streamlit main entry point & UI tabs
├── config.py                # Central configuration (Ollama URLs, paths, chunk settings)
├── requirements.txt         # Project dependencies
├── architecture.md          # Architecture diagram reference
├── .gitignore               # Git ignore file for local vectors, cache, and virtual environment
├── embeddings/              # Local embedding provider initialization
│   └── provider.py
├── ingestion/               # Loaders for PDF, TXT, and Web URLs
│   ├── pdf.py
│   ├── text.py
│   └── web.py
├── processing/              # Text splitting & chunking logic
│   └── chunker.py
├── vectorstores/            # FAISS and Chroma vector store handlers
│   ├── chroma.py
│   └── faiss.py
├── retrieval/               # Search algorithms and score retrieval
│   └── search.py
├── experiments/             # Vector store & chunk size benchmarking
│   └── comparison.py
├── storage/                 # Local persistent vector database storage (git-ignored)
│   ├── chroma/
│   └── faiss/
└── tests/                   # Automated unit tests
```

---

## 📤 Pushing to GitHub

To upload this repository to GitHub for the first time:

1. **Initialize Git (if not already initialized)**:
   ```bash
   git init
   ```

2. **Add files and create initial commit**:
   ```bash
   git add .
   git commit -m "Initial commit: Personal Knowledge Base Search application"
   ```

3. **Link to your GitHub repository and push**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   git push -u origin main
   ```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
