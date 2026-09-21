"""Personal Knowledge Base Search — Streamlit app.

Flow (per architecture.md):
Streamlit -> Ingestion -> Chunking -> Embeddings -> FAISS + Chroma -> Search -> Streamlit
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

import config
from embeddings.provider import get_embeddings
from experiments.comparison import compare_chunk_sizes, compare_faiss_vs_chroma
from ingestion.pdf import load_pdf
from ingestion.text import load_text
from ingestion.web import load_web
from processing.chunker import chunk_documents
from retrieval.search import search_with_scores
from vectorstores.chroma import chroma_has_data, load_chroma
from vectorstores.faiss import build_faiss, faiss_index_exists, load_faiss

st.set_page_config(page_title="Personal Knowledge Base Search", layout="wide")


@st.cache_resource
def get_cached_embeddings():
    """Reuse one Ollama embeddings client across reruns instead of recreating it."""
    return get_embeddings()


def load_uploaded_file(uploaded_file) -> list:
    """Write a Streamlit upload to a temp file so LangChain's loaders (which
    read from disk) can process it, then load it with the right loader."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            docs = load_pdf(tmp_path)
        else:
            docs = load_text(tmp_path)
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        return docs
    finally:
        os.remove(tmp_path)


def index_documents(raw_documents: list, embeddings) -> int:
    """Chunk new documents and add them to both persisted vector stores.

    Chroma's PersistentClient writes to disk on every add automatically.
    FAISS is an in-memory index, so it needs an explicit save_local() call
    after every change to stay in sync with disk.
    """
    chunks = chunk_documents(raw_documents)

    chroma_store = load_chroma(embeddings)
    chroma_store.add_documents(chunks)

    if faiss_index_exists():
        faiss_store = load_faiss(embeddings)
        faiss_store.add_documents(chunks)
    else:
        faiss_store = build_faiss(chunks, embeddings)
    faiss_store.save_local(config.FAISS_DIR)

    st.session_state.raw_documents.extend(raw_documents)
    return len(chunks)


if "raw_documents" not in st.session_state:
    st.session_state.raw_documents = []

st.title("Personal Knowledge Base Search")
tab_kb, tab_search, tab_experiments = st.tabs(["Knowledge Base", "Search", "Experiments"])

embeddings = get_cached_embeddings()

# ---------------------------------------------------------------- Knowledge Base
with tab_kb:
    st.header("Add to your knowledge base")
    st.caption(
        "Files and pages are split into chunks, embedded locally with Ollama "
        "(nomic-embed-text), and stored in both FAISS and Chroma."
    )

    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files", type=["txt", "pdf"], accept_multiple_files=True
    )
    if st.button("Index uploaded files", disabled=not uploaded_files):
        with st.spinner("Loading, chunking, and embedding..."):
            raw_documents = []
            for uploaded_file in uploaded_files:
                raw_documents.extend(load_uploaded_file(uploaded_file))
            num_chunks = index_documents(raw_documents, embeddings)
        st.success(f"Indexed {len(uploaded_files)} file(s) into {num_chunks} chunks.")

    st.divider()

    url = st.text_input("Add a web page by URL")
    if st.button("Index URL", disabled=not url):
        with st.spinner("Fetching, chunking, and embedding..."):
            raw_documents = load_web(url)
            num_chunks = index_documents(raw_documents, embeddings)
        st.success(f"Indexed {url} into {num_chunks} chunks.")

# ---------------------------------------------------------------- Search
with tab_search:
    st.header("Search your knowledge base")

    has_data = faiss_index_exists() or chroma_has_data(embeddings)
    if not has_data:
        st.info("Add some documents in the Knowledge Base tab first.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Query", key="search_query")
        with col2:
            store_choice = st.selectbox("Vector store", ["FAISS", "Chroma"])
        top_k = st.slider("Top-K results", min_value=1, max_value=10, value=config.DEFAULT_TOP_K)

        if st.button("Search", disabled=not query):
            store = load_faiss(embeddings) if store_choice == "FAISS" else load_chroma(embeddings)
            results = search_with_scores(store, query, k=top_k)

            if not results:
                st.warning("No results found.")
            for doc, score in results:
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page")
                label = f"{source}" + (f" (page {page})" if page is not None else "")
                with st.expander(f"{label} — score {score:.4f}"):
                    st.write(doc.page_content)

# ---------------------------------------------------------------- Experiments
with tab_experiments:
    st.header("Compare retrieval configurations")
    st.caption("Runs against everything you've indexed in the Knowledge Base tab this session.")

    if not st.session_state.raw_documents:
        st.info("Add some documents in the Knowledge Base tab first.")
    else:
        exp_query = st.text_input("Query", key="experiment_query")

        st.subheader("FAISS vs Chroma")
        if st.button("Compare vector stores", disabled=not exp_query):
            with st.spinner("Building both stores and searching..."):
                chunks = chunk_documents(st.session_state.raw_documents)
                results = compare_faiss_vs_chroma(chunks, embeddings, exp_query, k=config.DEFAULT_TOP_K)
            cols = st.columns(len(results))
            for col, (name, r) in zip(cols, results.items()):
                with col:
                    st.metric(f"{name} build time", f"{r['build_seconds']:.3f}s")
                    st.metric(f"{name} search time", f"{r['search_seconds']:.3f}s")
                    top_doc, top_score = r["hits"][0]
                    st.write(f"Top result (score {top_score:.4f}):")
                    st.caption(top_doc.page_content[:300])

        st.subheader("Chunk size comparison (FAISS)")
        chunk_sizes = st.multiselect(
            "Chunk sizes to compare", [250, 500, 1000, 2000], default=[500, 1000]
        )
        if st.button("Compare chunk sizes", disabled=not (exp_query and chunk_sizes)):
            with st.spinner("Chunking, embedding, and searching at each size..."):
                results = compare_chunk_sizes(
                    st.session_state.raw_documents, embeddings, exp_query, chunk_sizes=chunk_sizes
                )
            cols = st.columns(len(results))
            for col, (size, r) in zip(cols, results.items()):
                with col:
                    st.metric(f"chunk_size={size}", f"{r['num_chunks']} chunks")
                    top_doc, top_score = r["hits"][0]
                    st.write(f"Top result (score {top_score:.4f}):")
                    st.caption(top_doc.page_content[:300])
