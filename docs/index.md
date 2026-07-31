# Hybrid RAG Assistant Documentation

Welcome to the documentation for the **Hybrid Retrieval-Augmented Generation (RAG) Assistant**!

This project provides a robust, production-grade implementation of a modular RAG pipeline featuring:
- Concurrent dual-vector retrieval (Dense Embeddings + BM25 Sparse Vectors).
- Reciprocal Rank Fusion (RRF) for merging search results.
- Cross-Encoder reranking for context prioritization.
- Budget-aware context sliding-window token truncation.
- Automated pipeline evaluation using the Ragas framework.
- Modern FastAPI REST API and Streamlit UI.
- Clean containerization with Docker Compose.

---

## Contents
- **[Architecture](architecture.md)**: Deep dive into the pipeline design and execution flow.
- **[Configuration](configuration.md)**: Detailed settings parameters.
- **[API Reference](api.md)**: FastAPI endpoints overview.
