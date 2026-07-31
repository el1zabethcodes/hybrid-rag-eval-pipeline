# Architecture Overview

The Hybrid RAG Assistant is structured around modular, decoupling components connected sequentially by the `Pipeline` orchestrator.

## Components

### 1. Hybrid Retriever
The `HybridRetriever` is responsible for querying the vector database. When a query is received:
- A **Dense Embedder** encodes the query into a dense representation (e.g. 384 dimensions).
- A **Sparse Encoder** creates BM25 sparse vectors representing term frequencies.
- Both dense and sparse searches are issued concurrently against Qdrant.
- Raw candidates from both searches are unified using **Reciprocal Rank Fusion (RRF)**.

### 2. Cross-Encoder Reranker
The `CrossEncoderReranker` takes the merged candidate list and scores the relevance of each candidate chunk against the original query using a Cross-Encoder transformer model. The candidates are then sorted in descending order of relevance.

### 3. LLM Generator
The `LLMGenerator` wraps the query and reranked contexts into a prompt. To prevent context overflow, it measures the token count of each chunk using `tiktoken` and truncates chunks that exceed the configured token budget. The truncated prompt is dispatched to OpenAI, Ollama, or LiteLLM.

---

## Process Concurrency & Safety
During query processing, the orchestrator holds a thread-safe `asyncio.Lock`. This write lock prevents write operations (like document ingestion) from modifying vector indexes concurrently, ensuring consistency and preventing indices mutation during active user queries.
