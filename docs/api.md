# API Reference

The FastAPI server exposes endpoints for health checking, querying, document ingestion, and metrics evaluation.

---

## Endpoints

### 1. `GET /health`
Verifies that all downstream components are active.
- **Returns**: `HealthResponse` mapping `ok` status of Qdrant and LLM.

### 2. `POST /query`
Performs synchronous retrieval, reranking, and generation.
- **Payload**: `QueryRequest` containing `query` and `top_k`.
- **Returns**: `QueryResponse` with the final answer and structured context metadata.

### 3. `POST /query/stream`
Same workflow as `/query`, but returns streaming response.
- **Payload**: `QueryRequest`.
- **Returns**: Server-Sent Events (SSE) token strings.

### 4. `POST /ingest`
Ingests a list of documents. Acquires the write lock to block concurrent queries.
- **Payload**: `IngestRequest`.
- **Returns**: `IngestResponse` with collection name and chunks count.

### 5. `POST /evaluate`
Evaluates RAG pipeline outputs using Ragas.
- **Payload**: `EvaluateRequest` containing QA dataset.
- **Returns**: `EvaluateResponse` enclosing a generated JSON report.
