# Requirements Document

## Introduction

This document specifies requirements for a production-grade **Hybrid RAG Assistant with Cross-Encoder Reranking and Ragas Evaluation** — a modular Python system that ingests documents, retrieves relevant context via hybrid dense+sparse search, reranks candidates using a cross-encoder, generates answers via a configurable LLM backend, and evaluates pipeline quality using the Ragas framework. The system exposes a FastAPI REST interface and a Streamlit UI, is containerized with Docker Compose, and ships with a full test suite and MkDocs documentation.

---

## Glossary

- **Pipeline**: The end-to-end sequence — ingest → retrieve → rerank → generate → (optionally) evaluate.
- **Hybrid_Retriever**: The component that queries Qdrant using both dense vector search and sparse BM25 keyword search, then merges results.
- **Dense_Embedder**: The sub-component of Hybrid_Retriever that encodes text into dense vectors using a sentence-transformers model.
- **Sparse_Encoder**: The sub-component of Hybrid_Retriever that converts text into BM25 sparse term-weight vectors.
- **Reranker**: The component that applies a cross-encoder model to score and reorder a candidate list of document chunks.
- **Generator**: The component that calls an LLM backend (OpenAI / Ollama / local model via LiteLLM) to produce a final answer from a query and retrieved context.
- **Evaluator**: The component that runs Ragas metrics (Faithfulness, Answer Relevance, Context Precision) over a QA dataset.
- **API**: The FastAPI application layer that exposes REST endpoints for query, ingest, and evaluation operations.
- **UI**: The Streamlit application layer providing an interactive visual demo of the pipeline.
- **Config**: The Pydantic-Settings / YAML configuration loader that resolves settings from `.env` files and YAML configs.
- **Document**: A unit of text (e.g., a paragraph or file section) that is ingested, chunked, embedded, and stored in Qdrant.
- **Chunk**: A fixed-size or semantically bounded segment of a Document used as retrieval unit.
- **Candidate**: A retrieved Chunk prior to reranking.
- **Context**: The ordered list of top-K Chunks selected after reranking, passed to the Generator.
- **QA_Dataset**: A structured dataset of question–answer pairs and expected contexts used by the Evaluator.
- **Qdrant**: The vector database used for storing and searching Chunks.
- **Collection**: A Qdrant collection that holds all indexed Chunks for a specific domain or session.

---

## Requirements

---

### Requirement 1: Configuration Management

**User Story:** As a developer, I want all system parameters (model names, API keys, Qdrant URL, chunk size, top-K values, score thresholds) to be managed through a single hierarchical configuration system, so that I can change behaviour across environments without modifying source code.

#### Acceptance Criteria

1. THE Config SHALL load settings from a `.env` file and from YAML configuration files, with YAML values overriding `.env` defaults.
2. WHEN a required configuration key is absent at startup, THE Config SHALL raise a descriptive `ConfigurationError` identifying the missing key by name.
3. THE Config SHALL expose all settings as a typed Pydantic v2 `BaseSettings` model with strict type validation.
4. WHERE a `YAML_CONFIG_PATH` environment variable is set, THE Config SHALL additionally load and merge settings from the YAML file at that path.
5. THE Config SHALL provide separate sub-models for `QdrantSettings`, `EmbedderSettings`, `RerankerSettings`, `LLMSettings`, `RetrievalSettings`, and `EvaluationSettings`.

---

### Requirement 2: Document Ingestion

**User Story:** As a developer, I want to ingest raw text documents into the system, so that they are chunked, embedded, and indexed in Qdrant for later retrieval.

#### Acceptance Criteria

1. WHEN a list of `Document` objects is submitted for ingestion, THE Pipeline SHALL split each Document into Chunks of configurable `chunk_size` tokens with configurable `chunk_overlap`.
2. THE Dense_Embedder SHALL encode each Chunk into a dense vector of the dimensionality specified by the configured sentence-transformers model.
3. THE Sparse_Encoder SHALL encode each Chunk into a BM25 sparse vector representing term weights over the corpus vocabulary.
4. WHEN ingestion completes successfully, THE Pipeline SHALL store each Chunk's dense vector, sparse vector, and metadata in the configured Qdrant Collection.
5. IF the Qdrant Collection does not exist at ingestion time, THEN THE Pipeline SHALL create the Collection with the correct vector configuration before inserting Chunks.
6. WHEN a batch of Chunks exceeds the configured `ingestion_batch_size`, THE Pipeline SHALL upload Chunks in sequential batches without dropping any Chunk.
7. IF the Qdrant service is unreachable during ingestion, THEN THE Pipeline SHALL complete all splitting and encoding steps and raise an `IngestionError` only at the storage step, with a message describing the connection failure, without persisting any Chunks through an alternative mechanism.

---

### Requirement 3: Hybrid Retrieval

**User Story:** As a developer, I want the system to retrieve document chunks using both semantic dense search and keyword-based sparse search, so that I get better recall across diverse query types.

#### Acceptance Criteria

1. WHEN a query string is received, THE Hybrid_Retriever SHALL encode the query using both the Dense_Embedder and the Sparse_Encoder in parallel.
2. THE Hybrid_Retriever SHALL query the Qdrant Collection using dense vector search with configurable `dense_top_k` and sparse vector search with configurable `sparse_top_k`.
3. THE Hybrid_Retriever SHALL merge the dense and sparse result sets using Reciprocal Rank Fusion (RRF) with a configurable fusion constant `rrf_k`.
4. WHEN merging results, THE Hybrid_Retriever SHALL deduplicate Candidates by Chunk identifier so that each Chunk appears at most once in the merged list.
5. THE Hybrid_Retriever SHALL return the top `retrieval_top_k` Candidates sorted by descending RRF score.
6. IF the merged result set contains fewer Candidates than `retrieval_top_k`, THEN THE Hybrid_Retriever SHALL return all available Candidates without padding.
7. FOR ALL query strings of equal semantic content but differing surface form (e.g., synonyms), THE Hybrid_Retriever SHALL return at least one overlapping Candidate in the top-`retrieval_top_k` results (semantic consistency property).

---

### Requirement 4: Cross-Encoder Reranking

**User Story:** As a developer, I want the top retrieved candidates to be reranked using a cross-encoder model, so that the most relevant chunks are placed first before being passed to the LLM.

#### Acceptance Criteria

1. WHEN a query and a list of Candidates are provided, THE Reranker SHALL score each (query, Candidate text) pair using the configured cross-encoder model.
2. THE Reranker SHALL return Candidates sorted by descending cross-encoder score.
3. THE Reranker SHALL truncate the output to the configured `rerank_top_k` Candidates.
4. IF `rerank_top_k` exceeds the number of input Candidates, THEN THE Reranker SHALL return all input Candidates ranked by cross-encoder score without error, regardless of how low those scores are.
5. THE Reranker SHALL process all (query, candidate) pairs in a single batched forward pass to minimise latency.
6. FOR ALL input candidate lists, the output of THE Reranker SHALL contain only Candidates that were present in the input list (no hallucinated chunks).
7. FOR ALL permutations of the same input candidate list, THE Reranker SHALL produce an identical output order (determinism property, given fixed model weights).

---

### Requirement 5: LLM Answer Generation

**User Story:** As a developer, I want the system to generate a natural language answer from a query and its reranked context, so that end users receive a coherent, grounded response.

#### Acceptance Criteria

1. WHEN a query string and a Context (ordered list of Chunk texts) are provided, THE Generator SHALL construct a prompt that includes all Context texts and the query.
2. THE Generator SHALL call the configured LLM backend (OpenAI / Ollama / local model via LiteLLM) with the constructed prompt.
3. THE Generator SHALL return a structured `GenerationResult` object containing the answer text, the model identifier used, and the token usage statistics.
4. WHERE `stream_response` is set to `true` in `LLMSettings`, THE Generator SHALL yield answer tokens incrementally as an async generator; IF the backend returns an error at any point during streaming, THEN THE Generator SHALL immediately stop yielding tokens, discard any buffered partial output, and raise a `GenerationError` — ensuring no partial token sequences are surfaced to the caller without a corresponding error signal.
5. IF the LLM backend returns an error response, THEN THE Generator SHALL raise a `GenerationError` with the upstream error message and HTTP status code.
6. THE Generator SHALL enforce a configurable `max_context_tokens` limit by truncating the Context to fit within that limit before constructing the prompt.
7. THE Generator SHALL support at least three LLM backends — `openai`, `ollama`, and `litellm` — selectable via the `LLMSettings.provider` configuration key.

---

### Requirement 6: End-to-End Query Pipeline

**User Story:** As a user, I want to submit a natural language question and receive an answer grounded in the indexed documents, so that I can query my knowledge base conversationally.

#### Acceptance Criteria

1. WHEN a query string is submitted to the Pipeline, THE Pipeline SHALL execute the steps Hybrid_Retriever → Reranker → Generator in sequential order.
2. THE Pipeline SHALL return a `PipelineResult` object containing the answer text, the Context chunks used, the reranked scores, and end-to-end latency in milliseconds.
3. IF any pipeline step raises an error, THEN THE Pipeline SHALL propagate a `PipelineError` wrapping the original exception and identifying the failing step by name.
4. THE Pipeline SHALL be fully async (using `asyncio`) so that concurrent requests do not block each other.
5. WHILE a query request is actively being processed by the Pipeline (from the moment Hybrid_Retriever begins until PipelineResult is returned), THE Pipeline SHALL not issue any write operations to the Qdrant Collection; write operations (e.g., ingestion) submitted during a query's execution SHALL be queued or rejected, but SHALL be permitted when no query is in flight.

---

### Requirement 7: Ragas Evaluation

**User Story:** As an ML engineer, I want to run Ragas evaluation metrics over a QA dataset, so that I can measure and track the quality of the RAG pipeline.

#### Acceptance Criteria

1. WHEN a QA_Dataset is provided, THE Evaluator SHALL run the pipeline for each question and collect the generated answer and Context.
2. THE Evaluator SHALL compute Faithfulness, Answer Relevance, and Context Precision scores for each question–answer pair using the `ragas` library.
3. THE Evaluator SHALL return an `EvaluationReport` object containing per-question scores and aggregate mean scores for each metric.
4. THE Evaluator SHALL persist the `EvaluationReport` as a JSON file to a configurable `eval_output_dir` path.
5. IF a question in the QA_Dataset produces a pipeline error, THEN THE Evaluator SHALL record the error in the `EvaluationReport` for that question and continue evaluating the remaining questions.
6. WHERE `eval_sample_size` is configured, THE Evaluator SHALL evaluate only a random sample of that size from the QA_Dataset, using a configurable random seed for reproducibility.
7. FOR ALL QA_Datasets evaluated with the same random seed, THE Evaluator SHALL select an identical sample (reproducibility property).

---

### Requirement 8: FastAPI REST Interface

**User Story:** As a developer, I want to interact with the pipeline through a REST API, so that I can integrate the assistant into external applications.

#### Acceptance Criteria

1. THE API SHALL expose a `POST /query` endpoint that accepts a JSON body with `query` (string) and optional `top_k` (integer) fields and returns a `PipelineResult` JSON response.
2. THE API SHALL expose a `POST /ingest` endpoint that accepts a JSON body with a list of document objects and returns an ingestion status response.
3. THE API SHALL expose a `POST /evaluate` endpoint that accepts a JSON body with a QA dataset path or inline records and returns an `EvaluationReport` JSON response.
4. THE API SHALL expose a `GET /health` endpoint that returns HTTP 200 with a JSON body `{"status": "ok"}` when all dependencies (Qdrant, LLM backend) are reachable and no active internal errors are present.
5. WHEN a request body fails Pydantic validation, THE API SHALL return HTTP 422 with a JSON error body describing all validation errors.
6. IF an internal pipeline error occurs during a request, THEN THE API SHALL return HTTP 500 with a JSON body containing the error message and the failing step name.
7. THE API SHALL include OpenAPI documentation auto-generated by FastAPI, accessible at `/docs`; IF FastAPI fails to generate the OpenAPI schema at startup, THEN THE API SHALL log a warning and continue serving all REST endpoints — the availability of `/docs` SHALL NOT be a prerequisite for a successful startup or for the `GET /health` endpoint to return HTTP 200.
8. WHEN the `POST /query` endpoint is called with `stream=true` in the request body, THE API SHALL return a streaming `text/event-stream` response with incremental answer tokens.

---

### Requirement 9: Streamlit UI

**User Story:** As a non-technical user, I want a visual interface to query the assistant and view retrieved context, so that I can explore the system without using the API directly.

#### Acceptance Criteria

1. THE UI SHALL provide a text input field for entering a natural language query and a submit button to trigger the pipeline.
2. WHEN a query is submitted via THE UI, THE UI SHALL display the generated answer, the list of Context chunks with their source metadata, and the per-chunk reranker scores.
3. THE UI SHALL provide a sidebar panel to configure `top_k`, `rerank_top_k`, and the LLM provider without restarting the application.
4. WHEN the pipeline is processing a query or performing any time-consuming operation (file upload, configuration change), THE UI SHALL display a loading indicator or disable the submit button, with at least one of these feedback mechanisms active at all times during the operation.
5. IF the pipeline returns an error, THEN THE UI SHALL display the error message in a visible error banner rather than silently failing.
6. THE UI SHALL display a collapsible evaluation panel where a user can upload a QA dataset file and trigger an evaluation run, viewing the resulting metric scores.

---

### Requirement 10: Testing

**User Story:** As a developer, I want a comprehensive test suite, so that I can verify correctness of all modules and detect regressions.

#### Acceptance Criteria

1. THE test suite SHALL include unit tests for Hybrid_Retriever, Reranker, Generator, Evaluator, and Config modules using `pytest`.
2. THE test suite SHALL include integration tests for the `POST /query`, `POST /ingest`, and `GET /health` API endpoints using `pytest` with a test FastAPI client.
3. WHEN running unit tests, THE test suite SHALL use mock objects for Qdrant, LLM backends, and cross-encoder models so that external services are not required.
4. THE test suite SHALL achieve a minimum of 80% line coverage across the `src/` directory as measured by `pytest-cov`.
5. THE test suite SHALL include at least one property-based test using `hypothesis` verifying that the RRF merging function preserves all unique Chunk identifiers from both input lists (no data loss property).
6. THE test suite SHALL use `pytest-asyncio` for all async test cases.
7. WHEN the full test suite is executed via `pytest`, THE test suite SHALL complete without requiring any live external service (Qdrant, OpenAI, HuggingFace) by using fixtures and mocks.

---

### Requirement 11: Containerisation and Deployment

**User Story:** As a DevOps engineer, I want the full system to be runnable with a single `docker compose up` command, so that I can deploy and reproduce the environment without manual setup.

#### Acceptance Criteria

1. THE repository SHALL include a `docker-compose.yml` that defines services for the FastAPI API, the Streamlit UI, and the Qdrant vector database.
2. THE Dockerfile for the API service SHALL use a multi-stage build: a `builder` stage that installs dependencies and a `runtime` stage that copies only the necessary artifacts.
3. WHEN `docker compose up` is executed, THE API service SHALL become healthy (respond to `GET /health` with HTTP 200) within 60 seconds of startup under normal conditions; WHERE startup takes longer than 60 seconds due to resource contention, THE API service SHALL be considered successfully deployed once it does become healthy.
4. THE `docker-compose.yml` SHALL configure Qdrant with a persistent named volume so that indexed data survives container restarts.
5. THE `docker-compose.yml` SHALL define an `env_file` directive pointing to `.env` so that secrets are never hardcoded in the compose file.
6. IF the Qdrant service has not yet started when the API service starts, THEN THE API service SHALL retry the Qdrant connection up to 5 times with exponential back-off before raising a startup error.

---

### Requirement 12: Documentation

**User Story:** As a developer, I want clear, browsable documentation, so that I can understand the architecture, configuration, and usage of the system without reading source code.

#### Acceptance Criteria

1. THE repository SHALL include a `README.md` with quickstart instructions covering environment setup, configuration, running via Docker Compose, and running the test suite.
2. THE repository SHALL include a `docs/` directory configured for `mkdocs` with at least the following pages: Architecture Overview, Configuration Reference, API Reference, Evaluation Guide, and Contributing Guide.
3. THE Config module SHALL be documented with all supported configuration keys, their types, default values, and descriptions in the Configuration Reference page.
4. ALL public classes and functions in `src/` SHALL have Google-style docstrings in English.
5. THE repository SHALL include a `pyproject.toml` that declares all runtime and development dependencies with pinned minimum versions, project metadata, and `pytest` / `mypy` / `ruff` tool configurations.
