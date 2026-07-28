# Implementation Plan: Hybrid RAG Evaluation Pipeline

## Overview

Build a production-grade Hybrid RAG Assistant in Python using a strict bottom-up order:
project skeleton → config → data models → ingestion → retrieval sub-components → hybrid
retriever → reranking → generation → pipeline orchestrator → evaluation → FastAPI API →
Streamlit UI → tests → Docker/docs. Each task wires its output directly into the next,
leaving no orphaned code.

---

## Tasks

- [x] 1. Project scaffold and tooling
  - Create the directory tree from the design (`src/`, `tests/unit/`, `tests/property/`,
    `tests/integration/`, `ui/`, `docs/`)
  - Write `pyproject.toml` with all runtime and dev dependencies (pinned minimum
    versions), `[tool.pytest.ini_options]`, `[tool.mypy]`, and `[tool.ruff]` sections
  - Add a minimal `config.yaml` and `.env.example` showing every configuration key
  - Create `__init__.py` files in all `src/` sub-packages
  - _Requirements: 12.5_

- [x] 2. Typed exception hierarchy
  - [x] 2.1 Implement `src/exceptions.py` with `HybridRAGError` base and all sub-types
    - Define `HybridRAGError`, `ConfigurationError`, `IngestionError`, `RetrievalError`,
      `RerankerError`, `GenerationError`, `StreamingError`, `PipelineError`
      (with `step_name` and `cause` fields), and `EvaluationError`
    - _Requirements: 1.2, 2.7, 3.1, 4.1, 5.5, 6.3_

- [x] 3. Configuration system
  - [x] 3.1 Implement `src/config/settings.py`
    - Write all Pydantic v2 sub-models: `QdrantSettings`, `EmbedderSettings`,
      `SparseEncoderSettings`, `RerankerSettings`, `LLMSettings`, `RetrievalSettings`,
      `IngestionSettings`, `EvaluationSettings`
    - Write `AppSettings(BaseSettings)` with `settings_customise_sources` implementing
      the env → YAML → `.env` priority chain and the `YamlConfigSettingsSource`
    - Raise `ConfigurationError` (wrapping `ValidationError`) on missing required keys
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 3.2 Write property tests for configuration (P1, P2)
    - **Property 1: YAML configuration overrides .env defaults**
    - **Validates: Requirements 1.1**
    - **Property 2: Missing required key produces named ConfigurationError**
    - **Validates: Requirements 1.2**
    - File: `tests/property/test_config_props.py`
    - Strategies: `st.text()`, `st.integers()`, `st.floats()` for config values;
      `st.sampled_from(required_keys)` for missing-key cases

  - [ ]* 3.3 Write unit tests for `AppSettings`
    - Test YAML override, missing key, wrong type, sub-model instantiation
    - File: `tests/unit/test_config.py`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 4. Core data models
  - [x] 4.1 Implement `src/models.py` with all shared DTOs
    - Define `Document`, `Chunk`, `ScoredChunk`, `QAPair` as Pydantic v2 `BaseModel`
    - Define `GenerationResult`, `PipelineResult`, `IngestionResult`,
      `EvaluationReport`, `QuestionResult`, `QuestionError` as Python `dataclass`
    - _Requirements: 2.1, 3.5, 5.3, 6.2, 7.3_

- [x] 5. Dense embedder
  - [x] 5.1 Implement `src/retrieval/dense_embedder.py`
    - Wrap `SentenceTransformer` with sync `encode()` and async `aencode()` (thread-pool)
    - Validate post-encoding that vector length == `EmbedderSettings.vector_dim`
    - _Requirements: 2.2_

  - [ ]* 5.2 Write property test for dense embedding dimension (P4)
    - **Property 4: Dense embedding dimension matches configured model**
    - **Validates: Requirements 2.2**
    - File: `tests/property/test_embedder_props.py`
    - Strategy: `st.text(min_size=1)` for chunk texts; mock `SentenceTransformer`

  - [ ]* 5.3 Write unit tests for `DenseEmbedder`
    - Test sync and async encode, dimension validation, mock model
    - File: `tests/unit/test_dense_embedder.py`
    - _Requirements: 2.2_

- [ ] 6. Sparse encoder
  - [x] 6.1 Implement `src/retrieval/sparse_encoder.py`
    - Wrap `fastembed` BM25 model; expose sync `encode()` and async `aencode()`
    - Return `SparseVector(indices, values)` per text; use thread-pool for async
    - _Requirements: 2.3_

  - [ ]* 6.2 Write property test for sparse encoding (P5)
    - **Property 5: Sparse encoding produces non-empty term-weight vectors**
    - **Validates: Requirements 2.3**
    - File: `tests/property/test_sparse_props.py`
    - Strategy: `st.text(min_size=1)` for chunk texts; mock `fastembed` model

  - [ ]* 6.3 Write unit tests for `SparseEncoder`
    - Test output structure, non-empty weights, async path
    - File: `tests/unit/test_sparse_encoder.py`
    - _Requirements: 2.3_

- [x] 7. RRF merger
  - [x] 7.1 Implement `src/retrieval/rrf.py` — pure function `rrf_merge()`
    - Implement rank assignment, per-chunk RRF score accumulation, deduplication,
      descending sort, and top-`top_k` truncation
    - _Requirements: 3.3, 3.4, 3.5, 10.5_

  - [ ]* 7.2 Write property tests for RRF merge correctness (P7)
    - **Property 7: RRF merge — no data loss, correct scores, deduplication, sorted output**
    - **Validates: Requirements 3.3, 3.4, 3.5, 10.5**
    - File: `tests/property/test_rrf_props.py`
    - Strategy: `st.lists(chunk_id_strategy)` for dense/sparse lists; check all four sub-properties

  - [ ]* 7.3 Write unit tests for `rrf_merge`
    - Test empty inputs, fully overlapping lists, single-list input, top_k clipping
    - File: `tests/unit/test_rrf.py`
    - _Requirements: 3.3, 3.4, 3.5_

- [x] 8. Checkpoint — core retrieval components
  - Ensure all tests in `tests/unit/` and `tests/property/` written so far pass.
  - Run `pytest tests/unit tests/property --tb=short`; ask the user if any failures arise.

- [x] 9. Ingestion service
  - [x] 9.1 Implement `src/ingestion/ingestor.py`
    - Implement sliding-window token chunking (`chunk_size`, `chunk_overlap`)
    - Run `DenseEmbedder.aencode()` and `SparseEncoder.aencode()` in parallel via
      `asyncio.gather`
    - Create Qdrant collection (dual named vectors: `"dense"` + `"sparse"`) if absent
    - Upload in batches of `ingestion_batch_size`; raise `IngestionError` on Qdrant
      write failure (after all encoding is complete)
    - Return `IngestionResult`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 9.2 Write property tests for ingestion (P3, P6)
    - **Property 3: Chunking produces correctly sized and overlapping chunks**
    - **Validates: Requirements 2.1**
    - **Property 6: Ingestion stores all chunks without data loss**
    - **Validates: Requirements 2.4, 2.6**
    - File: `tests/property/test_ingestion_props.py`
    - Strategies: `st.text(min_size=1)`, `st.integers(50, 512)` for chunk_size;
      `st.lists(chunk_strategy, min_size=1)`, `st.integers(1, 50)` for batch_size

  - [ ]* 9.3 Write unit tests for `IngestionService`
    - Test chunk splitting, batch upload, collection creation, `IngestionError` on
      Qdrant failure; mock Qdrant client
    - File: `tests/unit/test_ingestor.py`
    - _Requirements: 2.1, 2.5, 2.6, 2.7_

- [ ] 10. Hybrid retriever
  - [ ] 10.1 Implement `src/retrieval/hybrid_retriever.py`
    - Encode query with both encoders concurrently via `asyncio.gather`
    - Issue two Qdrant searches (dense, sparse); inject Qdrant client via constructor
    - Apply `rrf_merge()` and return top-`retrieval_top_k` `ScoredChunk` objects sorted
      by descending RRF score; raise `RetrievalError` on Qdrant failure
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 10.2 Write unit tests for `HybridRetriever`
    - Verify both encoders called, Qdrant queried twice, output is sorted, empty results
      handled, `RetrievalError` raised on Qdrant failure
    - File: `tests/unit/test_hybrid_retriever.py`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 11. Cross-encoder reranker
  - [~] 11.1 Implement `src/reranking/reranker.py`
    - Build `(query, text)` pairs; call `CrossEncoder.predict()` in a single batched
      forward pass (thread-pool executor)
    - Sort candidates by descending score; truncate to `rerank_top_k`
    - Handle `rerank_top_k > len(candidates)` gracefully; raise `RerankerError` on
      model failure
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 11.2 Write property tests for reranker invariants (P8)
    - **Property 8: Reranker output is a sorted, truncated subset of input**
    - **Validates: Requirements 4.2, 4.3, 4.6, 4.7**
    - File: `tests/property/test_reranker_props.py`
    - Strategies: `st.lists(chunk_strategy, min_size=1)`, `st.integers(1, 20)` for top_k

  - [ ]* 11.3 Write unit tests for `CrossEncoderReranker`
    - Test single batched call, truncation, top_k > input length, mock `CrossEncoder`
    - File: `tests/unit/test_reranker.py`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 12. LLM generator
  - [~] 12.1 Implement `src/generation/generator.py` with strategy-pattern backends
    - Define `LLMBackend` ABC; implement `OpenAIBackend`, `OllamaBackend`,
      `LiteLLMBackend`
    - Implement context truncation with `tiktoken` before prompt construction
    - Build prompt template (system + context blocks + user query)
    - Implement `generate()` returning `GenerationResult` and `stream()` returning
      async generator; raise `GenerationError` / `StreamingError` on backend errors
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 12.2 Write property tests for generator (P9, P10, P11, P12)
    - **Property 9: Prompt contains all context texts and the query**
    - **Validates: Requirements 5.1**
    - **Property 10: GenerationResult always contains answer, model ID, and token usage**
    - **Validates: Requirements 5.3**
    - **Property 11: Any LLM backend error raises GenerationError**
    - **Validates: Requirements 5.5**
    - **Property 12: Context truncation respects max_context_tokens**
    - **Validates: Requirements 5.6**
    - File: `tests/property/test_generator_props.py`
    - Strategies: `st.text()` for query, `st.lists(st.text(), min_size=1)` for context,
      `st.sampled_from([400,401,429,500,503])` for error codes,
      `st.integers(100, 4096)` for max_tokens

  - [ ]* 12.3 Write unit tests for `LLMGenerator`
    - Test all three providers (mock HTTP), streaming happy path, streaming error
      (partial discard), context truncation boundary
    - File: `tests/unit/test_generator.py`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [ ] 13. Pipeline orchestrator
  - [~] 13.1 Implement `src/pipeline/pipeline.py`
    - Wire `HybridRetriever → CrossEncoderReranker → LLMGenerator` in sequential
      async steps; measure end-to-end latency
    - Catch any step error and re-raise as `PipelineError(step_name=..., cause=...)`
    - Protect against concurrent writes during active queries with `asyncio.Lock`
    - Return `PipelineResult`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 13.2 Write property tests for pipeline (P13, P14)
    - **Property 13: PipelineResult always contains all required fields**
    - **Validates: Requirements 6.2**
    - **Property 14: Pipeline step errors are wrapped in PipelineError with step name**
    - **Validates: Requirements 6.3**
    - File: `tests/property/test_pipeline_props.py`
    - Strategies: `st.text()` for queries; `st.sampled_from(["retriever","reranker","generator"])`,
      `st.from_type(Exception)` for error injection

  - [ ]* 13.3 Write unit tests for `Pipeline`
    - Test execution order, step error wrapping for each step, `PipelineResult` fields,
      latency field populated
    - File: `tests/unit/test_pipeline.py`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [~] 14. Checkpoint — pipeline core complete
  - Run `pytest tests/unit tests/property --tb=short`; confirm all tests pass.
  - Ask the user if any design deviations have emerged before continuing.

- [ ] 15. Ragas evaluator
  - [~] 15.1 Implement `src/evaluation/evaluator.py`
    - Implement `evaluate()`: optional sampling with seeded `random`, run `Pipeline.query()`
      per question, build `ragas.Dataset`, call `ragas.evaluate()` with Faithfulness /
      AnswerRelevancy / ContextPrecision
    - Record per-question errors (continue on failure); build `EvaluationReport`; persist
      JSON to `eval_output_dir/report_{timestamp}.json`; raise `EvaluationError` on fatal
      ragas failure
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ]* 15.2 Write property tests for evaluator (P15, P16, P17)
    - **Property 15: Evaluator processes all questions and records errors without stopping**
    - **Validates: Requirements 7.3, 7.5**
    - **Property 16: EvaluationReport persists as lossless JSON round-trip**
    - **Validates: Requirements 7.4**
    - **Property 17: Evaluation sampling is reproducible across runs with the same seed**
    - **Validates: Requirements 7.6, 7.7**
    - File: `tests/property/test_evaluator_props.py`
    - Strategies: `st.lists(qa_pair_strategy)` with random failure indices; `st.integers()`
      for seed; random mock ragas outputs; `tmp_path` fixture for file I/O

  - [ ]* 15.3 Write unit tests for `RagasEvaluator`
    - Test sample-with-seed, per-question error recording, JSON persistence, metric
      aggregation; mock `ragas.evaluate()` and `Pipeline.query()`
    - File: `tests/unit/test_evaluator.py`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [ ] 16. FastAPI application
  - [~] 16.1 Implement `src/api/schemas.py` — Pydantic v2 request/response models
    - Define `QueryRequest`, `QueryResponse`, `IngestRequest`, `IngestResponse`,
      `EvaluateRequest`, `EvaluateResponse`, `HealthResponse`; all fields typed and
      documented
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [~] 16.2 Implement `src/api/app.py` — `create_app()` factory and global middleware
    - Register global exception handler for `PipelineError` → HTTP 500 with
      `{"error": ..., "step": ...}`
    - Instantiate pipeline components once; store in `app.state`
    - Implement Qdrant startup probe with 5-attempt exponential back-off
    - _Requirements: 6.3, 8.6, 11.6_

  - [~] 16.3 Implement `src/api/routers/health.py` — `GET /health`
    - Probe Qdrant and LLM backend connectivity; return 200 `{"status": "ok"}` or 503
    - _Requirements: 8.4_

  - [~] 16.4 Implement `src/api/routers/ingest.py` — `POST /ingest`
    - Validate request body; call `IngestionService.ingest()`; return `IngestResponse`
    - _Requirements: 8.2, 8.5_

  - [~] 16.5 Implement `src/api/routers/query.py` — `POST /query`
    - Handle standard JSON response and `stream=true` SSE path (`text/event-stream`)
    - Call `Pipeline.query()` or `Pipeline.stream()` accordingly
    - _Requirements: 8.1, 8.5, 8.8_

  - [~] 16.6 Implement `src/api/routers/evaluate.py` — `POST /evaluate`
    - Validate request body; call `RagasEvaluator.evaluate()`; return `EvaluateResponse`
    - _Requirements: 8.3, 8.5_

  - [ ]* 16.7 Write property tests for API contracts (P18, P19)
    - **Property 18: API returns HTTP 422 for any invalid request body**
    - **Validates: Requirements 8.5**
    - **Property 19: API returns HTTP 500 with step name for any PipelineError**
    - **Validates: Requirements 8.6**
    - File: `tests/property/test_api_props.py`
    - Strategies: `st.fixed_dictionaries` with wrong types; `st.sampled_from(step_names)`

  - [ ]* 16.8 Write integration tests for all API endpoints
    - Cover: `POST /query` (200, 422, 500, SSE stream), `POST /ingest` (200, 422),
      `POST /evaluate` (200, 422), `GET /health` (200 healthy, 503 Qdrant down)
    - Use `httpx.AsyncClient` with FastAPI `create_app()` and all mocked dependencies
    - File: `tests/integration/test_api.py`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.8, 10.2_

- [~] 17. Checkpoint — API complete
  - Run `pytest tests/ --tb=short --cov=src --cov-report=term-missing`
  - Confirm ≥ 80% line coverage across `src/`; ask the user about any gaps.

- [ ] 18. Streamlit UI
  - [~] 18.1 Implement `ui/app.py`
    - Build query input + submit button; call `POST /query` via `httpx`
    - Display answer, context chunks with metadata, per-chunk reranker scores
    - Sidebar for `top_k`, `rerank_top_k`, LLM provider (persisted in `st.session_state`)
    - Show `st.spinner()` during all API calls; display errors in `st.error()` banners
    - Add collapsible evaluation panel: file uploader → `POST /evaluate` → metric scores
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [ ] 19. Docker Compose and containerisation
  - [~] 19.1 Write `Dockerfile` (multi-stage: `builder` + `runtime`)
    - Install deps in `builder` stage; copy only runtime artifacts to `runtime` stage
    - _Requirements: 11.2_

  - [~] 19.2 Write `docker-compose.yml`
    - Define `api`, `ui`, and `qdrant` services
    - Configure Qdrant persistent named volume and `env_file: .env` for secrets
    - Add health-check on the `api` service (`GET /health`)
    - _Requirements: 11.1, 11.3, 11.4, 11.5_

- [ ] 20. Documentation
  - [~] 20.1 Write `README.md`
    - Cover environment setup, configuration, Docker Compose quickstart, test suite
      invocation, and project structure overview
    - _Requirements: 12.1_

  - [~] 20.2 Write MkDocs pages under `docs/`
    - Create `docs/architecture.md`, `docs/configuration.md` (all keys, types, defaults),
      `docs/api_reference.md`, `docs/evaluation_guide.md`, `docs/contributing.md`
    - Add `mkdocs.yml` at repo root with `nav` referencing all five pages
    - _Requirements: 12.2, 12.3_

  - [~] 20.3 Add Google-style docstrings to all public classes and functions in `src/`
    - Cover `DenseEmbedder`, `SparseEncoder`, `rrf_merge`, `HybridRetriever`,
      `CrossEncoderReranker`, `LLMGenerator` (all backends), `Pipeline`,
      `IngestionService`, `RagasEvaluator`, `AppSettings`, all router functions
    - _Requirements: 12.4_

- [~] 21. Final checkpoint — full suite
  - Run `pytest tests/ -x --tb=short --cov=src --cov-report=term-missing --asyncio-mode=auto`
  - All tests must pass; coverage must be ≥ 80%; ask the user if anything needs adjustment.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP build.
- Each task references specific requirements from `requirements.md` for traceability.
- Checkpoints (tasks 8, 14, 17, 21) are mandatory gates — do not skip them.
- Property tests validate universal correctness invariants using `hypothesis`
  (`max_examples=100`); each test file must include the tag comment
  `# Feature: hybrid-rag-eval-pipeline, Property N: <property_text>`.
- Unit tests mock all external I/O (Qdrant, LLM backends, sentence-transformers,
  CrossEncoder, ragas); no live services are required (`requirements: 10.3, 10.7`).
- All async tests must use `pytest-asyncio` (`asyncio-mode = auto` in `pyproject.toml`).
- The `create_app()` factory accepts injected mocked dependencies for integration tests.
- Docker Compose health-check timeout is 60 s per Requirement 11.3.

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1"] },
    { "id": 1, "tasks": ["3.1", "4.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "5.1", "6.1", "7.1"] },
    { "id": 3, "tasks": ["5.2", "5.3", "6.2", "6.3", "7.2", "7.3"] },
    { "id": 4, "tasks": ["9.1"] },
    { "id": 5, "tasks": ["9.2", "9.3", "10.1"] },
    { "id": 6, "tasks": ["10.2", "11.1"] },
    { "id": 7, "tasks": ["11.2", "11.3", "12.1"] },
    { "id": 8, "tasks": ["12.2", "12.3", "13.1"] },
    { "id": 9, "tasks": ["13.2", "13.3", "15.1"] },
    { "id": 10, "tasks": ["15.2", "15.3", "16.1"] },
    { "id": 11, "tasks": ["16.2"] },
    { "id": 12, "tasks": ["16.3", "16.4", "16.5", "16.6"] },
    { "id": 13, "tasks": ["16.7", "16.8", "18.1"] },
    { "id": 14, "tasks": ["19.1"] },
    { "id": 15, "tasks": ["19.2", "20.1", "20.2", "20.3"] }
  ]
}
```
