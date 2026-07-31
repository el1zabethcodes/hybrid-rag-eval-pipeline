# Configuration System

The application configuration uses Pydantic Settings to resolve settings in the following priority order:
1. Environment variables
2. YAML configuration file
3. `.env` file variables
4. Base class model defaults

---

## Configuration Parameter Groups

### Qdrant Settings
- `QDRANT__URL`: Host url of the Qdrant service.
- `QDRANT__COLLECTION_NAME`: Target Qdrant collection name.

### Embedder Settings
- `EMBEDDER__MODEL_NAME`: HuggingFace model identifier for generating dense embeddings.
- `EMBEDDER__VECTOR_DIM`: Dimensionality of vectors.

### Reranker Settings
- `RERANKER__MODEL_NAME`: Cross-Encoder model identifier.
- `RERANKER__RERANK_TOP_K`: Candidate limit.

### LLM Settings
- `LLM__PROVIDER`: API backend, one of `openai`, `ollama`, `litellm`.
- `LLM__MODEL_NAME`: Target generative model.
- `LLM__MAX_CONTEXT_TOKENS`: Maximum context window tokens.
