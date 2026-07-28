"""Application configuration for the Hybrid RAG Evaluation Pipeline.

This module implements a hierarchical Pydantic v2 settings system.

Resolution priority (highest → lowest):
    1. Shell environment variables
    2. YAML file (path given by ``YAML_CONFIG_PATH`` env var)
    3. ``.env`` file
    4. Model default values

All settings are strongly typed via Pydantic v2 sub-models.  A missing
required key raises :exc:`~src.exceptions.ConfigurationError` at
construction time so problems are surfaced before the application serves
any traffic.

Usage::

    from src.config import get_settings

    settings = get_settings()
    print(settings.qdrant.url)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from typing_extensions import Literal

from src.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Try to import yaml; provide a clear error if not installed.
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore[import]
except ImportError as _yaml_import_err:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = _yaml_import_err
else:
    _YAML_IMPORT_ERROR = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class QdrantSettings(BaseModel):
    """Connection and collection settings for the Qdrant vector database.

    Args:
        url: Full URL of the Qdrant service (default: ``http://localhost:6333``).
        collection_name: Name of the Qdrant collection used for document chunks.
        api_key: Optional API key for authenticated Qdrant Cloud deployments.
    """

    url: str = "http://localhost:6333"
    collection_name: str = "rag_chunks"
    api_key: Optional[str] = None


class EmbedderSettings(BaseModel):
    """Settings for the sentence-transformers dense embedding model.

    Args:
        model_name: HuggingFace model identifier for the embedding model.
        vector_dim: Dimensionality of the output embedding vectors.
        batch_size: Number of texts encoded in a single forward pass.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dim: int = 384
    batch_size: int = 64


class SparseEncoderSettings(BaseModel):
    """Settings for the BM25 sparse encoder.

    Args:
        model_name: Model identifier recognised by ``fastembed``.
            Defaults to ``"Qdrant/bm25"``.
    """

    model_name: str = "Qdrant/bm25"


class RerankerSettings(BaseModel):
    """Settings for the cross-encoder reranking model.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
        rerank_top_k: Maximum number of candidates to return after reranking.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5


class LLMSettings(BaseModel):
    """Settings for the LLM answer-generation backend.

    Args:
        provider: One of ``"openai"``, ``"ollama"``, or ``"litellm"``.
        model_name: Model identifier sent to the backend.
        max_context_tokens: Hard cap on context tokens included in each prompt.
        stream_response: Whether to yield tokens incrementally via streaming.
        api_key: Optional API key (required for the OpenAI provider).
        base_url: Optional override for the backend base URL (required for
            Ollama and LiteLLM providers).
    """

    provider: Literal["openai", "ollama", "litellm"] = "openai"
    model_name: str = "gpt-4o-mini"
    max_context_tokens: int = 4096
    stream_response: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class RetrievalSettings(BaseModel):
    """Settings controlling retrieval search and fusion parameters.

    Args:
        dense_top_k: Number of dense-search candidates fetched from Qdrant.
        sparse_top_k: Number of sparse-search candidates fetched from Qdrant.
        retrieval_top_k: Final number of candidates returned after RRF fusion.
        rrf_k: Fusion constant used in the Reciprocal Rank Fusion formula.
    """

    dense_top_k: int = 20
    sparse_top_k: int = 20
    retrieval_top_k: int = 10
    rrf_k: int = 60


class IngestionSettings(BaseModel):
    """Settings for the document ingestion pipeline.

    Args:
        chunk_size: Target size (in tokens) for each document chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        ingestion_batch_size: Maximum number of chunks per Qdrant upsert batch.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    ingestion_batch_size: int = 100


class EvaluationSettings(BaseModel):
    """Settings for the Ragas evaluation runner.

    Args:
        eval_output_dir: Directory path where evaluation report JSON files are
            persisted.
        eval_sample_size: Optional maximum number of QA pairs to evaluate.
            When ``None`` the full dataset is used.
        random_seed: Seed for the sampling RNG to ensure reproducible samples.
        judge_llm_provider: LLM provider used by Ragas as the evaluation judge.
            One of ``"openai"``, ``"ollama"``, or ``"litellm"``.
        judge_llm_model: Model identifier for the judge LLM.
        judge_llm_api_key: API key for the judge LLM backend.  When absent and
            ``judge_llm_provider`` is ``"openai"``, the value is read from the
            ``RAGAS_LLM_API_KEY`` environment variable.
        judge_llm_base_url: Base URL override for the judge LLM backend.
            Required for ``"ollama"`` and ``"litellm"`` providers.
    """

    eval_output_dir: str = "./eval_results"
    eval_sample_size: Optional[int] = None
    random_seed: int = 42
    judge_llm_provider: Literal["openai", "ollama", "litellm"] = "openai"
    judge_llm_model: str = "gpt-4o-mini"
    judge_llm_api_key: Optional[str] = None
    judge_llm_base_url: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML settings source
# ---------------------------------------------------------------------------


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """A ``pydantic-settings`` source that reads a YAML file.

    The YAML file may contain top-level keys corresponding to either the
    top-level ``AppSettings`` fields (e.g. ``qdrant``, ``llm``) or to their
    nested sub-keys using the same structure as the model.

    The source is added to the priority chain only when the ``YAML_CONFIG_PATH``
    environment variable is set and points to an existing file.

    Args:
        settings_cls: The ``AppSettings`` class being constructed.
        yaml_file: Path to the YAML configuration file.

    Raises:
        :exc:`~src.exceptions.ConfigurationError`: If ``pyyaml`` is not
            installed or the YAML file cannot be read.
    """

    def __init__(
        self,
        settings_cls: Type[BaseSettings],
        yaml_file: str | Path,
    ) -> None:
        super().__init__(settings_cls)
        self._yaml_file = Path(yaml_file)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load and parse the YAML file, returning a flat dict.

        Returns:
            Parsed YAML contents as a dictionary.

        Raises:
            :exc:`~src.exceptions.ConfigurationError`: On import or parse errors.
        """
        if _YAML_IMPORT_ERROR is not None:  # pragma: no cover
            raise ConfigurationError(
                "PyYAML is required to load YAML configuration files. "
                "Install it with: pip install pyyaml"
            ) from _YAML_IMPORT_ERROR

        if not self._yaml_file.exists():
            # A missing file is not fatal — return empty dict so the source
            # contributes nothing, letting lower-priority sources take effect.
            return {}

        try:
            with self._yaml_file.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # pragma: no cover
            raise ConfigurationError(
                f"Failed to read YAML configuration file '{self._yaml_file}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # PydanticBaseSettingsSource interface
    # ------------------------------------------------------------------

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        """Return the value for *field_name* from the YAML data, if present.

        Args:
            field: Pydantic ``FieldInfo`` descriptor for the field.
            field_name: The field name on ``AppSettings``.

        Returns:
            A ``(value, field_name, is_complex)`` tuple as required by
            ``PydanticBaseSettingsSource``.
        """
        field_value = self._data.get(field_name)
        return field_value, field_name, self.field_is_complex(field)

    def __call__(self) -> dict[str, Any]:
        """Return a dictionary of all values provided by this source.

        Returns:
            Mapping from field name to parsed YAML value for every top-level
            key found in the YAML file.
        """
        d: dict[str, Any] = {}
        for field_name in self.settings_cls.model_fields:
            val, key, is_complex = self.get_field_value(
                self.settings_cls.model_fields[field_name], field_name
            )
            if val is not None:
                d[key] = val
        return d


# ---------------------------------------------------------------------------
# Root settings model
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Root configuration model for the Hybrid RAG Evaluation Pipeline.

    All configuration values are resolved in the following priority order
    (highest wins):

    1. Shell / OS environment variables
    2. YAML file pointed to by ``YAML_CONFIG_PATH``
    3. ``.env`` file (loaded via ``pydantic-settings`` ``env_file`` support)
    4. Pydantic model default values

    Nested sub-models (e.g. ``qdrant``, ``llm``) can be overridden via
    environment variables using double-underscore nesting, e.g.::

        export QDRANT__URL=http://qdrant-prod:6333
        export LLM__API_KEY=sk-...

    Args:
        qdrant: Qdrant connection and collection settings.
        embedder: Dense embedding model settings.
        sparse_encoder: BM25 sparse encoder settings.
        reranker: Cross-encoder reranker settings.
        llm: LLM answer-generation backend settings.
        retrieval: Retrieval search and fusion parameters.
        ingestion: Document ingestion pipeline settings.
        evaluation: Ragas evaluation runner settings.

    Raises:
        :exc:`~src.exceptions.ConfigurationError`: When a required
            configuration key is absent or a value fails type validation.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    qdrant: QdrantSettings = QdrantSettings()
    embedder: EmbedderSettings = EmbedderSettings()
    sparse_encoder: SparseEncoderSettings = SparseEncoderSettings()
    reranker: RerankerSettings = RerankerSettings()
    llm: LLMSettings = LLMSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    ingestion: IngestionSettings = IngestionSettings()
    evaluation: EvaluationSettings = EvaluationSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Build the ordered tuple of settings sources.

        Priority (index 0 = highest priority):
            1. Environment variables (``env_settings``)
            2. YAML file (inserted here when ``YAML_CONFIG_PATH`` is set)
            3. ``.env`` file (``dotenv_settings``)

        Args:
            settings_cls: The ``AppSettings`` class.
            init_settings: Source for values passed directly to the constructor.
            env_settings: Source for OS/shell environment variables.
            dotenv_settings: Source for values read from the ``.env`` file.
            file_secret_settings: Source for secrets-directory files (not used here).

        Returns:
            Ordered tuple of ``PydanticBaseSettingsSource`` instances.
        """
        sources: list[PydanticBaseSettingsSource] = [env_settings]

        yaml_path = os.environ.get("YAML_CONFIG_PATH")
        if yaml_path:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))

        sources.append(dotenv_settings)
        return tuple(sources)

    @classmethod
    def create(cls) -> "AppSettings":
        """Construct ``AppSettings``, wrapping ``ValidationError`` in
        :exc:`~src.exceptions.ConfigurationError`.

        Returns:
            A fully-validated ``AppSettings`` instance.

        Raises:
            :exc:`~src.exceptions.ConfigurationError`: When a required
                configuration key is absent or a value fails type validation.
                The error message contains the name of the offending key.
        """
        try:
            return cls()
        except ValidationError as exc:
            # Extract the first missing / invalid field name for the message.
            missing_fields = [
                ".".join(str(loc) for loc in err["loc"])
                for err in exc.errors()
            ]
            field_list = ", ".join(f"'{f}'" for f in missing_fields)
            raise ConfigurationError(
                f"Missing or invalid configuration key(s): {field_list}"
            ) from exc


# ---------------------------------------------------------------------------
# Module-level cached accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the singleton ``AppSettings`` instance.

    The instance is constructed once and cached for the lifetime of the
    process.  To reload settings (e.g. in tests), call
    ``get_settings.cache_clear()`` before invoking this function again.

    Returns:
        The cached ``AppSettings`` instance.

    Raises:
        :exc:`~src.exceptions.ConfigurationError`: Propagated from
            :meth:`AppSettings.create` when configuration is invalid.
    """
    return AppSettings.create()
