"""LLM generation component for the Hybrid RAG Evaluation Pipeline.

The :class:`~src.generation.generator.LLMGenerator` builds an instruction
prompt using the query and retrieved context, truncates the context to a
token-budget using ``tiktoken``, and forwards the request to a configurable
LLM backend.
"""

from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import tiktoken

from src.config.settings import LLMSettings
from src.exceptions import GenerationError, StreamingError
from src.models import GenerationResult


@dataclass(frozen=True)
class _BackendResult:
    answer: str
    model_id: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMBackend(abc.ABC):
    """Abstract LLM backend interface used by :class:`LLMGenerator`."""

    @abc.abstractmethod
    async def generate(self, prompt: str) -> _BackendResult:
        """Generate a complete answer for the given prompt."""

    @abc.abstractmethod
    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream answer tokens/chunks for the given prompt."""


class OpenAIBackend(LLMBackend):
    """OpenAI Chat Completions backend."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        if not self._settings.api_key and not os.getenv("OPENAI_API_KEY"):
            raise GenerationError(
                "Missing OpenAI API key. Set llm.api_key or OPENAI_API_KEY."
            )
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                "OpenAI backend selected but the 'openai' package is not installed."
            ) from exc
        self._client = AsyncOpenAI(api_key=self._settings.api_key, base_url=self._settings.base_url)

    async def generate(self, prompt: str) -> _BackendResult:
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"OpenAI request failed: {exc}") from exc

        answer = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        return _BackendResult(
            answer=answer,
            model_id=self._settings.model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"OpenAI streaming request failed: {exc}") from exc

        async for event in stream:
            try:
                delta = event.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
            except Exception as exc:  # noqa: BLE001
                raise StreamingError(f"Failed to parse OpenAI stream event: {exc}") from exc


class OllamaBackend(LLMBackend):
    """Ollama HTTP backend."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        if not settings.base_url:
            raise GenerationError("Ollama backend requires llm.base_url to be set.")
        self._base_url = settings.base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, prompt: str) -> _BackendResult:
        url = f"{self._base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self._settings.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            response = await self._client.post(url, json=payload)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Ollama request failed: {exc}") from exc

        if response.status_code >= 400:
            raise GenerationError(
                f"Ollama returned HTTP {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        answer = ""
        if isinstance(data, dict):
            message = data.get("message")
            if isinstance(message, dict):
                answer = str(message.get("content") or "")
            if not answer:
                answer = str(data.get("response") or "")
        return _BackendResult(
            answer=answer.strip(),
            model_id=self._settings.model_name,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=None,
        )

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        url = f"{self._base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self._settings.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        try:
            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise GenerationError(
                        f"Ollama returned HTTP {response.status_code}: {body.decode('utf-8', errors='replace')}",
                        status_code=response.status_code,
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk, dict) and chunk.get("done") is True:
                        break
                    content = ""
                    message = chunk.get("message") if isinstance(chunk, dict) else None
                    if isinstance(message, dict):
                        content = str(message.get("content") or "")
                    if not content and isinstance(chunk, dict):
                        content = str(chunk.get("response") or "")
                    if content:
                        yield content
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StreamingError(f"Ollama streaming failed: {exc}") from exc


class LiteLLMBackend(LLMBackend):
    """LiteLLM backend wrapper."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        try:
            import litellm  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                "LiteLLM backend selected but the 'litellm' package is not installed."
            ) from exc
        self._litellm = litellm

    async def generate(self, prompt: str) -> _BackendResult:
        try:
            response = await self._litellm.acompletion(
                model=self._settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._settings.api_key,
                base_url=self._settings.base_url,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LiteLLM request failed: {exc}") from exc

        message = response["choices"][0]["message"]
        answer = str(message.get("content") or "").strip()
        usage = response.get("usage") if isinstance(response, dict) else None
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        return _BackendResult(
            answer=answer,
            model_id=self._settings.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            stream = await self._litellm.acompletion(
                model=self._settings.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._settings.api_key,
                base_url=self._settings.base_url,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"LiteLLM streaming request failed: {exc}") from exc

        async for event in stream:
            try:
                choices = event.get("choices") if isinstance(event, dict) else None
                if not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                content = delta.get("content") if isinstance(delta, dict) else None
                if content:
                    yield str(content)
            except Exception as exc:  # noqa: BLE001
                raise StreamingError(f"Failed to parse LiteLLM stream event: {exc}") from exc


class LLMGenerator:
    """Generate answers from retrieved context using a configurable LLM backend.

    Args:
        settings: LLM configuration.
        backend: Optional injected backend implementation (primarily for tests).
    """

    _SYSTEM_INSTRUCTION = "You are a helpful assistant. Answer using only the provided context."

    def __init__(self, settings: LLMSettings, backend: Optional[LLMBackend] = None) -> None:
        self._settings = settings
        self._backend = backend or self._create_backend(settings)
        self._encoding = self._get_encoding(settings.model_name)

    async def generate(self, query: str, context: List[str]) -> GenerationResult:
        """Generate an answer using the configured backend.

        Args:
            query: User query.
            context: Context chunk texts (post-rerank order).

        Returns:
            A :class:`~src.models.GenerationResult` with answer text and token usage.

        Raises:
            GenerationError: If the backend call fails.
        """
        truncated_context = self._truncate_context(context)
        prompt = self._build_prompt(query, truncated_context)
        try:
            backend_result = await self._backend.generate(prompt)
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Generation failed: {exc}") from exc
        prompt_tokens = backend_result.prompt_tokens
        completion_tokens = backend_result.completion_tokens
        total_tokens = backend_result.total_tokens

        if prompt_tokens is None:
            prompt_tokens = self._count_tokens(prompt)
        if completion_tokens is None:
            completion_tokens = self._count_tokens(backend_result.answer)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return GenerationResult(
            answer=backend_result.answer,
            model_id=backend_result.model_id,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
        )

    async def stream(self, query: str, context: List[str]) -> AsyncGenerator[str, None]:
        """Stream an answer using the configured backend.

        Args:
            query: User query.
            context: Context chunk texts (post-rerank order).

        Yields:
            Token/chunk strings as produced by the backend.

        Raises:
            GenerationError: If a failure happens before streaming starts.
            StreamingError: If a failure happens mid-stream.
        """
        truncated_context = self._truncate_context(context)
        prompt = self._build_prompt(query, truncated_context)

        emitted_any = False
        try:
            async for token in self._backend.stream(prompt):
                emitted_any = True
                yield token
        except StreamingError:
            raise
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            if emitted_any:
                raise StreamingError(f"Streaming failed: {exc}") from exc
            raise GenerationError(f"Generation failed: {exc}") from exc

    def _truncate_context(self, context: List[str]) -> List[str]:
        budget = max(0, int(self._settings.max_context_tokens))
        if budget == 0 or not context:
            return []

        kept: List[str] = []
        used = 0
        for chunk in context:
            chunk_tokens = self._count_tokens(chunk)
            if used + chunk_tokens > budget:
                break
            kept.append(chunk)
            used += chunk_tokens
        return kept

    def _build_prompt(self, query: str, context: List[str]) -> str:
        context_block = "\n---\n".join(context)
        return (
            f"System: {self._SYSTEM_INSTRUCTION}\n"
            "Context:\n"
            f"{context_block}\n"
            f"User: {query}\n"
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    @staticmethod
    def _get_encoding(model_name: str) -> tiktoken.Encoding:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _create_backend(settings: LLMSettings) -> LLMBackend:
        if settings.provider == "openai":
            return OpenAIBackend(settings)
        if settings.provider == "ollama":
            return OllamaBackend(settings)
        if settings.provider == "litellm":
            return LiteLLMBackend(settings)
        raise GenerationError(f"Unsupported LLM provider: {settings.provider}")
