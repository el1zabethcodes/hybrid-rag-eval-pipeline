from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List

import pytest

from src.config.settings import LLMSettings
from src.exceptions import GenerationError, StreamingError
from src.generation.generator import LLMBackend, LLMGenerator, _BackendResult


class DummyBackend(LLMBackend):
    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer
        self.last_prompt: str | None = None

    async def generate(self, prompt: str) -> _BackendResult:
        self.last_prompt = prompt
        return _BackendResult(answer=self.answer, model_id="dummy-model")

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        self.last_prompt = prompt
        yield "a"
        yield "b"


class FailingStreamBackend(LLMBackend):
    async def generate(self, prompt: str) -> _BackendResult:
        raise RuntimeError(prompt)

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        _ = prompt
        yield "x"
        raise RuntimeError("boom")


def test_generate_prompt_contains_query_and_context() -> None:
    backend = DummyBackend(answer="answer")
    settings = LLMSettings(
        provider="openai",
        model_name="gpt-4o-mini",
        max_context_tokens=10_000,
        stream_response=False,
        api_key=None,
        base_url=None,
    )
    generator = LLMGenerator(settings, backend=backend)

    query = "What is RAG?"
    context = ["chunk one", "chunk two"]
    result = asyncio.run(generator.generate(query, context))

    assert result.answer == "answer"
    assert backend.last_prompt is not None
    assert query in backend.last_prompt
    for c in context:
        assert c in backend.last_prompt


def test_context_truncation_respects_budget() -> None:
    backend = DummyBackend(answer="answer")
    chunk1 = "hello world"
    chunk2 = "another chunk"
    encoding = LLMGenerator._get_encoding("gpt-4o-mini")
    tokens1 = len(encoding.encode(chunk1))
    tokens2 = len(encoding.encode(chunk2))
    budget = tokens1 + tokens2 - 1
    settings = LLMSettings(
        provider="openai",
        model_name="gpt-4o-mini",
        max_context_tokens=budget,
        stream_response=False,
        api_key=None,
        base_url=None,
    )
    generator = LLMGenerator(settings, backend=backend)
    result = asyncio.run(generator.generate("q", [chunk1, chunk2]))

    assert result.answer == "answer"
    assert backend.last_prompt is not None
    assert chunk1 in backend.last_prompt
    assert chunk2 not in backend.last_prompt


@pytest.mark.asyncio
async def test_stream_passthrough() -> None:
    backend = DummyBackend()
    settings = LLMSettings(
        provider="openai",
        model_name="gpt-4o-mini",
        max_context_tokens=10_000,
        stream_response=True,
        api_key=None,
        base_url=None,
    )
    generator = LLMGenerator(settings, backend=backend)

    chunks: List[str] = []
    async for tok in generator.stream("q", ["ctx"]):
        chunks.append(tok)

    assert chunks == ["a", "b"]


@pytest.mark.asyncio
async def test_stream_raises_streaming_error_after_partial_output() -> None:
    settings = LLMSettings(
        provider="openai",
        model_name="gpt-4o-mini",
        max_context_tokens=10_000,
        stream_response=True,
        api_key=None,
        base_url=None,
    )
    generator = LLMGenerator(settings, backend=FailingStreamBackend())

    seen = []
    with pytest.raises(StreamingError):
        async for tok in generator.stream("q", ["ctx"]):
            seen.append(tok)

    assert seen == ["x"]


@pytest.mark.asyncio
async def test_generate_wraps_backend_errors() -> None:
    settings = LLMSettings(
        provider="openai",
        model_name="gpt-4o-mini",
        max_context_tokens=10_000,
        stream_response=False,
        api_key=None,
        base_url=None,
    )
    generator = LLMGenerator(settings, backend=FailingStreamBackend())

    with pytest.raises(GenerationError):
        await generator.generate("q", ["ctx"])
