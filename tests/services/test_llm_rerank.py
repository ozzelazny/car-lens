"""Unit tests for the Phase 6.2 :mod:`services.recognize_api.llm_rerank` module.

The real Anthropic client is never invoked. We construct
:class:`LLMReRanker` with a hand-rolled stub client whose
``messages.create`` method can be programmed to return a canned
response object or raise a chosen exception, then assert that
:meth:`LLMReRanker.rerank` produces the documented contract:

* Successful integer parse -> ``chosen_index`` matches.
* Whitespace / newline padded responses parse correctly.
* Out-of-range integer -> ``chosen_index == -1``.
* Malformed (no integer) response -> ``chosen_index == -1``.
* ``TimeoutError`` raised by the SDK -> ``chosen_index == -1`` and
  ``error`` mentions "timeout".
* Any other exception -> ``chosen_index == -1`` and ``error`` contains
  the exception's stringified message.

The module also has an image-resize helper; we cover it by feeding a
PIL image with a known oversize aspect ratio and checking the
post-resize dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image as PILImage  # noqa: E402

from services.recognize_api.llm_rerank import LLMReRanker, RerankResult  # noqa: E402

# --------------------------------------------------------------- stub client


@dataclass
class _StubTextBlock:
    """Mimics ``anthropic.types.TextBlock`` for ``_extract_text``."""

    text: str


@dataclass
class _StubMessage:
    """Mimics an ``anthropic.types.Message`` response object."""

    content: list[_StubTextBlock]


class _StubMessages:
    """The ``messages`` attribute on :class:`_StubAnthropicClient`."""

    def __init__(
        self,
        *,
        text: str | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self._text = text
        self._exc = exc
        self.last_kwargs: dict[str, Any] | None = None
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return _StubMessage(content=[_StubTextBlock(text=self._text or "")])


class _StubAnthropicClient:
    """Minimal stand-in for :class:`anthropic.Anthropic`.

    Only exposes ``.messages.create`` since that's all
    :meth:`LLMReRanker.rerank` calls.
    """

    def __init__(
        self,
        *,
        text: str | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self.messages = _StubMessages(text=text, exc=exc)


# --------------------------------------------------------------- fixtures


def _make_image(width: int, height: int, color: tuple[int, int, int] = (200, 50, 50)) -> Any:
    return PILImage.new("RGB", (width, height), color=color)


def _build_reranker(
    *,
    text: str | None = None,
    exc: BaseException | None = None,
    max_image_dim: int = 768,
    timeout_seconds: float = 10.0,
) -> tuple[LLMReRanker, _StubAnthropicClient]:
    stub = _StubAnthropicClient(text=text, exc=exc)
    reranker = LLMReRanker(
        api_key="sk-test-stub",
        model="claude-sonnet-4-6",
        timeout_seconds=timeout_seconds,
        max_image_dim=max_image_dim,
        client=stub,
    )
    return reranker, stub


# --------------------------------------------------------------- tests


def test_llm_reranker_resizes_image_to_max_dim() -> None:
    """A 2000x1000 image is shrunk to fit inside a 768x768 box.

    Aspect ratio is preserved, so the result is 768x384.
    """
    reranker, _ = _build_reranker(text="0", max_image_dim=768)
    big = _make_image(2000, 1000)
    resized = reranker._resize_to_fit(big)
    assert resized.size == (768, 384)


def test_llm_reranker_does_not_upscale_small_image() -> None:
    """A 320x240 image is returned as-is, never upscaled."""
    reranker, _ = _build_reranker(text="0", max_image_dim=768)
    small = _make_image(320, 240)
    resized = reranker._resize_to_fit(small)
    # No copy made -- we get back the same object.
    assert resized is small
    assert resized.size == (320, 240)


def test_llm_reranker_parses_integer_response() -> None:
    """``"2"`` -> ``chosen_index == 2``."""
    reranker, stub = _build_reranker(text="2")
    result = reranker.rerank(
        image=_make_image(640, 480),
        candidates=["a", "b", "c", "d", "e"],
    )
    assert isinstance(result, RerankResult)
    assert result.chosen_index == 2
    assert result.error is None
    assert result.raw_response == "2"
    assert result.latency_ms >= 0.0
    assert stub.messages.call_count == 1


def test_llm_reranker_parses_response_with_whitespace() -> None:
    """``" 3 \\n"`` -> ``chosen_index == 3``."""
    reranker, _ = _build_reranker(text=" 3 \n")
    result = reranker.rerank(
        image=_make_image(640, 480),
        candidates=["a", "b", "c", "d", "e"],
    )
    assert result.chosen_index == 3
    assert result.error is None


def test_llm_reranker_handles_out_of_range_response() -> None:
    """``"7"`` with 5 candidates -> ``chosen_index == -1``."""
    reranker, _ = _build_reranker(text="7")
    result = reranker.rerank(
        image=_make_image(640, 480),
        candidates=["a", "b", "c", "d", "e"],
    )
    assert result.chosen_index == -1
    assert result.error is not None


def test_llm_reranker_handles_malformed_response() -> None:
    """Non-integer prose -> ``chosen_index == -1``."""
    reranker, _ = _build_reranker(text="the answer is car")
    result = reranker.rerank(
        image=_make_image(640, 480),
        candidates=["a", "b", "c", "d", "e"],
    )
    assert result.chosen_index == -1
    assert result.error is not None


def test_llm_reranker_handles_api_timeout() -> None:
    """A ``TimeoutError`` raised by the SDK is swallowed cleanly."""
    reranker, _ = _build_reranker(exc=TimeoutError("request timed out"))
    result = reranker.rerank(
        image=_make_image(640, 480),
        candidates=["a", "b", "c"],
    )
    assert result.chosen_index == -1
    assert result.error is not None
    assert "timeout" in result.error.lower()


def test_llm_reranker_handles_generic_exception() -> None:
    """A ``RuntimeError("boom")`` is reported in ``error`` without raising."""
    reranker, _ = _build_reranker(exc=RuntimeError("boom"))
    result = reranker.rerank(
        image=_make_image(640, 480),
        candidates=["a", "b", "c"],
    )
    assert result.chosen_index == -1
    assert result.error is not None
    assert "boom" in result.error


def test_llm_reranker_handles_empty_candidates() -> None:
    """No candidates -> short-circuit to ``chosen_index == -1`` (no API call)."""
    reranker, stub = _build_reranker(text="0")
    result = reranker.rerank(image=_make_image(64, 64), candidates=[])
    assert result.chosen_index == -1
    assert result.error is not None
    # SDK should never have been called.
    assert stub.messages.call_count == 0


def test_llm_reranker_passes_timeout_to_sdk() -> None:
    """The ``timeout_seconds`` kwarg is forwarded to ``messages.create``."""
    reranker, stub = _build_reranker(text="0", timeout_seconds=5.5)
    reranker.rerank(image=_make_image(64, 64), candidates=["a", "b"])
    assert stub.messages.last_kwargs is not None
    assert stub.messages.last_kwargs["timeout"] == 5.5
    assert stub.messages.last_kwargs["model"] == "claude-sonnet-4-6"


def test_llm_reranker_sends_base64_image_in_request() -> None:
    """The request payload includes a base64 JPEG block + a text prompt."""
    reranker, stub = _build_reranker(text="0")
    reranker.rerank(
        image=_make_image(64, 64),
        candidates=["Honda Civic 2012", "Toyota Camry 2012"],
    )
    assert stub.messages.last_kwargs is not None
    messages = stub.messages.last_kwargs["messages"]
    assert len(messages) == 1
    content = messages[0]["content"]
    image_blocks = [b for b in content if b["type"] == "image"]
    text_blocks = [b for b in content if b["type"] == "text"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["type"] == "base64"
    assert image_blocks[0]["source"]["media_type"] == "image/jpeg"
    assert len(image_blocks[0]["source"]["data"]) > 0
    assert len(text_blocks) == 1
    assert "Honda Civic 2012" in text_blocks[0]["text"]
    assert "Toyota Camry 2012" in text_blocks[0]["text"]


def test_llm_reranker_requires_api_key() -> None:
    """An empty API key is rejected at construction."""
    with pytest.raises(ValueError, match="api_key"):
        LLMReRanker(api_key="", client=_StubAnthropicClient(text="0"))
