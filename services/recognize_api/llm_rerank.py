"""Phase 6.2 — Cloud-LLM re-rank fallback for the recognize() service.

When the on-device top-1 confidence is below a configured threshold we
fall back to the Anthropic Claude API to disambiguate among the top-K
candidate (year, make, model) labels using the image itself. Per the
Phase 5.4 evaluation our top-5 contains the right answer ~98% of the
time; an LLM looking at the image plus the 5 candidate names can pick
the correct trim/year far more reliably than the cosine-similarity
top-1 alone.

This module is **strictly additive**: it never raises. Any failure
(network timeout, malformed response, parse error, exceeded
out-of-range index) results in ``RerankResult(chosen_index=-1, ...)``
and the caller leaves the original ranking untouched. The recognize
endpoint is sync, so we use the sync :class:`anthropic.Anthropic`
client; at the expected 1-2 inflight QPS the small blocking call is
fine.

The image is **down-sized to fit inside ``max_image_dim`` (px)** before
upload — Claude vision charges per pixel, and 768 px is plenty to
disambiguate trim-year edges. We JPEG-encode at quality 85 and
base64-encode (the SDK accepts both ``image/jpeg`` and ``image/png``
but JPEG is meaningfully smaller for a real photo).
"""

from __future__ import annotations

import base64
import io
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- import only for type checkers
    from PIL import Image as PILImageType

logger = logging.getLogger(__name__)


#: Regex that pulls the first non-negative integer token out of the
#: model's response. ``re.search`` is intentional — we tolerate leading
#: whitespace / newlines and quietly ignore trailing prose (the system
#: prompt forbids prose, but defence-in-depth).
_INT_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class RerankResult:
    """Outcome of one re-rank call.

    Attributes
    ----------
    chosen_index:
        Index into the candidate list (``0..len(candidates)-1``) that
        the LLM picked. ``-1`` on any failure (parse, timeout, API
        error, out-of-range response).
    raw_response:
        The raw text content the LLM returned. Useful for logging /
        debugging; empty string when the call itself failed.
    latency_ms:
        Wall-clock latency of the rerank call (including resizing + the
        API round trip) in milliseconds.
    error:
        Human-readable error message on failure; ``None`` on success.
    """

    chosen_index: int
    raw_response: str
    latency_ms: float
    error: str | None


class LLMReRanker:
    """Thin wrapper around the Anthropic Claude vision API.

    The instance is safe to share across requests: the underlying
    ``anthropic.Anthropic`` client handles its own connection pool.
    """

    _SYSTEM_PROMPT = (
        "You are a vehicle identification assistant. Given a car photo and a "
        "numbered list of candidate (year, make, model) options, choose the "
        "single option that best matches the photo. Output ONLY the chosen "
        'number (e.g. "2"). No prose, no explanation.'
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        timeout_seconds: float = 10.0,
        max_image_dim: int = 768,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required to construct LLMReRanker")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = float(timeout_seconds)
        self._max_image_dim = int(max_image_dim)
        if client is not None:
            self._client = client
        else:  # pragma: no cover -- exercised in production only; tests inject ``client``
            import anthropic  # noqa: PLC0415 -- import lazily so tests can stub

            self._client = anthropic.Anthropic(api_key=api_key)

    # ----------------------------------------------------------------- props

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    @property
    def max_image_dim(self) -> int:
        return self._max_image_dim

    # ----------------------------------------------------------------- API

    def rerank(
        self,
        *,
        image: PILImageType.Image,
        candidates: list[str],
    ) -> RerankResult:
        """Pick the best candidate index for ``image`` from ``candidates``.

        Never raises. On any failure the returned
        :class:`RerankResult` carries ``chosen_index=-1`` and a non-None
        ``error`` field.
        """
        start = time.perf_counter()
        if not candidates:
            return RerankResult(
                chosen_index=-1,
                raw_response="",
                latency_ms=(time.perf_counter() - start) * 1000.0,
                error="no candidates provided",
            )
        try:
            jpeg_b64 = self._encode_image(image)
        except Exception as exc:  # noqa: BLE001 -- never raise
            return RerankResult(
                chosen_index=-1,
                raw_response="",
                latency_ms=(time.perf_counter() - start) * 1000.0,
                error=f"image encoding failed: {exc!s}",
            )

        prompt_text = self._build_user_prompt(candidates)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=8,
                system=self._SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": jpeg_b64,
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ],
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- the SDK raises many types
            latency_ms = (time.perf_counter() - start) * 1000.0
            err = self._format_error(exc)
            logger.warning("llm_rerank: API call failed: %s", err)
            return RerankResult(
                chosen_index=-1,
                raw_response="",
                latency_ms=latency_ms,
                error=err,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        raw_text = self._extract_text(response)
        chosen = self._parse_index(raw_text, n_candidates=len(candidates))
        return RerankResult(
            chosen_index=chosen,
            raw_response=raw_text,
            latency_ms=latency_ms,
            error=None if chosen >= 0 else f"could not parse a valid index from {raw_text!r}",
        )

    # ----------------------------------------------------------------- helpers

    def _encode_image(self, image: PILImageType.Image) -> str:
        """Resize -> JPEG -> base64 the input image.

        Resizing preserves the aspect ratio and only shrinks (never
        upscales) so a 320x240 thumbnail stays at 320x240. The longest
        side is clipped to :attr:`max_image_dim`.
        """
        from PIL import Image as PILImage  # noqa: PLC0415

        resized = self._resize_to_fit(image)
        # Ensure RGB so JPEG encoding always succeeds (RGBA / P modes
        # would explode here).
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        # Free the resized copy promptly — PIL holds the raw pixel
        # buffer until close().
        if resized is not image:
            resized.close()
        del PILImage  # quiet pyflakes (we import only to ensure PIL is available)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _resize_to_fit(self, image: PILImageType.Image) -> PILImageType.Image:
        """Shrink ``image`` so the longest side is at most ``max_image_dim``."""
        from PIL import Image as PILImage  # noqa: PLC0415

        width, height = image.size
        longest = max(width, height)
        if longest <= self._max_image_dim:
            return image
        scale = self._max_image_dim / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        # ``PILImage.Resampling.LANCZOS`` on Pillow >= 10. Fall back to
        # the int constant for very old Pillow (none of our supported
        # envs hit this, but it's a one-liner safety net).
        try:
            resample = PILImage.Resampling.LANCZOS
        except AttributeError:  # pragma: no cover
            resample = PILImage.LANCZOS  # type: ignore[attr-defined]
        return image.resize(new_size, resample=resample)

    @staticmethod
    def _build_user_prompt(candidates: list[str]) -> str:
        lines = ["Candidates:"]
        for i, name in enumerate(candidates):
            lines.append(f"{i}. {name}")
        lines.append("")
        lines.append("Which number is the correct match?")
        return "\n".join(lines)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text body out of a Claude messages-API response.

        The response is an ``anthropic.types.Message`` whose ``content``
        is a list of typed blocks. We concatenate every ``text`` block
        (the system prompt asks for only a number, so in practice there
        is one) and return the joined string. If the response shape is
        unfamiliar, returns ``""`` so the parser falls through to a
        ``chosen_index=-1`` outcome.
        """
        content = getattr(response, "content", None)
        if content is None:
            return ""
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _parse_index(raw: str, *, n_candidates: int) -> int:
        """Pull the first non-negative integer from ``raw`` and bounds-check it.

        Returns ``-1`` if no integer is present or the integer is
        outside ``[0, n_candidates - 1]``.
        """
        match = _INT_RE.search(raw.strip())
        if match is None:
            return -1
        try:
            value = int(match.group(0))
        except ValueError:  # pragma: no cover -- regex guarantees digits
            return -1
        if 0 <= value < n_candidates:
            return value
        return -1

    @staticmethod
    def _format_error(exc: BaseException) -> str:
        """Render an exception into a short, human-readable error string.

        We special-case timeouts so the caller can pattern-match on
        ``"timeout"`` in tests / metrics.
        """
        import anthropic  # noqa: PLC0415

        if isinstance(exc, TimeoutError | anthropic.APITimeoutError):
            return f"timeout after request: {exc!s}"
        message = str(exc).strip()
        if not message:
            message = exc.__class__.__name__
        return message


__all__ = ["LLMReRanker", "RerankResult"]
