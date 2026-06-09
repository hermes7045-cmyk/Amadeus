from __future__ import annotations

import asyncio
import io
from typing import Any
from urllib.parse import urlparse

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency is installed in normal runtime
    openai = None
    OpenAI = None

from ..audio import write_wav_bytes
from ..models import ProviderStatusSnapshot, TranscriptionResult
from .base import ASRProvider


class OpenAITranscriptionsASRProvider(ASRProvider):
    name = "openai-transcriptions"
    label = "OpenAI Transcriptions"

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        api_key: str = "",
        model: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        language: str = "",
        prompt: str = "",
        temperature: float | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("ASR sample_rate must be positive")

        self._sample_rate = sample_rate
        self._api_key = api_key.strip()
        self._model = model.strip()
        self._base_url = base_url.strip()
        self._timeout = max(1.0, timeout)
        self._language = language.strip()
        self._prompt = prompt.strip()
        self._temperature = temperature
        self._client = self._build_client()

    async def close(self) -> None:
        if self._client is None:
            return
        await asyncio.to_thread(self._client.close)

    async def status_snapshot(self) -> ProviderStatusSnapshot:
        state, detail = self._status_state()
        return ProviderStatusSnapshot(
            kind="asr",
            name=self.name,
            label=self.label,
            selected=False,
            available=state == "ready",
            state=state,
            detail=detail,
            resource_directory=self._base_url,
        )

    async def transcribe_samples(self, samples: list[float]) -> TranscriptionResult:
        if not samples:
            return TranscriptionResult(text="")

        state, detail = self._status_state()
        if state != "ready":
            raise RuntimeError(detail)

        return await asyncio.to_thread(self._transcribe_samples_sync, list(samples))

    def _build_client(self):
        if OpenAI is None:
            return None
        return OpenAI(
            api_key=self._api_key or "EMPTY",
            base_url=self._base_url,
            timeout=self._timeout,
        )

    def _transcribe_samples_sync(self, samples: list[float]) -> TranscriptionResult:
        client = self._require_client()
        audio_buffer = io.BytesIO(write_wav_bytes(samples, self._sample_rate))
        audio_buffer.name = "audio.wav"

        request_kwargs: dict[str, Any] = {
            "file": audio_buffer,
            "model": self._model,
        }
        if self._language:
            request_kwargs["language"] = self._language
        if self._prompt:
            request_kwargs["prompt"] = self._prompt
        if self._temperature is not None:
            request_kwargs["temperature"] = self._temperature

        try:
            response = client.audio.transcriptions.create(**request_kwargs)
        except Exception as exc:
            raise _transcription_error(exc) from exc
        finally:
            audio_buffer.close()

        return self._parse_transcription_response(response)

    def _require_client(self):
        if self._client is None:
            raise RuntimeError(
                "OpenAI Python SDK is unavailable. Install it with: pip install openai",
            )
        return self._client

    def _status_state(self) -> tuple[str, str]:
        if OpenAI is None:
            return (
                "unavailable",
                "OpenAI Python SDK is unavailable. Install it with: pip install openai",
            )
        if not self._base_url:
            return "missing", "OpenAI transcriptions provider is missing ECHOBOT_ASR_OPENAI_BASE_URL."
        if not self._model:
            return "missing", "OpenAI transcriptions provider is missing ECHOBOT_ASR_OPENAI_MODEL."
        if self._uses_official_openai_endpoint() and self._api_key.upper() in {"", "EMPTY"}:
            return "missing", "OpenAI official transcription endpoint requires ECHOBOT_ASR_OPENAI_API_KEY."
        return (
            "ready",
            f"OpenAI-compatible ASR ready: model={self._model}, base_url={self._base_url}",
        )

    def _uses_official_openai_endpoint(self) -> bool:
        parsed = urlparse(self._base_url)
        return parsed.netloc.lower() == "api.openai.com"

    def _parse_transcription_response(self, response: Any) -> TranscriptionResult:
        if response is None:
            return TranscriptionResult(text="")

        if isinstance(response, str):
            return TranscriptionResult(text=response.strip())

        if isinstance(response, dict):
            data = dict(response)
        elif hasattr(response, "model_dump") and callable(response.model_dump):
            data = dict(response.model_dump())
        else:
            data = {
                "text": getattr(response, "text", ""),
                "language": getattr(response, "language", ""),
                "segments": getattr(response, "segments", None),
            }

        transcript_text = str(data.get("text", "") or "").strip()
        if not transcript_text:
            transcript_text = _segments_to_text(data.get("segments"))

        language = str(data.get("language", "") or "").strip()
        return TranscriptionResult(
            text=transcript_text,
            language=language,
        )


def _transcription_error(exc: Exception) -> RuntimeError:
    if openai is not None and isinstance(exc, openai.APIStatusError):
        status_code = getattr(exc, "status_code", "unknown")
        return RuntimeError(
            f"ASR provider request failed: status={status_code}, detail={exc}",
        )
    if openai is not None and isinstance(exc, openai.APITimeoutError):
        return RuntimeError(f"ASR provider request timed out: {exc}")
    if openai is not None and isinstance(exc, openai.APIConnectionError):
        return RuntimeError(f"ASR provider network error: {exc}")
    if openai is not None and isinstance(exc, openai.OpenAIError):
        return RuntimeError(f"ASR provider request failed: {exc}")
    return RuntimeError(f"ASR provider request failed: {exc}")


def _segments_to_text(segments: Any) -> str:
    if not isinstance(segments, list):
        return ""

    parts: list[str] = []
    for item in segments:
        if isinstance(item, dict):
            segment_text = str(item.get("text", "") or "")
        else:
            segment_text = str(getattr(item, "text", "") or "")
        if segment_text:
            parts.append(segment_text)
    return "".join(parts).strip()
