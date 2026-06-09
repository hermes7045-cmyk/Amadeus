from __future__ import annotations

import asyncio
from typing import Any

from .providers import ASRProvider
from .vad import VADSession, VADStepResult


class RealtimeASRSession:
    def __init__(
        self,
        asr_provider: ASRProvider,
        vad_session: VADSession,
    ) -> None:
        self._asr_provider = asr_provider
        self._vad_session = vad_session

    async def accept_audio_bytes(self, audio_bytes: bytes) -> list[dict[str, Any]]:
        step_result = await asyncio.to_thread(
            self._vad_session.accept_audio_bytes,
            audio_bytes,
        )
        return await self._build_events(step_result)

    async def flush(self) -> list[dict[str, Any]]:
        step_result = await asyncio.to_thread(self._vad_session.flush)
        return await self._build_events(step_result)

    async def reset(self) -> None:
        await asyncio.to_thread(self._vad_session.reset)

    async def _build_events(self, step_result: VADStepResult) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if step_result.speech_started:
            events.append({"type": "speech_start"})

        transcript_events: list[dict[str, Any]] = []
        for segment in step_result.segments:
            result = await self._asr_provider.transcribe_samples(segment.samples)
            if not result.text:
                continue

            transcript_events.append(
                {
                    "type": "transcript",
                    "text": result.text,
                    "language": result.language,
                    "final": True,
                    "start_ms": segment.start_ms,
                }
            )

        if step_result.speech_ended and transcript_events:
            events.append({"type": "speech_end"})
        events.extend(transcript_events)
        return events
