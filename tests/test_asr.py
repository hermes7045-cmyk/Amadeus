from __future__ import annotations

import unittest

from echobot.asr import (
    ASRService,
    ProviderStatusSnapshot,
    SpeechSegment,
    TranscriptionResult,
    VADStepResult,
)
from echobot.asr.providers import ASRProvider
from echobot.asr.vad import VADProvider, VADSession


class FakeASRProvider(ASRProvider):
    def __init__(
        self,
        *,
        name: str = "fake-asr",
        available: bool = True,
        detail: str = "ASR ready",
    ) -> None:
        self.name = name
        self.label = "Fake ASR"
        self._available = available
        self._detail = detail
        self.startup_calls = 0
        self.transcribed_samples: list[list[float]] = []

    async def on_startup(self) -> None:
        self.startup_calls += 1

    async def status_snapshot(self) -> ProviderStatusSnapshot:
        return ProviderStatusSnapshot(
            kind="asr",
            name=self.name,
            label=self.label,
            selected=False,
            available=self._available,
            state="ready" if self._available else "missing",
            detail=self._detail,
            resource_directory="D:/fake/asr",
        )

    async def transcribe_samples(self, samples: list[float]) -> TranscriptionResult:
        self.transcribed_samples.append(list(samples))
        return TranscriptionResult(
            text=f"text-{len(samples)}",
            language="zh",
        )


class FakeVADSession(VADSession):
    def __init__(self, steps: list[VADStepResult]) -> None:
        self._steps = list(steps)
        self.reset_calls = 0

    def accept_audio_bytes(self, audio_bytes: bytes) -> VADStepResult:
        del audio_bytes
        if self._steps:
            return self._steps.pop(0)
        return VADStepResult()

    def flush(self) -> VADStepResult:
        if self._steps:
            return self._steps.pop(0)
        return VADStepResult()

    def reset(self) -> None:
        self.reset_calls += 1


class FakeVADProvider(VADProvider):
    def __init__(
        self,
        session: VADSession,
        *,
        name: str = "fake-vad",
        available: bool = True,
        detail: str = "VAD ready",
    ) -> None:
        self.name = name
        self.label = "Fake VAD"
        self._session = session
        self._available = available
        self._detail = detail
        self.startup_calls = 0

    async def on_startup(self) -> None:
        self.startup_calls += 1

    async def status_snapshot(self) -> ProviderStatusSnapshot:
        return ProviderStatusSnapshot(
            kind="vad",
            name=self.name,
            label=self.label,
            selected=False,
            available=self._available,
            state="ready" if self._available else "missing",
            detail=self._detail,
            resource_directory="D:/fake/vad",
        )

    async def create_session(self) -> VADSession:
        return self._session


class ASRServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_snapshot_reports_selected_providers(self) -> None:
        service = ASRService(
            {
                "fake-asr": FakeASRProvider(),
                "backup-asr": FakeASRProvider(name="backup-asr"),
            },
            {
                "fake-vad": FakeVADProvider(FakeVADSession([])),
            },
            selected_asr_provider="fake-asr",
            selected_vad_provider="fake-vad",
            sample_rate=16000,
        )

        snapshot = await service.status_snapshot()

        self.assertTrue(snapshot.available)
        self.assertEqual("fake-asr", snapshot.selected_asr_provider)
        self.assertEqual("fake-vad", snapshot.selected_vad_provider)
        self.assertTrue(snapshot.always_listen_supported)
        self.assertEqual(2, len(snapshot.asr_providers))
        self.assertTrue(
            any(item.name == "fake-asr" and item.selected for item in snapshot.asr_providers)
        )
        self.assertTrue(
            any(item.name == "fake-vad" and item.selected for item in snapshot.vad_providers)
        )

    async def test_status_snapshot_marks_always_listen_unavailable_without_vad(self) -> None:
        service = ASRService(
            {"fake-asr": FakeASRProvider()},
            {},
            selected_asr_provider="fake-asr",
            selected_vad_provider=None,
            sample_rate=16000,
        )

        snapshot = await service.status_snapshot()

        self.assertTrue(snapshot.available)
        self.assertFalse(snapshot.always_listen_supported)
        self.assertEqual("", snapshot.selected_vad_provider)
        self.assertIn("VAD provider", snapshot.detail)

    async def test_realtime_session_uses_vad_segments_and_asr_provider(self) -> None:
        asr_provider = FakeASRProvider()
        vad_session = FakeVADSession(
            [
                VADStepResult(
                    speech_started=True,
                    speech_ended=True,
                    segments=[
                        SpeechSegment(samples=[0.1, 0.2], start_ms=0),
                        SpeechSegment(samples=[0.3], start_ms=120),
                    ],
                )
            ]
        )
        service = ASRService(
            {"fake-asr": asr_provider},
            {"fake-vad": FakeVADProvider(vad_session)},
            selected_asr_provider="fake-asr",
            selected_vad_provider="fake-vad",
            sample_rate=16000,
        )

        session = await service.create_realtime_session()
        events = await session.accept_audio_bytes(b"chunk")
        await session.reset()

        self.assertEqual("speech_start", events[0]["type"])
        self.assertEqual("speech_end", events[1]["type"])
        self.assertEqual("transcript", events[2]["type"])
        self.assertEqual("text-2", events[2]["text"])
        self.assertEqual("transcript", events[3]["type"])
        self.assertEqual("text-1", events[3]["text"])
        self.assertEqual([[0.1, 0.2], [0.3]], asr_provider.transcribed_samples)
        self.assertEqual(1, vad_session.reset_calls)

    async def test_create_realtime_session_requires_ready_vad_provider(self) -> None:
        service = ASRService(
            {"fake-asr": FakeASRProvider()},
            {
                "fake-vad": FakeVADProvider(
                    FakeVADSession([]),
                    available=False,
                    detail="VAD missing",
                )
            },
            selected_asr_provider="fake-asr",
            selected_vad_provider="fake-vad",
            sample_rate=16000,
        )

        with self.assertRaisesRegex(RuntimeError, "VAD missing"):
            await service.create_realtime_session()

    async def test_set_selected_asr_provider_switches_active_provider(self) -> None:
        first_provider = FakeASRProvider(name="fake-asr")
        second_provider = FakeASRProvider(name="backup-asr")
        service = ASRService(
            {
                "fake-asr": first_provider,
                "backup-asr": second_provider,
            },
            {},
            selected_asr_provider="fake-asr",
            selected_vad_provider=None,
            sample_rate=16000,
        )

        await service.set_selected_asr_provider("backup-asr")
        snapshot = await service.status_snapshot()

        self.assertEqual("backup-asr", service.selected_asr_provider)
        self.assertEqual("backup-asr", snapshot.selected_asr_provider)
        self.assertEqual(1, second_provider.startup_calls)

    async def test_set_selected_asr_provider_starts_selected_vad_provider(self) -> None:
        asr_provider = FakeASRProvider()
        vad_provider = FakeVADProvider(FakeVADSession([]))
        service = ASRService(
            {"fake-asr": asr_provider},
            {"fake-vad": vad_provider},
            selected_asr_provider="fake-asr",
            selected_vad_provider="fake-vad",
            sample_rate=16000,
        )

        await service.set_selected_asr_provider("fake-asr")

        self.assertEqual(1, asr_provider.startup_calls)
        self.assertEqual(1, vad_provider.startup_calls)
