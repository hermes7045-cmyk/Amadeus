from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class VoiceOption:
    name: str
    short_name: str
    locale: str = ""
    gender: str = ""
    display_name: str = ""


@dataclass(slots=True)
class SynthesizedSpeech:
    audio_bytes: bytes
    content_type: str
    file_extension: str
    provider: str
    voice: str


@dataclass(slots=True, frozen=True)
class TTSProviderStatus:
    name: str
    label: str
    available: bool
    state: str = "ready"
    detail: str = ""


@dataclass(slots=True, frozen=True)
class TTSSynthesisOptions:
    voice: str | None = None
    speed: float | None = None
    volume: str | None = None
    pitch: str | None = None


class TTSProvider(ABC):
    name: str
    label: str

    @property
    @abstractmethod
    def default_voice(self) -> str:
        raise NotImplementedError

    def status(self) -> TTSProviderStatus:
        return TTSProviderStatus(
            name=self.name,
            label=self.label,
            available=True,
        )

    async def list_voices(self) -> list[VoiceOption]:
        return []

    async def close(self) -> None:
        return None

    @abstractmethod
    async def synthesize(
        self,
        *,
        text: str,
        options: TTSSynthesisOptions | None = None,
    ) -> SynthesizedSpeech:
        raise NotImplementedError
