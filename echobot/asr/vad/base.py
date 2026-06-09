from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..models import ProviderStatusSnapshot


@dataclass(slots=True, frozen=True)
class SpeechSegment:
    samples: list[float]
    start_ms: int


@dataclass(slots=True, frozen=True)
class VADStepResult:
    speech_started: bool = False
    speech_ended: bool = False
    segments: list[SpeechSegment] = field(default_factory=list)


class VADSession(ABC):
    @abstractmethod
    def accept_audio_bytes(self, audio_bytes: bytes) -> VADStepResult:
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> VADStepResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class VADProvider(ABC):
    name: str
    label: str

    async def on_startup(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    async def status_snapshot(self) -> ProviderStatusSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def create_session(self) -> VADSession:
        raise NotImplementedError
