from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ProviderStatusSnapshot, TranscriptionResult


class ASRProvider(ABC):
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
    async def transcribe_samples(self, samples: list[float]) -> TranscriptionResult:
        raise NotImplementedError
