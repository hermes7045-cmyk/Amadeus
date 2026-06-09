from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class ProviderStatusSnapshot:
    kind: str
    name: str
    label: str
    selected: bool
    available: bool
    state: str
    detail: str = ""
    resource_directory: str = ""


@dataclass(slots=True, frozen=True)
class ASRStatusSnapshot:
    available: bool
    state: str
    detail: str
    sample_rate: int
    selected_asr_provider: str
    selected_vad_provider: str
    always_listen_supported: bool
    asr_providers: list[ProviderStatusSnapshot] = field(default_factory=list)
    vad_providers: list[ProviderStatusSnapshot] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class TranscriptionResult:
    text: str
    language: str = ""
