from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from .providers.gpt_sovits import GPTSoVITSTTSProvider
from .service import TTSService

DEFAULT_TTS_PROVIDER = "gpt-sovits"


def build_default_tts_service(workspace: Path) -> TTSService:
    return TTSService(
        {
            "gpt-sovits": build_default_gpt_sovits_tts_provider(),
        },
        default_provider=_env_text("AMADEUS_TTS_PROVIDER", DEFAULT_TTS_PROVIDER),
    )


def _resolve_optional_path(workspace: Path, raw_path: str) -> Path | None:
    normalized_path = raw_path.strip()
    if not normalized_path:
        return None

    candidate = Path(normalized_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return workspace / candidate


def _env_flag(name: str, default: bool) -> bool:
    raw_value = str(os.environ.get(name, "")).strip().lower()
    if not raw_value:
        return default
    return raw_value in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str) -> str:
    raw_value = str(os.environ.get(name, "")).strip()
    return raw_value or default


def _env_int(name: str, default: int) -> int:
    raw_value = str(os.environ.get(name, "")).strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = str(os.environ.get(name, "")).strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_csv(name: str) -> list[str]:
    raw_value = str(os.environ.get(name, "")).strip()
    if not raw_value:
        return []
    return [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]


def _env_json_object(name: str) -> dict[str, Any]:
    raw_value = str(os.environ.get(name, "")).strip()
    if not raw_value:
        return {}

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def build_default_gpt_sovits_tts_provider() -> GPTSoVITSTTSProvider:
    return GPTSoVITSTTSProvider(
        api_url=_env_text(
            "AMADEUS_TTS_GPT_SOVITS_API_URL",
            "http://127.0.0.1:9880/tts",
        ),
        ref_audio_path=_env_text(
            "AMADEUS_TTS_GPT_SOVITS_REF_AUDIO",
            "voices/CRS_JP.wav",
        ),
        prompt_text=_env_text(
            "AMADEUS_TTS_GPT_SOVITS_PROMPT_TEXT",
            "極端な管理社会全体主義まゆりがバナナを食べたいと思っても、今日がバナナを食べていい日でなければ食べることは許さ。",
        ),
        prompt_lang=_env_text("AMADEUS_TTS_GPT_SOVITS_PROMPT_LANG", "ja"),
        text_lang=_env_text("AMADEUS_TTS_GPT_SOVITS_TEXT_LANG", "ja"),
        timeout=max(1.0, _env_float("AMADEUS_TTS_GPT_SOVITS_TIMEOUT", 60.0)),
        speed_factor=max(0.1, _env_float("AMADEUS_TTS_GPT_SOVITS_SPEED", 0.9)),
    )
