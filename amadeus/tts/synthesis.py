from __future__ import annotations

from .base import TTSSynthesisOptions


MIN_TTS_SPEED = 0.25
MAX_TTS_SPEED = 4.0


def build_tts_synthesis_options(
    *,
    voice: str | None = None,
    rate: str | None = None,
    volume: str | None = None,
    pitch: str | None = None,
) -> TTSSynthesisOptions:
    return TTSSynthesisOptions(
        voice=_clean_optional_text(voice),
        speed=parse_tts_speed(rate),
        volume=_clean_optional_text(volume),
        pitch=_clean_optional_text(pitch),
    )


def parse_tts_speed(rate: str | None) -> float | None:
    raw_rate = _clean_optional_text(rate)
    if raw_rate is None:
        return None

    try:
        if raw_rate.endswith("%"):
            percent_text = raw_rate[:-1].strip()
            percent_value = float(percent_text)
            if raw_rate.startswith(("+", "-")):
                speed = 1.0 + (percent_value / 100.0)
            else:
                speed = percent_value / 100.0
        else:
            speed = float(raw_rate)
    except ValueError as exc:
        raise ValueError(f"Invalid TTS rate: {rate}") from exc

    if speed <= 0:
        raise ValueError("TTS rate must be greater than zero")

    return min(MAX_TTS_SPEED, max(MIN_TTS_SPEED, speed))


def edge_rate_from_speed(speed: float | None) -> str | None:
    if speed is None:
        return None

    percent_delta = (speed - 1.0) * 100.0
    if abs(percent_delta) < 1e-9:
        return None

    sign = "+" if percent_delta > 0 else ""
    return f"{sign}{_format_number(percent_delta)}%"


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    normalized_value = str(value).strip()
    if not normalized_value:
        return None
    return normalized_value


def _format_number(value: float) -> str:
    formatted = f"{value:.2f}"
    return formatted.rstrip("0").rstrip(".")
