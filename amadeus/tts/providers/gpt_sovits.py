"""GPT-SoVITS TTS provider — native integration via HTTP API."""
from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from ..base import (
    SynthesizedSpeech,
    TTSProvider,
    TTSProviderStatus,
    TTSSynthesisOptions,
    VoiceOption,
)

KURISU_VOICE = VoiceOption(
    name="kurisu",
    short_name="kurisu",
    display_name="牧瀬紅莉栖",
    locale="ja-JP",
    gender="female",
)


class GPTSoVITSTTSProvider(TTSProvider):
    name = "gpt-sovits"
    label = "GPT-SoVITS (Kurisu)"

    def __init__(
        self,
        *,
        api_url: str = "http://127.0.0.1:9880/tts",
        ref_audio_path: str = "voices/CRS_JP.wav",
        prompt_text: str = (
            "極端な管理社会全体主義まゆりがバナナを食べたいと思っても、"
            "今日がバナナを食べていい日でなければ食べることは許さ。"
        ),
        prompt_lang: str = "ja",
        text_lang: str = "ja",
        timeout: float = 60.0,
        speed_factor: float = 0.9,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._ref_audio_path = ref_audio_path
        self._prompt_text = prompt_text
        self._prompt_lang = prompt_lang
        self._text_lang = text_lang
        self._timeout = max(1.0, timeout)
        self._speed_factor = speed_factor
        self._default_voice = "kurisu"

    @property
    def default_voice(self) -> str:
        return self._default_voice

    def status(self) -> TTSProviderStatus:
        try:
            req = request.Request(
                url=self._api_url,
                data=json.dumps({
                    "text": "test",
                    "text_lang": self._text_lang,
                    "ref_audio_path": self._ref_audio_path,
                    "prompt_lang": self._prompt_lang,
                    "prompt_text": self._prompt_text,
                    "text_split_method": "cut5",
                    "batch_size": 1,
                    "media_type": "wav",
                    "streaming_mode": False,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    return TTSProviderStatus(
                        name=self.name,
                        label=self.label,
                        available=True,
                        state="ready",
                    )
                return TTSProviderStatus(
                    name=self.name,
                    label=self.label,
                    available=False,
                    state="error",
                    detail=f"GPT-SoVITS returned {resp.status}",
                )
        except Exception as exc:
            return TTSProviderStatus(
                name=self.name,
                label=self.label,
                available=False,
                state="error",
                detail=str(exc),
            )

    async def list_voices(self) -> list[VoiceOption]:
        return [KURISU_VOICE]

    async def synthesize(
        self,
        *,
        text: str,
        options: TTSSynthesisOptions | None = None,
    ) -> SynthesizedSpeech:
        synthesis_options = options or TTSSynthesisOptions()
        speed = synthesis_options.speed if synthesis_options.speed is not None else self._speed_factor

        body = json.dumps({
            "text": text.strip(),
            "text_lang": self._text_lang,
            "ref_audio_path": self._ref_audio_path,
            "prompt_lang": self._prompt_lang,
            "prompt_text": self._prompt_text,
            "text_split_method": "cut5",
            "batch_size": 1,
            "media_type": "wav",
            "streaming_mode": False,
            "speed_factor": speed,
        }, ensure_ascii=False).encode("utf-8")

        http_req = request.Request(
            url=self._api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_req, timeout=self._timeout) as resp:
                audio_bytes = resp.read()
                return SynthesizedSpeech(
                    audio_bytes=audio_bytes,
                    content_type="audio/wav",
                    file_extension="wav",
                    provider=self.name,
                    voice=self._default_voice,
                )
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GPT-SoVITS TTS failed: status={exc.code}, detail={detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"GPT-SoVITS TTS network error: {exc.reason}") from exc

    async def close(self) -> None:
        pass
