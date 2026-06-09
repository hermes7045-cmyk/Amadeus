from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from unittest.mock import patch

from echobot.tts import (
    TTSProvider,
    TTSProviderStatus,
    TTSSynthesisOptions,
    TTSService,
    build_default_kokoro_tts_provider,
    build_default_openai_compatible_tts_provider,
    build_default_tts_service,
)
from echobot.tts.providers.edge import EdgeTTSProvider
from echobot.tts.providers.kokoro import (
    DEFAULT_KOKORO_VOICE,
    KokoroTTSProvider,
    kokoro_voice_options,
    speaker_id_for_voice,
)
from echobot.tts.providers.openai_compatible import OpenAICompatibleTTSProvider


class _FakeSpeechResponse:
    def __init__(
        self,
        *,
        content: bytes,
        content_type: str = "audio/wav",
    ) -> None:
        self._content = content
        self.response = SimpleNamespace(headers={"content-type": content_type})
        self.closed = False

    def read(self) -> bytes:
        return self._content

    def close(self) -> None:
        self.closed = True


class _FakeSpeechResource:
    def __init__(self, captured: dict[str, object], response: _FakeSpeechResponse) -> None:
        self._captured = captured
        self._response = response

    def create(self, **kwargs):
        self._captured["create_kwargs"] = dict(kwargs)
        return self._response


class _FakeAudioResource:
    def __init__(self, captured: dict[str, object], response: _FakeSpeechResponse) -> None:
        self.speech = _FakeSpeechResource(captured, response)


class _FakeOpenAIClient:
    def __init__(self, captured: dict[str, object], response: _FakeSpeechResponse) -> None:
        self.audio = _FakeAudioResource(captured, response)
        self._captured = captured

    def close(self) -> None:
        self._captured["closed"] = True


class _FakeUrlopenResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _CapturingTTSProvider(TTSProvider):
    name = "capturing"
    label = "Capturing TTS"

    def __init__(
        self,
        *,
        status: TTSProviderStatus | None = None,
    ) -> None:
        self.captured_text = ""
        self.captured_options: TTSSynthesisOptions | None = None
        self._status = status or TTSProviderStatus(
            name=self.name,
            label=self.label,
            available=True,
        )

    @property
    def default_voice(self) -> str:
        return "capturing-default"

    def status(self) -> TTSProviderStatus:
        return self._status

    async def synthesize(
        self,
        *,
        text: str,
        options: TTSSynthesisOptions | None = None,
    ) -> object:
        self.captured_text = text
        self.captured_options = options
        return SimpleNamespace(
            audio_bytes=b"unused",
            content_type="audio/wav",
            file_extension="wav",
            provider=self.name,
            voice=(options.voice if options else self.default_voice) or self.default_voice,
        )


class TTSFactoryTests(unittest.TestCase):
    def test_provider_modules_are_grouped_under_providers_package(self) -> None:
        self.assertEqual("echobot.tts.providers.edge", EdgeTTSProvider.__module__)
        self.assertEqual(
            "echobot.tts.providers.kokoro.provider",
            KokoroTTSProvider.__module__,
        )
        self.assertEqual(
            "echobot.tts.providers.openai_compatible",
            OpenAICompatibleTTSProvider.__module__,
        )

    def test_build_default_tts_service_registers_edge_kokoro_and_openai(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = build_default_tts_service(Path(temp_dir))

        self.assertEqual("edge", service.default_provider)
        self.assertEqual(
            ["edge", "kokoro", "openai-compatible"],
            service.provider_names(),
        )
        self.assertEqual("zh-CN-XiaoxiaoNeural", service.default_voice_for("edge"))
        self.assertEqual(DEFAULT_KOKORO_VOICE, service.default_voice_for("kokoro"))
        self.assertEqual("alloy", service.default_voice_for("openai-compatible"))

    def test_build_default_kokoro_tts_provider_reads_default_voice_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "ECHOBOT_TTS_KOKORO_AUTO_DOWNLOAD": "false",
                    "ECHOBOT_TTS_KOKORO_DEFAULT_VOICE": "af_maple",
                },
                clear=False,
            ):
                provider = build_default_kokoro_tts_provider(workspace)

        self.assertEqual("af_maple", provider.default_voice)

    def test_build_default_kokoro_tts_provider_falls_back_for_unknown_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "ECHOBOT_TTS_KOKORO_AUTO_DOWNLOAD": "false",
                    "ECHOBOT_TTS_KOKORO_DEFAULT_VOICE": "missing-voice",
                },
                clear=False,
            ):
                provider = build_default_kokoro_tts_provider(workspace)

        self.assertEqual(DEFAULT_KOKORO_VOICE, provider.default_voice)

    def test_build_default_tts_service_reads_default_provider_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "ECHOBOT_TTS_PROVIDER": "openai-compatible",
                },
                clear=False,
            ):
                service = build_default_tts_service(workspace)

        self.assertEqual("openai-compatible", service.default_provider)

    def test_build_default_openai_compatible_tts_provider_reads_voice_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ECHOBOT_TTS_OPENAI_DEFAULT_VOICE": "vivian",
                "ECHOBOT_TTS_OPENAI_VOICES": "vivian, ryan",
            },
            clear=False,
        ):
            provider = build_default_openai_compatible_tts_provider()

        self.assertEqual("vivian", provider.default_voice)


class TTSServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_normalizes_markdown_before_synthesizing(self) -> None:
        provider = _CapturingTTSProvider()
        service = TTSService({"capturing": provider}, default_provider="capturing")

        await service.synthesize(
            text="# Title\n- hello [world](https://example.com) `code` 😊",
        )

        self.assertEqual("Title hello world code", provider.captured_text)

    async def test_service_parses_shared_speed_option(self) -> None:
        provider = _CapturingTTSProvider()
        service = TTSService({"capturing": provider}, default_provider="capturing")

        await service.synthesize(
            text="Hello",
            rate="50%",
            volume="loud",
            pitch="+5Hz",
        )

        self.assertIsNotNone(provider.captured_options)
        self.assertEqual(0.5, provider.captured_options.speed)
        self.assertEqual("loud", provider.captured_options.volume)
        self.assertEqual("+5Hz", provider.captured_options.pitch)

    async def test_service_does_not_block_provider_synthesis_on_status(self) -> None:
        provider = _CapturingTTSProvider(
            status=TTSProviderStatus(
                name="capturing",
                label="Capturing TTS",
                available=False,
                state="missing",
                detail="Will prepare on first synthesis",
            )
        )
        service = TTSService({"capturing": provider}, default_provider="capturing")

        await service.synthesize(text="Hello")

        self.assertEqual("Hello", provider.captured_text)

    async def test_service_rejects_invalid_rate(self) -> None:
        provider = _CapturingTTSProvider()
        service = TTSService({"capturing": provider}, default_provider="capturing")

        with self.assertRaises(ValueError):
            await service.synthesize(text="Hello", rate="fast")


class OpenAICompatibleTTSProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesize_uses_openai_sdk_and_returns_audio(self) -> None:
        captured: dict[str, object] = {}
        response = _FakeSpeechResponse(content=b"fake-wav-bytes")

        def fake_openai_constructor(**kwargs):
            captured["client_kwargs"] = kwargs
            return _FakeOpenAIClient(captured, response)

        with patch(
            "echobot.tts.providers.openai_compatible.OpenAI",
            side_effect=fake_openai_constructor,
        ):
            provider = OpenAICompatibleTTSProvider(
                api_key="EMPTY",
                model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                base_url="http://localhost:8091/v1",
                default_voice="vivian",
                response_format="wav",
                instructions="Speak warmly",
                extra_body={
                    "language": "English",
                    "task_type": "CustomVoice",
                },
            )
            speech = await provider.synthesize(
                text="Hello, how are you?",
                options=TTSSynthesisOptions(speed=1.25),
            )
            await provider.close()

        self.assertEqual(
            {
                "api_key": "EMPTY",
                "base_url": "http://localhost:8091/v1",
                "timeout": 60.0,
            },
            captured["client_kwargs"],
        )
        self.assertEqual(
            {
                "input": "Hello, how are you?",
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "voice": "vivian",
                "instructions": "Speak warmly",
                "response_format": "wav",
                "speed": 1.25,
                "extra_body": {
                    "language": "English",
                    "task_type": "CustomVoice",
                },
            },
            captured["create_kwargs"],
        )
        self.assertEqual(b"fake-wav-bytes", speech.audio_bytes)
        self.assertEqual("audio/wav", speech.content_type)
        self.assertEqual("wav", speech.file_extension)
        self.assertEqual("openai-compatible", speech.provider)
        self.assertEqual("vivian", speech.voice)
        self.assertTrue(response.closed)
        self.assertTrue(captured["closed"])

    async def test_list_voices_reads_audio_voices_endpoint(self) -> None:
        with patch(
            "echobot.tts.providers.openai_compatible.OpenAI",
            return_value=_FakeOpenAIClient({}, _FakeSpeechResponse(content=b"unused")),
        ):
            with patch(
                "echobot.tts.providers.openai_compatible.request.urlopen",
                return_value=_FakeUrlopenResponse(
                    '{"voices":["aiden","vivian","ryan"]}',
                ),
            ):
                provider = OpenAICompatibleTTSProvider(
                    api_key="EMPTY",
                    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    base_url="http://localhost:8091/v1",
                    default_voice="vivian",
                )
                voices = await provider.list_voices()

        self.assertEqual(["vivian", "aiden", "ryan"], [voice.short_name for voice in voices])

    async def test_list_voices_falls_back_to_default_voice_when_endpoint_is_unavailable(self) -> None:
        with patch(
            "echobot.tts.providers.openai_compatible.OpenAI",
            return_value=_FakeOpenAIClient({}, _FakeSpeechResponse(content=b"unused")),
        ):
            with patch(
                "echobot.tts.providers.openai_compatible.request.urlopen",
                side_effect=URLError("refused"),
            ):
                provider = OpenAICompatibleTTSProvider(
                    api_key="EMPTY",
                    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    base_url="http://localhost:8091/v1",
                    default_voice="vivian",
                )
                voices = await provider.list_voices()

        self.assertEqual(["vivian"], [voice.short_name for voice in voices])

    async def test_list_voices_uses_configured_voice_names_without_network(self) -> None:
        with patch(
            "echobot.tts.providers.openai_compatible.OpenAI",
            return_value=_FakeOpenAIClient({}, _FakeSpeechResponse(content=b"unused")),
        ):
            with patch(
                "echobot.tts.providers.openai_compatible.request.urlopen",
            ) as mocked_urlopen:
                provider = OpenAICompatibleTTSProvider(
                    api_key="EMPTY",
                    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    base_url="http://localhost:8091/v1",
                    default_voice="vivian",
                    voices=["vivian", "ryan"],
                )
                voices = await provider.list_voices()

        mocked_urlopen.assert_not_called()
        self.assertEqual(["vivian", "ryan"], [voice.short_name for voice in voices])


class KokoroVoiceTests(unittest.TestCase):
    def test_speaker_id_for_voice_accepts_name_and_numeric_id(self) -> None:
        self.assertEqual(3, speaker_id_for_voice("zf_001"))
        self.assertEqual(3, speaker_id_for_voice("3"))

    def test_speaker_id_for_voice_rejects_unknown_voice(self) -> None:
        with self.assertRaises(ValueError):
            speaker_id_for_voice("missing-voice")

    def test_kokoro_voice_options_expose_known_voice(self) -> None:
        voices = kokoro_voice_options()
        zf_voice = next(
            voice
            for voice in voices
            if voice.short_name == DEFAULT_KOKORO_VOICE
        )

        self.assertEqual("zh-CN", zf_voice.locale)
        self.assertEqual("Female", zf_voice.gender)

    def test_kokoro_list_voices_does_not_start_prepare_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = KokoroTTSProvider(
                Path(temp_dir),
                auto_download=True,
            )

            voices = asyncio.run(provider.list_voices())

        self.assertTrue(voices)
        self.assertIsNone(provider._prepare_task)
