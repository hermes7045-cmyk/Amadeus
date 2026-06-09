from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from echobot.asr import OpenAITranscriptionsASRProvider, build_default_asr_service


class _FakeTranscriptions:
    def __init__(self, captured: dict[str, object], response) -> None:
        self._captured = captured
        self._response = response

    def create(self, **kwargs):
        file_obj = kwargs.get("file")
        if file_obj is not None:
            self._captured["file_name"] = getattr(file_obj, "name", "")
            self._captured["file_bytes"] = file_obj.getvalue()
        self._captured["create_kwargs"] = dict(kwargs)
        return self._response


class _FakeAudio:
    def __init__(self, captured: dict[str, object], response) -> None:
        self.transcriptions = _FakeTranscriptions(captured, response)


class _FakeOpenAIClient:
    def __init__(self, captured: dict[str, object], response) -> None:
        self.audio = _FakeAudio(captured, response)
        self._captured = captured

    def close(self) -> None:
        self._captured["closed"] = True


class OpenAITranscriptionsASRProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_samples_uses_openai_sdk_with_wav_file(self) -> None:
        captured: dict[str, object] = {}

        def fake_openai_constructor(**kwargs):
            captured["client_kwargs"] = kwargs
            return _FakeOpenAIClient(
                captured,
                {
                    "text": "hello world",
                    "language": "en",
                },
            )

        with patch(
            "echobot.asr.providers.openai_transcriptions.OpenAI",
            side_effect=fake_openai_constructor,
        ):
            provider = OpenAITranscriptionsASRProvider(
                sample_rate=16000,
                api_key="EMPTY",
                model="glm-asr",
                base_url="http://localhost:8000/v1",
                language="en",
                prompt="keep punctuation",
                temperature=0.2,
            )
            result = await provider.transcribe_samples([0.25, -0.25, 0.5])
            await provider.close()

        self.assertEqual("hello world", result.text)
        self.assertEqual("en", result.language)
        self.assertEqual(
            {
                "api_key": "EMPTY",
                "base_url": "http://localhost:8000/v1",
                "timeout": 60.0,
            },
            captured["client_kwargs"],
        )

        create_kwargs = captured["create_kwargs"]
        self.assertEqual("glm-asr", create_kwargs["model"])
        self.assertEqual("en", create_kwargs["language"])
        self.assertEqual("keep punctuation", create_kwargs["prompt"])
        self.assertEqual(0.2, create_kwargs["temperature"])

        self.assertEqual("audio.wav", captured["file_name"])
        wav_bytes = captured["file_bytes"]
        self.assertIn(b"RIFF", wav_bytes)
        self.assertIn(b"WAVE", wav_bytes)
        self.assertTrue(captured["closed"])

    async def test_transcribe_samples_accepts_plain_text_response(self) -> None:
        with patch(
            "echobot.asr.providers.openai_transcriptions.OpenAI",
            return_value=_FakeOpenAIClient({}, "plain transcript"),
        ):
            provider = OpenAITranscriptionsASRProvider(
                model="qwen3-asr",
                base_url="http://localhost:8000/v1",
            )
            result = await provider.transcribe_samples([0.1, 0.2])

        self.assertEqual("plain transcript", result.text)
        self.assertEqual("", result.language)

    async def test_transcribe_samples_joins_segment_text_when_text_is_missing(self) -> None:
        response = SimpleNamespace(
            text="",
            language="en",
            segments=[
                SimpleNamespace(text="hello "),
                SimpleNamespace(text="world"),
            ],
        )
        with patch(
            "echobot.asr.providers.openai_transcriptions.OpenAI",
            return_value=_FakeOpenAIClient({}, response),
        ):
            provider = OpenAITranscriptionsASRProvider(
                model="segment-model",
                base_url="http://localhost:8000/v1",
            )
            result = await provider.transcribe_samples([0.1, 0.2])

        self.assertEqual("hello world", result.text)
        self.assertEqual("en", result.language)

    async def test_status_snapshot_requires_real_api_key_for_official_openai_endpoint(self) -> None:
        with patch(
            "echobot.asr.providers.openai_transcriptions.OpenAI",
            return_value=_FakeOpenAIClient({}, {"text": "unused"}),
        ):
            provider = OpenAITranscriptionsASRProvider(
                model="gpt-4o-transcribe",
                base_url="https://api.openai.com/v1",
                api_key="EMPTY",
            )

        snapshot = await provider.status_snapshot()

        self.assertFalse(snapshot.available)
        self.assertEqual("missing", snapshot.state)
        self.assertIn("API_KEY", snapshot.detail)


class OpenAITranscriptionsFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_default_asr_service_can_select_openai_transcriptions_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            with patch(
                "echobot.asr.providers.openai_transcriptions.OpenAI",
                return_value=_FakeOpenAIClient({}, {"text": "unused"}),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "ECHOBOT_ASR_PROVIDER": "openai-transcriptions",
                        "ECHOBOT_ASR_OPENAI_API_KEY": "EMPTY",
                        "ECHOBOT_ASR_OPENAI_MODEL": "zai-org/GLM-ASR-Nano-2512",
                        "ECHOBOT_ASR_OPENAI_BASE_URL": "http://localhost:8000/v1",
                        "ECHOBOT_ASR_OPENAI_LANGUAGE": "en",
                        "ECHOBOT_ASR_OPENAI_PROMPT": "keep punctuation",
                        "ECHOBOT_ASR_OPENAI_TEMPERATURE": "0.1",
                        "ECHOBOT_ASR_SHERPA_AUTO_DOWNLOAD": "false",
                        "ECHOBOT_VAD_PROVIDER": "none",
                        "ECHOBOT_VAD_SILERO_AUTO_DOWNLOAD": "false",
                    },
                    clear=False,
                ):
                    service = build_default_asr_service(workspace)

            snapshot = await service.status_snapshot()

        self.assertTrue(snapshot.available)
        self.assertEqual("openai-transcriptions", snapshot.selected_asr_provider)
        self.assertEqual("", snapshot.selected_vad_provider)
        self.assertFalse(snapshot.always_listen_supported)
        self.assertIn(
            "openai-transcriptions",
            [item.name for item in snapshot.asr_providers],
        )
