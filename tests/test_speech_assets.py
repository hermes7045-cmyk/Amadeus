from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import echobot.speech_assets as speech_assets


class FakeDownloadResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self.headers = headers or {}

    def __enter__(self) -> FakeDownloadResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, _chunk_size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class SpeechAssetsTests(unittest.TestCase):
    def test_download_file_prints_progress_with_known_size(self) -> None:
        response = FakeDownloadResponse(
            [b"ab", b"cdef", b""],
            headers={"Content-Length": "6"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "model.bin"
            stderr = io.StringIO()

            with patch("echobot.speech_assets.urlopen", return_value=response):
                with patch("sys.stderr", stderr):
                    speech_assets.download_file(
                        "https://example.com/model.bin",
                        destination,
                        timeout_seconds=1.0,
                        progress_label="Test model",
                    )

            output = stderr.getvalue()
            downloaded_bytes = destination.read_bytes()

        self.assertEqual(b"abcdef", downloaded_bytes)
        self.assertIn("[download] Test model: starting", output)
        self.assertIn("100.0%", output)
        self.assertIn("completed", output)

    def test_download_file_prints_progress_when_size_is_unknown(self) -> None:
        response = FakeDownloadResponse([b"abc", b"def", b""])

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "model.bin"
            stderr = io.StringIO()

            with patch("echobot.speech_assets.urlopen", return_value=response):
                with patch("echobot.speech_assets._DOWNLOAD_PROGRESS_UPDATE_INTERVAL_SECONDS", 0.0):
                    with patch("sys.stderr", stderr):
                        speech_assets.download_file(
                            "https://example.com/model.bin",
                            destination,
                            timeout_seconds=1.0,
                            progress_label="Unknown model",
                        )

            output = stderr.getvalue()
            downloaded_bytes = destination.read_bytes()

        self.assertEqual(b"abcdef", downloaded_bytes)
        self.assertIn("size unknown", output)
        self.assertIn("downloaded", output)
        self.assertIn("completed", output)
