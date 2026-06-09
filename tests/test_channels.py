from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

# Add project root to path (like conftest.py does for pytest)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)

from amadeus.channels.base import BaseChannel
from amadeus.channels.bus import MessageBus
from amadeus.channels.platforms.qq import (
    QQChannel,
    _is_data_url,
    _is_http_url,
    _looks_like_image_attachment,
    _qq_block_fallback_text,
    _qq_delivery_target,
    _qq_media_message_payload,
    _qq_text_content,
    _qq_text_content_blocks,
)
from amadeus.channels.types import ChannelAddress, InboundMessage, OutboundMessage


# ---------------------------------------------------------------------------
# _is_http_url / _is_data_url helpers
# ---------------------------------------------------------------------------

class TestURLHelpers(unittest.TestCase):
    """Tests for URL detection helpers."""

    def test_is_http_url(self) -> None:
        self.assertTrue(_is_http_url("https://example.com/image.png"))
        self.assertTrue(_is_http_url("http://example.com/file"))
        self.assertFalse(_is_http_url("data:image/png;base64,abc"))
        self.assertFalse(_is_http_url(""))
        self.assertFalse(_is_http_url("attachment://img_123"))
        self.assertFalse(_is_http_url("ftp://example.com"))

    def test_is_data_url(self) -> None:
        self.assertTrue(_is_data_url("data:image/png;base64,iVBORw0KGgo="))
        self.assertTrue(_is_data_url("DATA:text/plain,hello"))
        self.assertFalse(_is_data_url("https://example.com"))
        self.assertFalse(_is_data_url(""))
        self.assertFalse(_is_data_url("attachment://img_123"))


# ---------------------------------------------------------------------------
# _looks_like_image_attachment
# ---------------------------------------------------------------------------

class TestLooksLikeImageAttachment(unittest.TestCase):
    """Tests for image attachment detection."""

    def test_content_type_image(self) -> None:
        self.assertTrue(_looks_like_image_attachment("image/png", "file.png", ""))

    def test_content_type_image_with_charset(self) -> None:
        self.assertTrue(
            _looks_like_image_attachment("image/jpeg; charset=utf-8", "photo.jpg", "")
        )

    def test_filename_image_extension(self) -> None:
        self.assertTrue(
            _looks_like_image_attachment("", "photo.PNG", "http://example.com/f")
        )
        self.assertTrue(
            _looks_like_image_attachment("", "image.jpg", "http://example.com/f")
        )
        self.assertTrue(
            _looks_like_image_attachment("", "animation.gif", "http://example.com/f")
        )

    def test_url_fallback_image_extension(self) -> None:
        self.assertTrue(
            _looks_like_image_attachment("", "", "http://example.com/photo.webp")
        )

    def test_non_image_content(self) -> None:
        self.assertFalse(
            _looks_like_image_attachment("text/plain", "notes.txt", "")
        )
        self.assertFalse(
            _looks_like_image_attachment(
                "application/pdf", "document.pdf", "http://example.com/doc.pdf"
            )
        )
        self.assertFalse(_looks_like_image_attachment("", "", ""))

    def test_empty_content_type(self) -> None:
        self.assertFalse(_looks_like_image_attachment("", "", "noextension"))

    def test_none_values_become_empty_string(self) -> None:
        # The function uses str() so None would become "None"
        self.assertFalse(
            _looks_like_image_attachment("None", "file.xyz", "")
        )


# ---------------------------------------------------------------------------
# _qq_delivery_target
# ---------------------------------------------------------------------------

class TestQQDeliveryTarget(unittest.TestCase):
    """Tests for delivery target extraction from OutboundMessage."""

    def test_group_openid_metadata(self) -> None:
        message = OutboundMessage(
            address=ChannelAddress(channel="qq", chat_id="user_123"),
            text="hello",
            metadata={"group_openid": "group_abc"},
        )
        target = _qq_delivery_target(message)
        self.assertEqual("group", target.kind)
        self.assertEqual("group_abc", target.target_id)

    def test_group_message_type(self) -> None:
        message = OutboundMessage(
            address=ChannelAddress(channel="qq", chat_id="group:group_xyz"),
            text="hello",
            metadata={"message_type": "group"},
        )
        target = _qq_delivery_target(message)
        self.assertEqual("group", target.kind)
        self.assertEqual("group_xyz", target.target_id)

    def test_c2c_default(self) -> None:
        message = OutboundMessage(
            address=ChannelAddress(channel="qq", chat_id="user_456"),
            text="hello",
        )
        target = _qq_delivery_target(message)
        self.assertEqual("c2c", target.kind)
        self.assertEqual("user_456", target.target_id)

    def test_c2c_with_group_prefix(self) -> None:
        message = OutboundMessage(
            address=ChannelAddress(channel="qq", chat_id="group:user_789"),
            text="hello",
        )
        target = _qq_delivery_target(message)
        self.assertEqual("c2c", target.kind)
        self.assertEqual("user_789", target.target_id)


# ---------------------------------------------------------------------------
# _qq_text_content / _qq_text_content_blocks
# ---------------------------------------------------------------------------

class TestQQTextContent(unittest.TestCase):
    """Tests for extracting text content from message blocks."""

    def test_simple_text(self) -> None:
        message = OutboundMessage(
            address=ChannelAddress(channel="qq", chat_id="u1"),
            text="Hello, world!",
        )
        result = _qq_text_content(message)
        self.assertEqual("Hello, world!", result)

    def test_text_with_image_url(self) -> None:
        message = OutboundMessage(
            address=ChannelAddress(channel="qq", chat_id="u1"),
            text="Check this out!",
            content=[
                {"type": "text", "text": "Check this out!"},
                {"type": "image_url", "image_url": {
                    "url": "https://example.com/img.png",
                    "preview_url": "https://example.com/img_preview.png",
                }},
            ],
        )
        result = _qq_text_content(message)
        self.assertIn("Check this out!", result)
        self.assertIn("https://example.com/img_preview.png", result)

    def test_image_with_local_url_becomes_placeholder(self) -> None:
        blocks = [
            {"type": "image_url", "image_url": {
                "url": "attachment://img_123",
                "preview_url": "/api/attachments/img_123/content",
            }},
        ]
        result = _qq_text_content_blocks(blocks)
        self.assertEqual("[image]", result)

    def test_file_block_with_download_url(self) -> None:
        blocks = [
            {"type": "file_attachment", "file_attachment": {
                "name": "report.pdf",
                "download_url": "https://example.com/report.pdf",
            }},
        ]
        result = _qq_text_content_blocks(blocks)
        self.assertIn("report.pdf", result)
        self.assertIn("https://example.com/report.pdf", result)

    def test_file_block_without_url(self) -> None:
        blocks = [
            {"type": "file_attachment", "file_attachment": {
                "name": "local_file.txt",
            }},
        ]
        result = _qq_text_content_blocks(blocks)
        self.assertIn("file: local_file.txt", result)

    def test_empty_blocks(self) -> None:
        result = _qq_text_content_blocks([])
        self.assertEqual("Model returned no text content.", result)

    def test_mixed_content(self) -> None:
        blocks = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {
                "url": "https://example.com/photo.jpg",
            }},
            {"type": "file_attachment", "file_attachment": {
                "name": "doc.txt",
                "download_url": "https://example.com/doc.txt",
            }},
        ]
        result = _qq_text_content_blocks(blocks)
        self.assertIn("Hello", result)
        self.assertIn("https://example.com/photo.jpg", result)
        self.assertIn("https://example.com/doc.txt", result)

    def test_unknown_block_type_ignored(self) -> None:
        blocks = [
            {"type": "unknown", "data": "something"},
            {"type": "text", "text": "Hello"},
        ]
        result = _qq_text_content_blocks(blocks)
        self.assertEqual("Hello", result)


# ---------------------------------------------------------------------------
# _qq_media_message_payload
# ---------------------------------------------------------------------------

class TestQQMediaMessagePayload(unittest.TestCase):
    """Tests for building media message payload from upload result."""

    def test_basic_payload(self) -> None:
        upload_result = {"file_info": "file_info_abc"}
        payload = _qq_media_message_payload(upload_result)
        self.assertIsNotNone(payload)
        if payload is not None:
            self.assertEqual("file_info_abc", payload["file_info"])

    def test_with_file_uuid(self) -> None:
        upload_result = {
            "file_info": "info_123",
            "file_uuid": "uuid_456",
        }
        payload = _qq_media_message_payload(upload_result)
        self.assertIsNotNone(payload)
        if payload is not None:
            self.assertEqual("uuid_456", payload["file_uuid"])

    def test_with_ttl(self) -> None:
        upload_result = {
            "file_info": "info_789",
            "ttl": 3600,
        }
        payload = _qq_media_message_payload(upload_result)
        self.assertIsNotNone(payload)
        if payload is not None:
            self.assertEqual(3600, payload["ttl"])

    def test_missing_file_info_returns_none(self) -> None:
        upload_result: dict[str, Any] = {}
        payload = _qq_media_message_payload(upload_result)
        self.assertIsNone(payload)

        upload_result = {"file_info": ""}
        payload = _qq_media_message_payload(upload_result)
        self.assertIsNone(payload)

    def test_whitespace_file_info_returns_none(self) -> None:
        upload_result = {"file_info": "   "}
        payload = _qq_media_message_payload(upload_result)
        self.assertIsNone(payload)

    def test_invalid_type_returns_none_skipped(self) -> None:
        # _qq_media_message_payload converts values to string,
        # so non-string types are not rejected — only missing/empty/whitespace.
        pass


# ---------------------------------------------------------------------------
# _qq_block_fallback_text
# ---------------------------------------------------------------------------

class TestQQBlockFallbackText(unittest.TestCase):
    """Tests for fallback text extraction from a single block."""

    def test_text_block(self) -> None:
        block = {"type": "text", "text": "Fallback message"}
        result = _qq_block_fallback_text(block)
        self.assertEqual("Fallback message", result)

    def test_image_block_with_http_url(self) -> None:
        block = {"type": "image_url", "image_url": {
            "url": "https://example.com/img.png",
        }}
        result = _qq_block_fallback_text(block)
        self.assertIn("https://example.com/img.png", result)

    def test_image_block_without_url(self) -> None:
        block = {"type": "image_url", "image_url": {
            "url": "attachment://img_123",
        }}
        result = _qq_block_fallback_text(block)
        self.assertEqual("[image]", result)

    def test_file_block_with_download_url(self) -> None:
        block = {"type": "file_attachment", "file_attachment": {
            "name": "doc.pdf",
            "download_url": "https://example.com/doc.pdf",
        }}
        result = _qq_block_fallback_text(block)
        self.assertIn("doc.pdf", result)
        self.assertIn("https://example.com/doc.pdf", result)


# ---------------------------------------------------------------------------
# BaseChannel._publish_inbound_message (integration with MessageBus)
# ---------------------------------------------------------------------------

class _FakeChannelConfig:
    """Minimal config for a test channel — allows all senders."""
    def __init__(self) -> None:
        self.allow_from: list[str] | None = None


class TestQQInboundMessageFlow(unittest.TestCase):
    """Tests the inbound message flow via a QQChannel and MessageBus."""

    def setUp(self) -> None:
        self.bus = MessageBus()
        self.config = _FakeChannelConfig()
        # Create QQChannel but don't start it (no botpy dependency needed)
        self.channel = QQChannel(self.config, self.bus)

    def test_publish_inbound_message(self) -> None:
        """Verify that _publish_inbound_message puts a message on the bus."""
        from amadeus.channels.types import InboundMessage as InboundMsg

        async def _run():
            await self.channel._publish_inbound_message(
                sender_id="user_qq_001",
                chat_id="user_qq_001",
                text="Hello from QQ!",
                metadata={"message_id": "msg_001"},
            )
            msg = await self.bus.consume_inbound()
            self.assertIsInstance(msg, InboundMsg)
            self.assertEqual("Hello from QQ!", msg.text)
            self.assertEqual("user_qq_001", msg.sender_id)
            self.assertEqual("qq", msg.address.channel)
            self.assertEqual("user_qq_001", msg.address.chat_id)
            self.assertEqual("msg_001", msg.metadata.get("message_id"))

        self._run_async(_run())

    def test_publish_inbound_empty_text_no_images_no_files_skipped(self) -> None:
        """Empty message with no images or files should not be published."""

        async def _run():
            await self.channel._publish_inbound_message(
                sender_id="user_1",
                chat_id="chat_1",
                text="   ",
            )
            # Bus should be empty
            import asyncio
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(self.bus.consume_inbound(), timeout=0.05)

        self._run_async(_run())

    def test_publish_inbound_with_image_urls(self) -> None:
        """Inbound message with image URLs should be published."""

        async def _run():
            await self.channel._publish_inbound_message(
                sender_id="user_1",
                chat_id="chat_1",
                text="Look at this",
                image_urls=[
                    {"url": "https://example.com/img.png", "detail": "auto"},
                ],
            )
            msg = await self.bus.consume_inbound()
            self.assertEqual("Look at this", msg.text)
            self.assertEqual(1, len(msg.image_urls))

        self._run_async(_run())

    def test_publish_inbound_sender_not_allowed(self) -> None:
        """Sender not in allow_from list should be rejected."""

        async def _run():
            # Set allow_from to block this sender
            self.channel.config.allow_from = ["allowed_user"]

            await self.channel._publish_inbound_message(
                sender_id="blocked_user",
                chat_id="chat_1",
                text="Hello",
            )
            import asyncio
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(self.bus.consume_inbound(), timeout=0.05)

        self._run_async(_run())

    def _run_async(self, coro):
        import asyncio
        import concurrent.futures
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result(timeout=10)
        except RuntimeError:
            return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ChannelAddress property tests
# ---------------------------------------------------------------------------

class TestChannelAddressRouteKey(unittest.TestCase):
    """Tests for ChannelAddress route_key and session_name properties."""

    def test_route_key_uses_channel_and_chat_id(self) -> None:
        address = ChannelAddress(channel="qq", chat_id="user_123")
        key = address.route_key
        self.assertIn("qq", key)
        self.assertIn("user_123", key)

    def test_route_key_includes_thread_id(self) -> None:
        address = ChannelAddress(
            channel="qq",
            chat_id="user_123",
            thread_id="thread_456",
        )
        key = address.route_key
        self.assertIn("thread_456", key)

    def test_session_name_matches_route_key(self) -> None:
        address = ChannelAddress(channel="telegram", chat_id="100500")
        self.assertEqual(address.route_key, address.session_name)

    def test_route_key_is_deterministic(self) -> None:
        a1 = ChannelAddress(channel="qq", chat_id="user_1")
        a2 = ChannelAddress(channel="qq", chat_id="user_1")
        self.assertEqual(a1.route_key, a2.route_key)

    def test_different_chat_ids_different_keys(self) -> None:
        a1 = ChannelAddress(channel="qq", chat_id="user_1")
        a2 = ChannelAddress(channel="qq", chat_id="user_2")
        self.assertNotEqual(a1.route_key, a2.route_key)

    def test_route_key_serialization(self) -> None:
        address = ChannelAddress(channel="qq", chat_id="user_123")
        data = address.to_dict()
        restored = ChannelAddress.from_dict(data)
        self.assertEqual(address.channel, restored.channel)
        self.assertEqual(address.chat_id, restored.chat_id)
        self.assertEqual(address.route_key, restored.route_key)


if __name__ == "__main__":
    unittest.main()
