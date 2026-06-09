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

from amadeus.channels.types import ChannelAddress, DeliveryTarget
from amadeus.gateway import (
    DEFAULT_DELIVERY_STORE_PATH,
    DEFAULT_ROUTE_SESSION_STORE_PATH,
    DeleteRouteSessionResult,
    DeliveryStore,
    RouteSessionStore,
    RouteSessionSummary,
)


# ---------------------------------------------------------------------------
# RouteSessionStore tests
# ---------------------------------------------------------------------------

class TestRouteSessionStore(unittest.TestCase):
    """Tests for RouteSessionStore CRUD operations and persistence."""

    def _make_route_key(self) -> str:
        return "test_channel__test_user__deadbeef"

    def _default_session_name(self, summary: RouteSessionSummary) -> str:
        """Return the auto-generated session name for comparison."""
        return summary.session_name

    def test_get_current_session_creates_first_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            summary = store.get_current_session(route_key)

            self.assertIsNotNone(summary)
            self.assertEqual("Session 1", summary.title)
            self.assertIn(route_key, summary.session_name)

    def test_list_sessions_empty_returns_one_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            sessions = store.list_sessions(route_key)

            self.assertEqual(1, len(sessions))
            self.assertEqual("Session 1", sessions[0].title)

    def test_create_multiple_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            first = store.get_current_session(route_key)
            second = store.create_session(route_key, title="Work")
            third = store.create_session(route_key, title="Personal")

            self.assertEqual("Session 1", first.title)
            self.assertEqual("Work", second.title)
            self.assertEqual("Personal", third.title)

            # Current should be the newly created one
            current = store.get_current_session(route_key)
            self.assertEqual(third.session_name, current.session_name)

    def test_list_sessions_after_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            # Auto-create first session, then create two more
            store.get_current_session(route_key)
            store.create_session(route_key, title="Work")
            store.create_session(route_key, title="Personal")

            sessions = store.list_sessions(route_key)

            self.assertEqual(3, len(sessions))
            titles = [s.title for s in sessions]
            self.assertIn("Session 1", titles)
            self.assertIn("Work", titles)
            self.assertIn("Personal", titles)

    def test_switch_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            first = store.get_current_session(route_key)
            second = store.create_session(route_key, title="Work")

            # Switch back to first session (index 2: ordered=[current, session_1])
            switched = store.switch_session(route_key, 2)
            self.assertEqual(first.session_name, switched.session_name)
            self.assertEqual(first.title, switched.title)

            # Verify current is now the first
            current = store.get_current_session(route_key)
            self.assertEqual(first.session_name, current.session_name)

    def test_switch_session_out_of_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            store.get_current_session(route_key)

            with self.assertRaises(ValueError):
                store.switch_session(route_key, 99)

    def test_rename_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            store.get_current_session(route_key)
            renamed = store.rename_current_session(route_key, "My Chat")

            self.assertEqual("My Chat", renamed.title)

            # Verify persisted
            current = store.get_current_session(route_key)
            self.assertEqual("My Chat", current.title)

    def test_rename_empty_title_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            store.get_current_session(route_key)

            with self.assertRaises(ValueError):
                store.rename_current_session(route_key, "   ")

    def test_delete_current_session_with_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            first = store.get_current_session(route_key)
            second = store.create_session(route_key, title="Work")

            result = store.delete_current_session(route_key)

            self.assertEqual(second.session_name, result.deleted.session_name)
            self.assertEqual(first.session_name, result.current.session_name)
            self.assertFalse(result.created_replacement)

            # Verify switching to deleted session no longer works
            with self.assertRaises(ValueError):
                store.switch_session(route_key, 2)

    def test_delete_last_session_creates_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            first = store.get_current_session(route_key)

            result = store.delete_current_session(route_key)

            self.assertEqual(first.session_name, result.deleted.session_name)
            self.assertTrue(result.created_replacement)
            self.assertNotEqual(first.session_name, result.current.session_name)

    def test_persistence_across_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            route_key = self._make_route_key()

            # First session
            store = RouteSessionStore(path)
            first = store.get_current_session(route_key)
            store.create_session(route_key, title="Work")

            # Reload from disk
            store2 = RouteSessionStore(path)
            sessions = store2.list_sessions(route_key)

            self.assertEqual(2, len(sessions))
            titles = [s.title for s in sessions]
            self.assertIn("Session 1", titles)
            self.assertIn("Work", titles)

    def test_touch_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            summary = store.get_current_session(route_key)
            original_updated = summary.updated_at

            store.touch_session(route_key, summary.session_name)
            current = store.get_current_session(route_key)

            # updated_at should have changed
            self.assertNotEqual(original_updated, current.updated_at)

    def test_remove_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            first = store.get_current_session(route_key)
            second = store.create_session(route_key, title="Work")

            removed = store.remove_session(first.session_name)
            self.assertTrue(removed)

            sessions = store.list_sessions(route_key)
            self.assertEqual(1, len(sessions))
            self.assertEqual(second.session_name, sessions[0].session_name)

    def test_remove_non_existent_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            store.get_current_session(route_key)

            removed = store.remove_session("nonexistent_session_name")
            self.assertFalse(removed)

    def test_replace_session_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            summary = store.get_current_session(route_key)
            old_name = summary.session_name
            new_name = old_name + "_renamed"

            changed = store.replace_session_name(old_name, new_name)
            self.assertTrue(changed)

            # Old name should no longer exist
            current = store.get_current_session(route_key)
            self.assertEqual(new_name, current.session_name)

    def test_route_key_isolation(self) -> None:
        """Different route keys should have independent sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)

            key_a = "channel_a__user_a__abc123"
            key_b = "channel_b__user_b__def456"

            a_first = store.get_current_session(key_a)
            store.create_session(key_a, title="Chat A")
            b_first = store.get_current_session(key_b)
            store.create_session(key_b, title="Chat B")

            a_sessions = store.list_sessions(key_a)
            b_sessions = store.list_sessions(key_b)

            self.assertEqual(2, len(a_sessions))
            self.assertEqual(2, len(b_sessions))

    def test_short_id_property(self) -> None:
        summary = RouteSessionSummary(
            session_name="telegram__12345__session__abc12345",
            title="Test",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        self.assertEqual("abc12345", summary.short_id)

    def test_short_id_no_delimiter(self) -> None:
        summary = RouteSessionSummary(
            session_name="short-name",
            title="Test",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        # Falls back to last 8 chars of "short-name"
        self.assertEqual("ort-name", summary.short_id)

    def test_route_session_summary_roundtrip(self) -> None:
        summary = RouteSessionSummary(
            session_name="test__session__abc123",
            title="My Session",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        data = summary.to_dict()
        restored = RouteSessionSummary.from_dict(data)
        self.assertEqual(summary.session_name, restored.session_name)
        self.assertEqual(summary.title, restored.title)

    def test_delete_current_session_via_delete_current(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route_sessions.json"
            store = RouteSessionStore(path)
            route_key = self._make_route_key()

            first = store.get_current_session(route_key)
            second = store.create_session(route_key, title="Second")
            third = store.create_session(route_key, title="Third")

            # Switch back to second, then delete current
            store.switch_session(route_key, 2)
            result = store.delete_current_session(route_key)

            self.assertEqual(second.session_name, result.deleted.session_name)
            # The current session should be the third (top of remaining ordered list)
            current = store.get_current_session(route_key)
            self.assertEqual(third.session_name, current.session_name)


# ---------------------------------------------------------------------------
# DeliveryStore tests
# ---------------------------------------------------------------------------

class TestDeliveryStore(unittest.TestCase):
    """Tests for DeliveryStore — persisting delivery targets."""

    def test_remember_and_get_session_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)
            address = ChannelAddress(channel="qq", chat_id="user_123")

            store.remember("session_1", address, {"message_id": "42"})

            target = store.get_session_target("session_1")
            self.assertIsNotNone(target)
            if target is not None:
                self.assertEqual("qq", target.address.channel)
                self.assertEqual("user_123", target.address.chat_id)
                self.assertEqual("42", target.metadata.get("message_id"))

    def test_get_latest_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)

            store.remember(
                "session_1",
                ChannelAddress(channel="qq", chat_id="user_1"),
                {"message_id": "1"},
            )
            store.remember(
                "session_2",
                ChannelAddress(channel="telegram", chat_id="user_2"),
                {"message_id": "2"},
            )

            latest = store.get_latest_target()
            self.assertIsNotNone(latest)
            if latest is not None:
                self.assertEqual("telegram", latest.address.channel)
                self.assertEqual("user_2", latest.address.chat_id)
                self.assertEqual("2", latest.metadata.get("message_id"))

    def test_forget_removes_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)
            address = ChannelAddress(channel="qq", chat_id="user_123")

            store.remember("session_1", address)
            store.forget("session_1")

            target = store.get_session_target("session_1")
            self.assertIsNone(target)

    def test_forget_non_existent_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)

            # Should not raise
            store.forget("nonexistent_session")

    def test_persistence_across_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            address = ChannelAddress(channel="qq", chat_id="user_123")

            store = DeliveryStore(path)
            store.remember("session_1", address, {"message_id": "42"})

            store2 = DeliveryStore(path)
            target = store2.get_session_target("session_1")

            self.assertIsNotNone(target)
            if target is not None:
                self.assertEqual("qq", target.address.channel)
                self.assertEqual("42", target.metadata.get("message_id"))

    def test_forget_updates_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)

            store.remember("session_1", ChannelAddress(channel="qq", chat_id="u1"))
            store.remember("session_2", ChannelAddress(channel="tg", chat_id="u2"))
            store.forget("session_2")

            latest = store.get_latest_target()
            self.assertIsNotNone(latest)
            if latest is not None:
                self.assertEqual("qq", latest.address.channel)

    def test_replace_session_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)
            address = ChannelAddress(channel="qq", chat_id="user_123")

            store.remember("old_name", address)
            changed = store.replace_session_name("old_name", "new_name")
            self.assertTrue(changed)

            self.assertIsNone(store.get_session_target("old_name"))
            self.assertIsNotNone(store.get_session_target("new_name"))

    def test_replace_non_existent_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)

            changed = store.replace_session_name("nonexistent", "new_name")
            self.assertFalse(changed)

    def test_get_latest_target_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "delivery.json"
            store = DeliveryStore(path)

            latest = store.get_latest_target()
            self.assertIsNone(latest)


# ---------------------------------------------------------------------------
# RouteSessionSummary data class tests
# ---------------------------------------------------------------------------

class TestRouteSessionSummary(unittest.TestCase):
    """Tests for RouteSessionSummary properties and serialization."""

    def test_to_dict(self) -> None:
        summary = RouteSessionSummary(
            session_name="route__session__abc",
            title="Test",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T01:00:00",
        )
        data = summary.to_dict()
        self.assertEqual(summary.session_name, data["session_name"])
        self.assertEqual(summary.title, data["title"])
        self.assertEqual(summary.created_at, data["created_at"])
        self.assertEqual(summary.updated_at, data["updated_at"])

    def test_from_dict_with_defaults(self) -> None:
        summary = RouteSessionSummary.from_dict({
            "session_name": "test__session__abc",
        })
        self.assertEqual("test__session__abc", summary.session_name)
        # Title defaults to "Session" for empty string
        self.assertEqual("Session", summary.title)
        self.assertIn("T", summary.created_at)  # ISO format datetime

    def test_from_dict_preserves_all_fields(self) -> None:
        data = {
            "session_name": "route__session__xyz",
            "title": "Custom Title",
            "created_at": "2024-06-15T10:30:00+00:00",
            "updated_at": "2024-06-15T11:00:00+00:00",
        }
        summary = RouteSessionSummary.from_dict(data)
        self.assertEqual("route__session__xyz", summary.session_name)
        self.assertEqual("Custom Title", summary.title)
        self.assertEqual("2024-06-15T10:30:00+00:00", summary.created_at)
        self.assertEqual("2024-06-15T11:00:00+00:00", summary.updated_at)


# ---------------------------------------------------------------------------
# DeliveryTarget data class tests
# ---------------------------------------------------------------------------

class TestDeliveryTarget(unittest.TestCase):
    """Tests for DeliveryTarget serialization."""

    def test_roundtrip(self) -> None:
        address = ChannelAddress(channel="test", chat_id="123", user_id="u1")
        target = DeliveryTarget(
            address=address,
            metadata={"mid": "abc"},
        )
        data = target.to_dict()
        restored = DeliveryTarget.from_dict(data)
        self.assertEqual(target.address.channel, restored.address.channel)
        self.assertEqual(target.address.chat_id, restored.address.chat_id)
        self.assertEqual(target.address.user_id, restored.address.user_id)
        self.assertEqual(target.metadata, restored.metadata)


if __name__ == "__main__":
    unittest.main()
