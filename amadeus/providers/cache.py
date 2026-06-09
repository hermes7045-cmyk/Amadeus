from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_DISABLE_ENV_VAR = "ECHOBOT_DISABLE_RESPONSE_CACHE"


class ResponseCache:
    """Thread-safe TTL cache with LRU eviction for LLM responses.

    Cache key is a SHA256 hash of the JSON-serialised payload.
    Entries expire after *ttl* seconds (default 300 = 5 min).
    At most *max_size* entries are kept; the least-recently used
    entry is evicted when the limit is exceeded.

    The cache can be disabled at runtime via the environment
    variable ``ECHOBOT_DISABLE_RESPONSE_CACHE=true``.
    """

    def __init__(self, max_size: int = 100, ttl: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    # --- enabled check (read every call so env var can be toggled at runtime) ---

    @property
    def enabled(self) -> bool:
        val = os.environ.get(_DISABLE_ENV_VAR, "").strip().lower()
        return val not in ("true", "1", "yes")

    # --- key helpers ---

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        """Deterministic SHA256 key from a request payload."""
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def make_system_key(system_text: str) -> str:
        """SHA256 key from system message text only (for prompt caching)."""
        return hashlib.sha256(system_text.encode("utf-8")).hexdigest()

    # --- core API ---

    def get(self, key: str) -> Any | None:
        """Return cached value, or *None* if missing/expired/disabled."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._cache[key]
                return None
            # LRU: move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key* with the configured TTL."""
        if not self.enabled:
            return
        with self._lock:
            self._cache[key] = (time.monotonic() + self._ttl, value)
            self._cache.move_to_end(key)
            self._evict_lru()

    def invalidate(self, key: str) -> None:
        """Remove a single entry (no-op if missing)."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()

    # --- internals ---

    def _evict_lru(self) -> None:
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # Least-recently used is at front

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
