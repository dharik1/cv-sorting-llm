"""
cache.py - Small on-disk JSON cache for LLM and embedding results.

Every model call is keyed by a SHA-256 hash of (prompt version, model, inputs), so re-running the
pipeline or refining a ranking never repeats an identical, slow model call. Use --no-cache to
disable it (for example when CVs must not be stored on disk at all).
"""
from __future__ import annotations

import hashlib
import json
import os


class JsonCache:
    """Stores JSON-serialisable values as one file per key inside a cache folder."""

    def __init__(self, folder: str, enabled: bool = True):
        """Create the cache folder (when enabled) and remember whether caching is active."""
        self.folder = folder
        self.enabled = enabled
        if enabled:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def make_key(*parts) -> str:
        """Build a stable SHA-256 key from any JSON-serialisable parts."""
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str):
        """Return the cached value for `key`, or None if absent, unreadable or caching is disabled."""
        if not self.enabled:
            return None
        try:
            with open(os.path.join(self.folder, key + ".json"), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value) -> None:
        """Persist `value` under `key`, writing atomically so an interrupted run cannot corrupt it."""
        if not self.enabled:
            return
        path = os.path.join(self.folder, key + ".json")
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        os.replace(path + ".tmp", path)
