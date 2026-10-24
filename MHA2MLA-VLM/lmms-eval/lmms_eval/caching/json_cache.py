# SPDX-License-Identifier: Apache-2.0
"""SQLite response cache with data-only JSON serialization.

A separate table leaves historical pickle-backed caches intact without reading
or deserializing them. Only evaluation response data is supported.
"""
from collections.abc import MutableMapping
import json
import math
import sqlite3
import threading


def _pack(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": str(value)}
    if isinstance(value, tuple):
        return {"tuple": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, dict):
        return {"mapping": [[_pack(key), _pack(item)] for key, item in value.items()]}
    raise TypeError(f"Unsupported response cache type: {type(value).__name__}")


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"tuple"}:
            return tuple(_unpack(item) for item in value["tuple"])
        if set(value) == {"mapping"}:
            return {_unpack(key): _unpack(item) for key, item in value["mapping"]}
        if set(value) == {"float"} and value["float"] in {"inf", "-inf", "nan"}:
            return float(value["float"])
        raise ValueError("Invalid response cache record")
    return value


class JsonCache(MutableMapping):
    """Mapping interface used by CachingLMM, with per-write autocommit."""

    def __init__(self, filename):
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(filename, timeout=30, isolation_level=None, check_same_thread=False)
        self._connection.execute("CREATE TABLE IF NOT EXISTS responses_json_v1 (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def __getitem__(self, key):
        with self._lock:
            row = self._connection.execute("SELECT value FROM responses_json_v1 WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return _unpack(json.loads(row[0]))

    def __setitem__(self, key, value):
        encoded = json.dumps(_pack(value), ensure_ascii=False, allow_nan=False)
        with self._lock:
            self._connection.execute("INSERT OR REPLACE INTO responses_json_v1 (key, value) VALUES (?, ?)", (key, encoded))

    def __delitem__(self, key):
        with self._lock:
            cursor = self._connection.execute("DELETE FROM responses_json_v1 WHERE key = ?", (key,))
        if not cursor.rowcount:
            raise KeyError(key)

    def __iter__(self):
        with self._lock:
            rows = self._connection.execute("SELECT key FROM responses_json_v1").fetchall()
        return iter(key for (key,) in rows)

    def __len__(self):
        with self._lock:
            return self._connection.execute("SELECT COUNT(*) FROM responses_json_v1").fetchone()[0]

    def commit(self):
        with self._lock:
            self._connection.commit()

    def close(self):
        with self._lock:
            self._connection.close()
