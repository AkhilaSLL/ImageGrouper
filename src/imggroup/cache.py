"""
On-disk feature cache.

Describing an image is the expensive part of a scan, and a dump barely changes
between runs.  Keying on ``(path, size, mtime)`` lets a rescan skip decoding
anything untouched, which turns a repeat scan of a large folder from minutes
into seconds - and, because the similarity slider re-groups from the cached
features alone, moving the slider costs nothing at all.

The cache is strictly an optimisation: every operation degrades to a no-op if
the database cannot be opened or written, and a stale row is detected by the
size/mtime check rather than trusted.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import sys
import time
from collections.abc import Iterable, Iterator

from .engine import CachedFeatures

APP_DIR_NAME = "ImageGrouper"
DB_NAME = "features.sqlite3"
MAX_AGE_DAYS = 120
MAX_ROWS = 400_000

# Bump whenever the stored vector changes meaning.  Rows written by an older
# descriptor are not merely stale - they describe images in a representation
# that no longer exists - so they are dropped rather than migrated.
SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    key       TEXT PRIMARY KEY,
    size      INTEGER NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    features  BLOB NOT NULL,
    used_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_used_at ON features(used_at);
"""


def default_dir() -> str:
    """Per-user application data directory."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support")
    else:
        base = (os.environ.get("XDG_CACHE_HOME")
                or os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(base, APP_DIR_NAME)


class FeatureCache:
    """SQLite-backed store of previously computed image descriptors."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.join(default_dir(), DB_NAME)
        self.available = False
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with self._session() as db:
                version = int(
                    db.execute("PRAGMA user_version").fetchone()[0])
                if version != SCHEMA_VERSION:
                    db.executescript("DROP TABLE IF EXISTS features")
                db.executescript(_SCHEMA)
                db.execute(f"PRAGMA user_version = {SCHEMA_VERSION:d}")
            self.available = True
        except Exception:
            pass                     # run without a cache rather than fail

    def _connect(self) -> sqlite3.Connection:
        # A fresh connection per operation: scans run on a worker thread and
        # SQLite connections are not safely shared across threads.
        db = sqlite3.connect(self.path, timeout=5.0)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Transactional connection that is always closed again."""
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    # ---- reads ------------------------------------------------------------ #
    def load(self, paths: Iterable[str]) -> dict[str, CachedFeatures]:
        """Fetch cached rows for ``paths``.  Missing or unreadable -> empty."""
        keys = [os.path.normcase(os.path.abspath(p)) for p in paths]
        if not keys or not self.available:
            return {}
        found: dict[str, CachedFeatures] = {}
        try:
            with self._session() as db:
                # SQLite caps host parameters per statement, so batch the IN().
                for start in range(0, len(keys), 500):
                    batch = keys[start:start + 500]
                    placeholders = ",".join("?" * len(batch))
                    rows = db.execute(
                        "SELECT key, size, mtime_ns, features"
                        f" FROM features WHERE key IN ({placeholders})", batch)
                    for row in rows:
                        found[row[0]] = CachedFeatures(
                            key=row[0], size=row[1], mtime_ns=row[2],
                            features=bytes(row[3]))
        except Exception:
            return {}
        return found

    def count(self) -> int:
        if not self.available:
            return 0
        try:
            with self._session() as db:
                return int(db.execute(
                    "SELECT COUNT(*) FROM features").fetchone()[0])
        except Exception:
            return 0

    # ---- writes ----------------------------------------------------------- #
    def store(self, entries: Iterable[CachedFeatures]) -> None:
        rows = [(e.key, e.size, e.mtime_ns, e.features,
                 int(time.time())) for e in entries]
        if not rows or not self.available:
            return
        try:
            with self._session() as db:
                db.executemany(
                    "INSERT INTO features (key, size, mtime_ns,"
                    " features, used_at) VALUES (?,?,?,?,?)"
                    " ON CONFLICT(key) DO UPDATE SET"
                    "   size=excluded.size, mtime_ns=excluded.mtime_ns,"
                    "   features=excluded.features,"
                    "   used_at=excluded.used_at", rows)
        except Exception:
            pass

    def prune(self, max_age_days: int = MAX_AGE_DAYS,
              max_rows: int = MAX_ROWS) -> None:
        """Drop long-unused rows so the file cannot grow without bound."""
        if not self.available:
            return
        try:
            with self._session() as db:
                db.execute("DELETE FROM features WHERE used_at < ?",
                           (int(time.time()) - max_age_days * 86400,))
                db.execute(
                    "DELETE FROM features WHERE key IN ("
                    "  SELECT key FROM features"
                    "  ORDER BY used_at DESC LIMIT -1 OFFSET ?)", (max_rows,))
        except Exception:
            pass

    def clear(self) -> None:
        if not self.available:
            return
        try:
            with self._session() as db:
                db.execute("DELETE FROM features")
        except Exception:
            return
        # VACUUM cannot run inside a transaction, and letting it raise in the
        # block above would roll the DELETE straight back.
        try:
            db = self._connect()
            db.isolation_level = None
            try:
                db.execute("VACUUM")
            finally:
                db.close()
        except Exception:
            pass
