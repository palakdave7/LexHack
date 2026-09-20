"""Local SQLite cache for CourtListener opinion bodies.

Every opinion we fetch is stored keyed by opinion_id. Fetches are expensive
(one HTTP round-trip each, rate-limited to 5000/hour on the free tier), so
we treat the cache as authoritative once populated - fetch once, reuse forever.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(".cache/opinions.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS opinions (
    opinion_id   INTEGER PRIMARY KEY,
    cluster_id   INTEGER NOT NULL,
    text         TEXT NOT NULL,
    source_field TEXT NOT NULL,      -- which CL field the text came from
    char_count   INTEGER NOT NULL,
    fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_opinions_cluster ON opinions(cluster_id);

CREATE TABLE IF NOT EXISTS fetch_failures (
    opinion_id   INTEGER PRIMARY KEY,
    reason       TEXT NOT NULL,
    failed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def get_opinion(opinion_id: int) -> str | None:
    """Return cached opinion text, or None if not cached."""
    with _conn() as con:
        row = con.execute(
            "SELECT text FROM opinions WHERE opinion_id = ?", (opinion_id,)
        ).fetchone()
        return row["text"] if row else None


def store_opinion(
    opinion_id: int, cluster_id: int, text: str, source_field: str
) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO opinions "
            "(opinion_id, cluster_id, text, source_field, char_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (opinion_id, cluster_id, text, source_field, len(text)),
        )


def record_failure(opinion_id: int, reason: str) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO fetch_failures (opinion_id, reason) VALUES (?, ?)",
            (opinion_id, reason),
        )


def had_failure(opinion_id: int) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT reason FROM fetch_failures WHERE opinion_id = ?", (opinion_id,)
        ).fetchone()
        return row["reason"] if row else None


def cache_stats() -> dict[str, int]:
    with _conn() as con:
        opinions = con.execute("SELECT COUNT(*) AS n FROM opinions").fetchone()["n"]
        chars = con.execute(
            "SELECT COALESCE(SUM(char_count), 0) AS n FROM opinions"
        ).fetchone()["n"]
        failures = con.execute("SELECT COUNT(*) AS n FROM fetch_failures").fetchone()[
            "n"
        ]
    return {"opinions_cached": opinions, "total_chars": chars, "failures": failures}
