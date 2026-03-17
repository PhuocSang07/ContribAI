"""Persistent memory system using SQLite.

Tracks analyzed repos, submitted PRs, and learning data
to avoid duplicate work and improve over time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyzed_repos (
    full_name   TEXT PRIMARY KEY,
    language    TEXT,
    stars       INTEGER,
    analyzed_at TEXT,
    findings    INTEGER DEFAULT 0,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS submitted_prs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    repo        TEXT NOT NULL,
    pr_number   INTEGER NOT NULL,
    pr_url      TEXT NOT NULL,
    title       TEXT NOT NULL,
    type        TEXT NOT NULL,
    status      TEXT DEFAULT 'open',
    branch      TEXT,
    fork        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    UNIQUE(repo, pr_number)
);

CREATE TABLE IF NOT EXISTS findings_cache (
    id          TEXT PRIMARY KEY,
    repo        TEXT NOT NULL,
    type        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    title       TEXT NOT NULL,
    file_path   TEXT,
    status      TEXT DEFAULT 'new',
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS run_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT,
    finished_at TEXT,
    repos_analyzed INTEGER DEFAULT 0,
    prs_created  INTEGER DEFAULT 0,
    findings     INTEGER DEFAULT 0,
    errors       INTEGER DEFAULT 0,
    metadata     TEXT DEFAULT '{}'
);
"""


class Memory:
    """Persistent memory backed by SQLite."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path).expanduser()
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        """Initialize database connection and schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("Memory initialized at %s", self._db_path)

    async def close(self):
        if self._db:
            await self._db.close()

    # ── Repos ──────────────────────────────────────────────────────────────

    async def has_analyzed(self, full_name: str) -> bool:
        """Check if a repo has been analyzed before."""
        cursor = await self._db.execute(
            "SELECT 1 FROM analyzed_repos WHERE full_name = ?", (full_name,)
        )
        return await cursor.fetchone() is not None

    async def record_analysis(self, full_name: str, language: str, stars: int, findings_count: int):
        """Record that a repo was analyzed."""
        await self._db.execute(
            """INSERT OR REPLACE INTO analyzed_repos
               (full_name, language, stars, analyzed_at, findings)
               VALUES (?, ?, ?, ?, ?)""",
            (full_name, language, stars, datetime.utcnow().isoformat(), findings_count),
        )
        await self._db.commit()

    async def get_analyzed_repos(self, limit: int = 50) -> list[dict]:
        """Get recently analyzed repos."""
        cursor = await self._db.execute(
            "SELECT * FROM analyzed_repos ORDER BY analyzed_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    # ── PRs ────────────────────────────────────────────────────────────────

    async def record_pr(
        self,
        repo: str,
        pr_number: int,
        pr_url: str,
        title: str,
        pr_type: str,
        branch: str = "",
        fork: str = "",
    ):
        """Record a submitted PR."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """INSERT OR REPLACE INTO submitted_prs
               (repo, pr_number, pr_url, title, type, branch, fork, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (repo, pr_number, pr_url, title, pr_type, branch, fork, now, now),
        )
        await self._db.commit()

    async def update_pr_status(self, repo: str, pr_number: int, status: str):
        """Update PR status."""
        await self._db.execute(
            "UPDATE submitted_prs SET status = ?, updated_at = ? WHERE repo = ? AND pr_number = ?",
            (status, datetime.utcnow().isoformat(), repo, pr_number),
        )
        await self._db.commit()

    async def get_prs(self, status: str | None = None, limit: int = 50) -> list[dict]:
        """Get submitted PRs, optionally filtered by status."""
        if status:
            cursor = await self._db.execute(
                "SELECT * FROM submitted_prs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM submitted_prs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    async def get_today_pr_count(self) -> int:
        """Get number of PRs created today."""
        today = datetime.utcnow().date().isoformat()
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM submitted_prs WHERE created_at LIKE ?",
            (f"{today}%",),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Run Log ────────────────────────────────────────────────────────────

    async def start_run(self) -> int:
        """Record the start of a pipeline run. Returns run ID."""
        cursor = await self._db.execute(
            "INSERT INTO run_log (started_at) VALUES (?)",
            (datetime.utcnow().isoformat(),),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def finish_run(
        self,
        run_id: int,
        repos_analyzed: int,
        prs_created: int,
        findings: int,
        errors: int,
    ):
        """Record the completion of a pipeline run."""
        await self._db.execute(
            """UPDATE run_log
               SET finished_at = ?, repos_analyzed = ?, prs_created = ?,
                   findings = ?, errors = ?
               WHERE id = ?""",
            (datetime.utcnow().isoformat(), repos_analyzed, prs_created, findings, errors, run_id),
        )
        await self._db.commit()

    async def get_stats(self) -> dict:
        """Get overall statistics."""
        stats = {}

        cursor = await self._db.execute("SELECT COUNT(*) FROM analyzed_repos")
        stats["total_repos_analyzed"] = (await cursor.fetchone())[0]

        cursor = await self._db.execute("SELECT COUNT(*) FROM submitted_prs")
        stats["total_prs_submitted"] = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM submitted_prs WHERE status = 'merged'"
        )
        stats["prs_merged"] = (await cursor.fetchone())[0]

        cursor = await self._db.execute("SELECT COUNT(*) FROM run_log")
        stats["total_runs"] = (await cursor.fetchone())[0]

        return stats
