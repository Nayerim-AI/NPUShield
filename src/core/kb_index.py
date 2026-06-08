"""FTS5 knowledge base index for NPUShield RAG."""

from __future__ import annotations

import sqlite3
import re
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class SearchResult:
    path: str
    text: str
    score: float


@dataclass
class KBIndex:
    db_path: str = ":memory:"

    def __post_init__(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS kb USING fts5("
            "  path, text,"
            "  tokenize='porter unicode61'"
            ")"
        )

    def add_document(self, path: str, text: str) -> None:
        self._conn.execute("INSERT INTO kb(path, text) VALUES (?, ?)", (path, text))

    def add_documents(self, documents: list[tuple[str, str]]) -> None:
        self._conn.executemany(
            "INSERT INTO kb(path, text) VALUES (?, ?)", documents
        )

    def search(self, query: str, limit: int = 5, path_prefix: str | None = None) -> list[SearchResult]:
        if not query.strip():
            return []
        cleaned = self._clean_query(query)
        if not cleaned:
            return []
        try:
            if path_prefix:
                rows = self._conn.execute(
                    "SELECT path, text, rank FROM kb WHERE kb MATCH ? AND path GLOB ? ORDER BY rank LIMIT ?",
                    (cleaned, f"{path_prefix.rstrip('/')}/*", limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT path, text, rank FROM kb WHERE kb MATCH ? ORDER BY rank LIMIT ?",
                    (cleaned, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            SearchResult(path=r[0], text=r[1], score=round(1.0 / abs(float(r[2] or 1.0)), 4))
            for r in rows
        ]

    def format_context(self, results: list[SearchResult], max_chars: int = 4000) -> str:
        if not results:
            return "tidak ada informasi yang cocok di knowledge base."
        parts = []
        total = 0
        for r in results:
            header = f"# {r.path} (score: {r.score})"
            remaining = max_chars - total - len(header) - 2
            body = r.text[:remaining] if remaining > 50 else r.text[:50]
            block = f"{header}\n{body}"
            parts.append(block)
            total += len(block)
            if total >= max_chars:
                break
        return "\n\n".join(parts)

    def document_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM kb").fetchone()
        return row[0] if row else 0

    @staticmethod
    def _clean_query(query: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_\- ]", " ", query)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) < 2:
            return ""
        tokens = cleaned.split()
        return " OR ".join(tokens)
