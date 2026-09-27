"""Synthetic-only ingestion and transactional lexical search."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from .dag import digest

MAX_BYTES = 2_000_000
MAX_DOCUMENTS = 1000


def collect(path: Path) -> list[Any]:
    # Bounded read remains bounded if the file grows after stat().
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("corpus exceeds 2 MB limit")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate(rows: list[Any]) -> list[dict[str, str]]:
    if not 0 < len(rows) <= MAX_DOCUMENTS:
        raise ValueError("invalid corpus size")
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "text", "license"}:
            raise ValueError("invalid document schema")
        if any(not isinstance(value, str) for value in row.values()):
            raise ValueError("document fields must be strings")
        if not re.fullmatch(r"[a-z0-9-]{1,64}", row["id"]) or row["license"] != "CC0-1.0":
            raise ValueError("invalid id or unsupported data license")
        text = normalize(row["text"])
        if not text or len(text) > 4000:
            raise ValueError("invalid text length")
        result.append({**row, "text": text})
    return result


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen_ids: dict[str, str] = {}
    by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        hashed = digest(row["text"])
        if row["id"] in seen_ids and seen_ids[row["id"]] != hashed:
            raise ValueError("conflicting duplicate id")
        seen_ids[row["id"]] = hashed
        by_hash.setdefault(hashed, {**row, "hash": hashed})
    return sorted(by_hash.values(), key=lambda row: row["id"])


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, text TEXT, hash TEXT UNIQUE);"
            "CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED, text);"
        )

    def replace(self, rows: list[dict[str, str]]) -> str:
        # The full snapshot, FTS table and corpus identity move together in one transaction.
        with self.db:
            self.db.execute("DELETE FROM docs")
            self.db.execute("DELETE FROM search")
            self.db.executemany("INSERT INTO docs VALUES (:id,:text,:hash)", rows)
            self.db.executemany("INSERT INTO search VALUES (:id,:text)", rows)
        return digest(rows)

    def search(self, query: str, limit: int = 2) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10 or len(query) > 1000:
            raise ValueError("invalid retrieval request")
        stop = {"what", "is", "the", "of", "a", "an", "does", "which", "how", "in", "for", "to"}
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in stop][:32]
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        rows = self.db.execute(
            "SELECT id,text,bm25(search) FROM search WHERE search MATCH ? "
            "ORDER BY bm25(search),id LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [{"id": row[0], "text": row[1], "score": row[2]} for row in rows]

    def close(self) -> None:
        self.db.close()
