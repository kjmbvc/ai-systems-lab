import json
import sqlite3
from pathlib import Path

import pytest

from ai_systems_lab.dag import Journal
from ai_systems_lab.pipeline import Pipeline, ResponseCache
from ai_systems_lab.store import Store, collect, deduplicate, validate


class FakeModel:
    """Explicit test double; does not prove model quality or model integration."""

    identity = "test-double-v1"

    def __init__(self):
        self.calls = 0

    async def answer(self, question, context):
        self.calls += 1
        return {"answer": "crash recovery"}


@pytest.fixture
def pipeline():
    store, journal, model = Store(), Journal(), FakeModel()
    value = Pipeline(store, journal, model, cache=True)
    yield value
    store.close()
    journal.close()


async def test_e2e_ingestion_query_cache_and_abstention(pipeline):
    stats = await pipeline.ingest(Path("data/documents.jsonl"))
    assert stats["input_count"] == 13 and stats["unique_count"] == 12
    assert await pipeline.ingest(Path("data/documents.jsonl")) == stats
    first = await pipeline.query(
        "What is the purpose of a write ahead log?", "crash recovery", "wal"
    )
    second = await pipeline.query("What is the purpose of a write ahead log?")
    assert first["retrieval_hit"] and first["exact_match"] and first["model_called"]
    assert second["cache_hit"] and not second["model_called"]
    assert pipeline.model.calls == 1
    unknown = await pipeline.query("xyzzyquux")
    assert unknown["answer"] == "I don't know" and not unknown["model_called"]
    expanded = await pipeline.query("durable journal")
    assert expanded["actions"] == ["retrieve", "reformulate_once", "answer"]


async def test_corpus_and_model_change_invalidate_cache(pipeline, tmp_path):
    path = Path("data/documents.jsonl")
    await pipeline.ingest(path)
    await pipeline.query("write ahead log")
    changed = tmp_path / "changed.jsonl"
    changed.write_text(
        json.dumps({"id": "wal", "text": "A write ahead log is a journal.", "license": "CC0-1.0"})
    )
    await pipeline.ingest(changed)
    assert not (await pipeline.query("write ahead log"))["cache_hit"]
    pipeline.model.identity = "test-double-v2"
    assert not (await pipeline.query("write ahead log"))["cache_hit"]


async def test_query_validation(pipeline):
    with pytest.raises(RuntimeError):
        await pipeline.query("hello")
    await pipeline.ingest(Path("data/documents.jsonl"))
    for query in ["", " " * 10, "x" * 1001]:
        with pytest.raises(ValueError):
            await pipeline.query(query)


@pytest.mark.parametrize(
    "row",
    [
        None,
        {},
        {"id": 1, "text": "x", "license": "CC0-1.0"},
        {"id": "a", "text": "", "license": "CC0-1.0"},
        {"id": "a", "text": "x", "license": "unknown"},
    ],
)
def test_validation_failures(row):
    with pytest.raises(ValueError):
        validate([row])


def test_dedup_normalization_and_conflict():
    rows = validate(
        [
            {"id": "a", "text": "A   log", "license": "CC0-1.0"},
            {"id": "b", "text": "A log", "license": "CC0-1.0"},
        ]
    )
    assert len(deduplicate(rows)) == 1
    with pytest.raises(ValueError):
        deduplicate([rows[0], {**rows[0], "text": "different"}])


def test_collect_bound_and_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"x" * 2_000_001)
    with pytest.raises(ValueError, match="limit"):
        collect(path)
    path.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        collect(path)


def test_transaction_rollback_and_query_escaping():
    store = Store()
    try:
        rows = deduplicate(validate(collect(Path("data/documents.jsonl"))))
        store.replace(rows)
        with pytest.raises(sqlite3.IntegrityError):
            store.replace([rows[0], rows[0]])
        assert store.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 12
        store.search('" OR * -- ; DROP TABLE docs')
        assert store.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 12
        assert store.search("write ahead log")[0]["id"] == "wal"
    finally:
        store.close()


def test_cache_lru_ttl(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("ai_systems_lab.pipeline.time.monotonic", lambda: clock[0])
    cache = ResponseCache(capacity=2, ttl_s=5)
    cache.put("a", {"answer": "A"})
    cache.put("b", {"answer": "B"})
    assert cache.get("a")["answer"] == "A"
    cache.put("c", {"answer": "C"})
    assert cache.get("b") is None
    clock[0] = 6
    assert cache.get("a") is None
