"""Bounded, rule-routed RAG: retrieve, reformulate once, answer or abstain."""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .dag import Executor, Journal, Node, digest
from .model import Model
from .store import Store, collect, deduplicate, validate


def exact_match(answer: str, expected: str) -> bool:
    def clean(text: str) -> str:
        return " ".join(re.findall(r"\w+", text.casefold()))

    return clean(answer) == clean(expected)


class ResponseCache:
    def __init__(self, capacity: int = 32, ttl_s: float = 3600) -> None:
        if capacity < 1 or ttl_s <= 0:
            raise ValueError("invalid cache bounds")
        self.capacity = capacity
        self.ttl_s = ttl_s
        self.entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        item = self.entries.get(key)
        if item is None:
            return None
        if time.monotonic() - item[0] >= self.ttl_s:
            del self.entries[key]
            return None
        self.entries.move_to_end(key)
        return dict(item[1])

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.entries[key] = (time.monotonic(), dict(value))
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


class Pipeline:
    def __init__(
        self, store: Store, journal: Journal, model: Model, *, cache: bool = False
    ) -> None:
        self.store = store
        self.journal = journal
        self.model = model
        self.executor = Executor(journal, memory_mb=2048, concurrency=2, namespace="rag-v1")
        self.cache = ResponseCache() if cache else None
        self.corpus_id = ""

    async def ingest(self, path: Path) -> dict[str, Any]:
        async def source(_: dict[str, Any]) -> Any:
            return collect(path)

        async def checked(data: dict[str, Any]) -> Any:
            return validate(data["collect"])

        async def unique(data: dict[str, Any]) -> Any:
            return deduplicate(data["validate"])

        async def index(data: dict[str, Any]) -> Any:
            return self.store.replace(data["deduplicate"])

        results = await self.executor.run(
            [
                Node("collect", source, memory_mb=8),
                Node("validate", checked, ("collect",), memory_mb=16),
                Node("deduplicate", unique, ("validate",), memory_mb=16),
                Node("index", index, ("deduplicate",), memory_mb=32),
            ]
        )
        self.corpus_id = results["index"]
        return {
            "input_count": len(results["collect"]),
            "unique_count": len(results["deduplicate"]),
            "corpus_id": self.corpus_id,
        }

    async def query(
        self, question: str, expected: str = "", expected_id: str = ""
    ) -> dict[str, Any]:
        if not self.corpus_id:
            raise RuntimeError("ingest a corpus first")
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("invalid question")
        started = time.perf_counter()

        async def retrieve(_: dict[str, Any]) -> Any:
            return self.store.search(question)

        async def route(data: dict[str, Any]) -> Any:
            hits = data["retrieve"]
            actions = ["retrieve"]
            # Explicit bounded policy, not a claim of an autonomous LLM planner.
            if not hits:
                expanded = re.sub(
                    r"\bdurable journal\b", "write ahead log", question, flags=re.IGNORECASE
                )
                if expanded != question:
                    hits = self.store.search(expanded)
                    actions.append("reformulate_once")
            actions.append("answer" if hits else "abstain")
            return {"hits": hits, "actions": actions}

        async def generate(data: dict[str, Any]) -> Any:
            hits = data["route"]["hits"]
            if not hits:
                return {"answer": "I don't know", "cache_hit": False, "model_called": False}
            context = "\n".join(hit["text"] for hit in hits)
            key = digest([self.corpus_id, self.model.identity, question, context])
            cached = self.cache.get(key) if self.cache else None
            if cached is not None:
                # Cached answers are reused, model timing is not presented as current execution.
                return {"answer": cached["answer"], "cache_hit": True, "model_called": False}
            answer = await self.model.answer(question, context)
            if self.cache:
                self.cache.put(key, {"answer": answer["answer"]})
            return {**answer, "cache_hit": False, "model_called": True}

        async def evaluate(data: dict[str, Any]) -> Any:
            generated = data["generate"]
            policy = data["route"]
            return {
                **generated,
                "actions": policy["actions"],
                "citations": [hit["id"] for hit in policy["hits"]],
                "exact_match": exact_match(generated["answer"], expected) if expected else None,
                "retrieval_hit": any(hit["id"] == expected_id for hit in policy["hits"])
                if expected_id
                else None,
            }

        results = await self.executor.run(
            [
                Node("retrieve", retrieve, memory_mb=16),
                Node("route", route, ("retrieve",), memory_mb=16),
                Node("generate", generate, ("route",), memory_mb=1536, timeout_s=90, retries=1),
                Node("evaluate", evaluate, ("generate", "route"), memory_mb=8),
            ],
            {"question": question, "corpus": self.corpus_id},
        )
        return {**results["evaluate"], "latency_ms": (time.perf_counter() - started) * 1000}
