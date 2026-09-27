"""Cooperative async DAG scheduling; resource claims are admission estimates, not RSS limits."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Node:
    name: str
    run: Callable[[dict[str, Any]], Awaitable[Any]]
    deps: tuple[str, ...] = ()
    memory_mb: int = 1
    timeout_s: float = 30.0
    retries: int = 0
    version: str = "1"
    pure: bool = False


class GraphError(ValueError):
    pass


class NodeFailed(RuntimeError):
    def __init__(self, node: str, cause: Exception) -> None:
        super().__init__(f"{node}: {type(cause).__name__}")
        self.node = node
        self.cause = cause


class Journal:
    """Single-process SQLite journal. Never stores exception messages or secrets."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS events (run TEXT, node TEXT, status TEXT, "
            "attempt INTEGER, elapsed_ms REAL, error TEXT);"
        )

    def event(
        self,
        run: str,
        node: str,
        status: str,
        attempt: int = 0,
        elapsed_ms: float = 0,
        error: str | None = None,
    ) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                (run, node, status, attempt, elapsed_ms, error),
            )

    def get(self, key: str) -> tuple[bool, Any]:
        row = self.db.execute("SELECT value FROM results WHERE key=?", (key,)).fetchone()
        return (True, json.loads(row[0])) if row else (False, None)

    def put(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO results VALUES (?,?)", (key, encoded))

    def events(self) -> list[dict[str, Any]]:
        cur = self.db.execute("SELECT * FROM events ORDER BY rowid")
        keys = [item[0] for item in cur.description]
        return [dict(zip(keys, row, strict=True)) for row in cur.fetchall()]

    def close(self) -> None:
        self.db.close()


class Executor:
    def __init__(
        self,
        journal: Journal,
        *,
        concurrency: int = 2,
        memory_mb: int = 2048,
        max_nodes: int = 1024,
        namespace: str = "default-v1",
    ) -> None:
        if min(concurrency, memory_mb, max_nodes) < 1:
            raise ValueError("positive resource limits required")
        self.journal = journal
        self.concurrency = concurrency
        self.memory_mb = memory_mb
        self.max_nodes = max_nodes
        self.namespace = namespace
        self._lock = asyncio.Lock()
        self.peak_active = 0
        self.peak_claimed_mb = 0

    def validate(self, nodes: list[Node]) -> dict[str, Node]:
        by_name = {n.name: n for n in nodes}
        if len(nodes) > self.max_nodes or len(by_name) != len(nodes):
            raise GraphError("graph too large or duplicate names")
        for n in nodes:
            if (
                not n.name
                or n.memory_mb < 1
                or n.memory_mb > self.memory_mb
                or not math.isfinite(n.timeout_s)
                or n.timeout_s <= 0
                or n.retries < 0
            ):
                raise GraphError("invalid node resource declaration")
            if len(set(n.deps)) != len(n.deps) or set(n.deps) - by_name.keys():
                raise GraphError("duplicate or missing dependency")
        # Kahn validation avoids recursion limits and runs before any user code.
        indegree = {n.name: len(n.deps) for n in nodes}
        children: dict[str, list[str]] = {n.name: [] for n in nodes}
        for n in nodes:
            for dep in n.deps:
                children[dep].append(n.name)
        ready = [name for name, count in indegree.items() if count == 0]
        visited = 0
        while ready:
            name = ready.pop()
            visited += 1
            for child in children[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(nodes):
            raise GraphError("cycle detected")
        return by_name

    async def _execute(self, node: Node, inputs: dict[str, Any], run_id: str) -> Any:
        key = digest([self.namespace, node.name, node.version, inputs]) if node.pure else ""
        if node.pure:
            found, value = self.journal.get(key)
            if found:
                self.journal.event(run_id, node.name, "cached")
                return value
        for attempt in range(1, node.retries + 2):
            start = time.perf_counter()
            self.journal.event(run_id, node.name, "started", attempt)
            try:
                async with asyncio.timeout(node.timeout_s):
                    value = await node.run(inputs)
                if node.pure:
                    self.journal.put(key, value)
                self.journal.event(
                    run_id, node.name, "succeeded", attempt, (time.perf_counter() - start) * 1000
                )
                return value
            except asyncio.CancelledError:
                self.journal.event(run_id, node.name, "cancelled", attempt)
                raise
            except Exception as exc:
                status = "retry" if attempt <= node.retries else "failed"
                self.journal.event(
                    run_id,
                    node.name,
                    status,
                    attempt,
                    (time.perf_counter() - start) * 1000,
                    type(exc).__name__,
                )
                if attempt > node.retries:
                    raise NodeFailed(node.name, exc) from exc
                await asyncio.sleep(min(0.01 * 2 ** (attempt - 1), 1.0))
        raise AssertionError("unreachable")

    async def run(
        self, nodes: list[Node], payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self.validate(nodes)
        # Sharing an executor serializes graphs, retaining a true per-instance resource bound.
        async with self._lock:
            run_id = uuid.uuid4().hex
            pending = {n.name: n for n in nodes}
            results: dict[str, Any] = {}
            active: dict[asyncio.Task[Any], Node] = {}
            claimed = 0
            self.journal.event(run_id, "__graph__", "started")
            try:
                while pending or active:
                    for name, node in list(pending.items()):
                        if len(active) >= self.concurrency:
                            break
                        if (
                            set(node.deps) <= results.keys()
                            and claimed + node.memory_mb <= self.memory_mb
                        ):
                            inputs = {dep: results[dep] for dep in node.deps}
                            inputs["__input__"] = dict(payload or {})
                            task = asyncio.create_task(self._execute(node, inputs, run_id))
                            active[task] = node
                            claimed += node.memory_mb
                            del pending[name]
                    self.peak_active = max(self.peak_active, len(active))
                    self.peak_claimed_mb = max(self.peak_claimed_mb, claimed)
                    done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                    # Consume every completed task even when one failed (no orphan exceptions).
                    failure: BaseException | None = None
                    for task in done:
                        node = active.pop(task)
                        claimed -= node.memory_mb
                        try:
                            results[node.name] = task.result()
                        except BaseException as exc:
                            failure = failure or exc
                    if failure:
                        raise failure
                self.journal.event(run_id, "__graph__", "succeeded")
                return results
            except BaseException as exc:
                for task in active:
                    task.cancel()
                await asyncio.gather(*active, return_exceptions=True)
                for name in pending:
                    self.journal.event(run_id, name, "skipped")
                self.journal.event(run_id, "__graph__", "failed", error=type(exc).__name__)
                raise


async def bounded_stream(
    source: AsyncIterable[Any],
    handler: Callable[[Any], Awaitable[None]],
    *,
    workers: int = 1,
    capacity: int = 4,
) -> None:
    """At most capacity queued + workers active + one producer-held item; no output buffer."""
    if min(workers, capacity) < 1:
        raise ValueError("positive stream limits required")
    queue: asyncio.Queue[Any] = asyncio.Queue(capacity)
    stop = object()

    async def produce() -> None:
        iterator = aiter(source)
        try:
            async for item in iterator:
                await queue.put(item)
            for _ in range(workers):
                await queue.put(stop)
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def consume() -> None:
        while True:
            item = await queue.get()
            try:
                if item is stop:
                    return
                await handler(item)
            finally:
                queue.task_done()

    async with asyncio.TaskGroup() as group:
        group.create_task(produce())
        for _ in range(workers):
            group.create_task(consume())
