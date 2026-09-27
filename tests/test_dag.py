import asyncio
from dataclasses import replace

import pytest

from ai_systems_lab.dag import Executor, GraphError, Journal, Node, NodeFailed, bounded_stream


@pytest.fixture
def journal():
    value = Journal()
    yield value
    value.close()


async def one(_):
    return 1


@pytest.mark.parametrize(
    "nodes",
    [
        [Node("a", one, ("b",)), Node("b", one, ("a",))],
        [Node("a", one, ("missing",))],
        [Node("a", one), Node("a", one)],
        [Node("a", one, ("a",))],
        [Node("a", one, memory_mb=3000)],
        [Node("a", one, timeout_s=0)],
        [Node("a", one, timeout_s=float("nan"))],
        [Node("a", one, retries=-1)],
    ],
)
async def test_invalid_graph_has_no_side_effects(journal, nodes):
    with pytest.raises(GraphError):
        await Executor(journal).run(nodes)
    assert journal.events() == []


async def test_empty_and_node_limit(journal):
    executor = Executor(journal, max_nodes=1)
    assert await executor.run([]) == {}
    with pytest.raises(GraphError):
        await executor.run([Node("a", one), Node("b", one)])


async def test_diamond_and_resource_bounds(journal):
    active = peak = 0

    async def branch(data):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return data["root"] + 1

    async def join(data):
        return data["left"] + data["right"]

    executor = Executor(journal, concurrency=2, memory_mb=10)
    nodes = [
        Node("root", one),
        Node("left", branch, ("root",), memory_mb=5),
        Node("right", branch, ("root",), memory_mb=5),
        Node("join", join, ("left", "right")),
    ]
    assert (await executor.run(nodes))["join"] == 4
    assert peak == executor.peak_active == 2
    assert executor.peak_claimed_mb == 10
    executor = Executor(journal, concurrency=2, memory_mb=9)
    peak = 0
    await executor.run(nodes)
    assert peak == 1  # Memory, not worker count, now limits parallelism.


async def test_retry_and_persistent_pure_replay(tmp_path):
    calls = 0

    async def flaky(data):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient")
        return data["__input__"]["value"]

    node = Node("a", flaky, retries=1, pure=True)
    path = tmp_path / "journal.sqlite"
    journal = Journal(path)
    await Executor(journal).run([node], {"value": 7})
    journal.close()
    journal = Journal(path)
    try:
        executor = Executor(journal)
        assert (await executor.run([node], {"value": 7}))["a"] == 7
        assert calls == 2
        await executor.run([node], {"value": 8})
        await executor.run([replace(node, version="2")], {"value": 8})
        await Executor(journal, namespace="changed").run([node], {"value": 8})
        assert calls == 5
        assert "cached" in [e["status"] for e in journal.events()]
    finally:
        journal.close()


async def test_timeout_retries_and_downstream_skip(journal):
    cleaned = 0

    async def slow(_):
        nonlocal cleaned
        try:
            await asyncio.sleep(10)
        finally:
            cleaned += 1

    with pytest.raises(NodeFailed) as error:
        await Executor(journal).run(
            [Node("slow", slow, timeout_s=0.01, retries=1), Node("child", one, ("slow",))]
        )
    assert isinstance(error.value.cause, TimeoutError)
    assert cleaned == 2
    assert any(e["node"] == "child" and e["status"] == "skipped" for e in journal.events())


async def test_failure_cancels_sibling(journal):
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def sibling(_):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.set()

    async def failing(_):
        await started.wait()
        raise RuntimeError("failure")

    with pytest.raises(NodeFailed):
        await Executor(journal).run([Node("sibling", sibling), Node("fail", failing)])
    assert cleaned.is_set()


async def test_external_cancellation(journal):
    started = asyncio.Event()

    async def slow(_):
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(Executor(journal).run([Node("slow", slow)]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(e["status"] == "cancelled" for e in journal.events())


async def test_backpressure():
    produced = completed = 0
    gate = asyncio.Event()

    async def source():
        nonlocal produced
        for i in range(100):
            produced += 1
            yield i

    async def handle(_):
        nonlocal completed
        await gate.wait()
        completed += 1

    task = asyncio.create_task(bounded_stream(source(), handle, workers=2, capacity=3))
    await asyncio.sleep(0.03)
    assert produced <= 3 + 2 + 1
    assert completed == 0
    gate.set()
    await task
    assert completed == 100


async def test_stream_failure_cancels_producer():
    closed = asyncio.Event()

    async def source():
        try:
            for i in range(1000):
                yield i
        finally:
            closed.set()

    async def fail(_):
        raise ValueError("failed")

    with pytest.raises(ExceptionGroup):
        await bounded_stream(source(), fail, capacity=1)
    # Async generator finalization is driven by the loop after producer cancellation.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert closed.is_set()
