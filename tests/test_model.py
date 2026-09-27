import asyncio
import os
import sys
from pathlib import Path

import psutil
import pytest

from ai_systems_lab.dag import Journal
from ai_systems_lab.model import LocalModel
from ai_systems_lab.pipeline import Pipeline
from ai_systems_lab.store import Store


async def test_model_timeout_kills_process(tmp_path):
    model = LocalModel(tmp_path, timeout_s=0.5)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    model.process = process
    await asyncio.sleep(0.1)
    descendants = psutil.Process(process.pid).children(recursive=True)
    with pytest.raises(TimeoutError):
        await model.answer("test", "context")
    assert model.process is None
    assert process.returncode is not None
    assert all(not child.is_running() for child in descendants)


async def test_model_external_cancel_kills_process(tmp_path):
    model = LocalModel(tmp_path)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    model.process = process
    task = asyncio.create_task(model.answer("test", "context"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.process is None and process.returncode is not None


@pytest.mark.model
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="opt-in real CPU weights")
async def test_real_model_rag():
    model = LocalModel(Path(".cache/model"))
    store, journal = Store(), Journal()
    try:
        pipeline = Pipeline(store, journal, model)
        await pipeline.ingest(Path("data/documents.jsonl"))
        result = await pipeline.query(
            "What is the purpose of a write ahead log?", "crash recovery", "wal"
        )
        assert result["model_called"] and result["retrieval_hit"]
        assert result["answer"] and result["output_tokens"] > 0
        assert result["encoder_ms"] > 0 and result["decode_ms"] > 0
        assert "test-double" not in model.identity
    finally:
        await model.close()
        store.close()
        journal.close()
