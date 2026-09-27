"""Persistent local CPU model subprocess with a killable inference boundary."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

MODEL_ID = "google/flan-t5-small"
REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
GENERATION = {"max_new_tokens": 32, "do_sample": False, "num_beams": 1}


class Model(Protocol):
    identity: str

    async def answer(self, question: str, context: str) -> dict[str, Any]: ...


class LocalModel:
    identity = f"{MODEL_ID}@{REVISION}:fp32:greedy32:prompt-v1"

    def __init__(self, cache_dir: Path, *, threads: int = 2, timeout_s: float = 60) -> None:
        if not 1 <= threads <= 64 or timeout_s <= 0:
            raise ValueError("invalid model resources")
        self.cache_dir = cache_dir
        self.threads = threads
        self.timeout_s = timeout_s
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.ready: dict[str, Any] = {}

    async def _start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "ai_systems_lab.model_worker",
            "--cache-dir",
            str(self.cache_dir),
            "--threads",
            str(self.threads),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # No stderr pipe: verbose library logs cannot fill a pipe and deadlock inference.
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        assert self.process.stdout
        line = await self.process.stdout.readline()
        self.ready = json.loads(line)
        if not self.ready.get("ready"):
            raise RuntimeError("model worker did not become ready")

    async def start(self) -> None:
        async with self.lock:
            try:
                async with asyncio.timeout(self.timeout_s):
                    await self._start()
            except BaseException:
                await self.close()
                raise

    async def answer(self, question: str, context: str) -> dict[str, Any]:
        if len(question) > 1000 or len(context) > 12000:
            raise ValueError("model input exceeds bound")
        async with self.lock:
            try:
                async with asyncio.timeout(self.timeout_s):
                    await self._start()
                    assert self.process and self.process.stdin and self.process.stdout
                    request = json.dumps({"question": question, "context": context}) + "\n"
                    self.process.stdin.write(request.encode())
                    await self.process.stdin.drain()
                    line = await self.process.stdout.readline()
                    value: dict[str, Any] = json.loads(line)
                    if "error" in value:
                        raise RuntimeError("model worker inference failed")
                    return value
            except BaseException:
                await self.close()
                raise

    async def close(self) -> None:
        if self.process:
            if self.process.returncode is None:
                try:
                    if os.name == "nt":
                        # Windows venv python.exe is a launcher: terminate its child too.
                        killer = await asyncio.create_subprocess_exec(
                            "taskkill",
                            "/PID",
                            str(self.process.pid),
                            "/T",
                            "/F",
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        await killer.wait()
                        if self.process.returncode is None:
                            self.process.kill()
                    else:
                        self.process.kill()
                except ProcessLookupError:
                    pass
            await self.process.wait()
            self.process = None
