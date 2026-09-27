"""Serial closed-loop workload. Raw timings include journal, retrieval, routing and IPC."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import time
from pathlib import Path

import psutil

from ai_systems_lab.dag import Journal, digest
from ai_systems_lab.model import LocalModel
from ai_systems_lab.pipeline import Pipeline
from ai_systems_lab.store import Store


def percentile(values: list[float], p: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)]


async def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(line) for line in Path("data/questions.jsonl").read_text().splitlines()]
    workload = questions * args.repeats
    random.Random(args.seed).shuffle(workload)
    model = LocalModel(args.model_cache, threads=2)
    process = psutil.Process()
    memory: list[dict] = []
    stop = asyncio.Event()

    async def sample_memory() -> None:
        start = time.perf_counter()
        while not stop.is_set():
            members = [process, *process.children(recursive=True)]
            rss = 0
            for member in members:
                try:
                    rss += member.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            memory.append({"elapsed_s": time.perf_counter() - start, "process_tree_rss_bytes": rss})
            await asyncio.sleep(0.02)

    monitor = asyncio.create_task(sample_memory())
    raw = []
    summaries = []
    load_start = time.perf_counter()
    try:
        await model.start()
        startup_ms = (time.perf_counter() - load_start) * 1000
        # Explicit warmup; not in measured samples and never primes a response cache.
        await model.answer(questions[0]["question"], "A write ahead log supports crash recovery.")
        for arm in args.arms.split(","):
            store = Store()
            journal = Journal()
            try:
                pipeline = Pipeline(store, journal, model, cache=arm == "cache")
                ingest_start = time.perf_counter()
                ingestion = await pipeline.ingest(Path("data/documents.jsonl"))
                ingestion_ms = (time.perf_counter() - ingest_start) * 1000
                start = time.perf_counter()
                arm_rows = []
                memory_start = len(memory)
                for index, question in enumerate(workload):
                    request_start = time.perf_counter()
                    try:
                        result = await pipeline.query(
                            question["question"], question["expected"], question["document_id"]
                        )
                        row = {
                            "arm": arm,
                            "index": index,
                            "query_id": question["id"],
                            "error": None,
                            **result,
                        }
                    except Exception as exc:
                        row = {
                            "arm": arm,
                            "index": index,
                            "query_id": question["id"],
                            "error": type(exc).__name__,
                            "latency_ms": (time.perf_counter() - request_start) * 1000,
                        }
                    raw.append(row)
                    arm_rows.append(row)
                elapsed = time.perf_counter() - start
                latencies = [row["latency_ms"] for row in arm_rows]
                success = [row for row in arm_rows if row["error"] is None]
                summaries.append(
                    {
                        "arm": arm,
                        "n": len(arm_rows),
                        "mean_ms": statistics.mean(latencies),
                        "p50_ms": percentile(latencies, 0.50),
                        "p95_ms": percentile(latencies, 0.95),
                        "p99_ms": percentile(latencies, 0.99),
                        "throughput_rps": len(arm_rows) / elapsed,
                        "error_rate": 1 - len(success) / len(arm_rows),
                        "exact_match": sum(r["exact_match"] for r in success) / len(arm_rows),
                        "retrieval_hit_at_2": sum(r["retrieval_hit"] for r in success)
                        / len(arm_rows),
                        "cache_hits": sum(r["cache_hit"] for r in success),
                        "model_calls": sum(r["model_called"] for r in success),
                        "ingestion_ms": ingestion_ms,
                        "ingestion": ingestion,
                        "peak_sampled_tree_rss_bytes": max(
                            (r["process_tree_rss_bytes"] for r in memory[memory_start:]), default=0
                        ),
                        "tail_warning": "n < 1000: p99 descriptive, not a reliable tail estimate",
                        "node_peak_active": pipeline.executor.peak_active,
                        "node_peak_claimed_mb": pipeline.executor.peak_claimed_mb,
                    }
                )
                (args.output / f"events-{len(summaries)}-{arm}.json").write_text(
                    json.dumps(journal.events(), indent=2), encoding="utf-8"
                )
                print(json.dumps(summaries[-1]), flush=True)
            finally:
                store.close()
                journal.close()
    finally:
        stop.set()
        await monitor
        await model.close()
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "transformers", "tokenizers", "safetensors", "psutil"]
    }
    source_files = sorted(Path("src").rglob("*.py")) + [Path("scripts/benchmark.py")]
    source_hash = digest(
        {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    )
    git = await asyncio.create_subprocess_exec(
        "git",
        "rev-parse",
        "HEAD",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    git_output, _ = await git.communicate()
    metadata = {
        "seed": args.seed,
        "workload_id": digest(workload),
        "source_id": source_hash,
        "git_sha": git_output.decode().strip() or "not-yet-committed",
        "versions": versions,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "host_memory_bytes": psutil.virtual_memory().total,
        "sqlite": sqlite3.sqlite_version,
        "model": model.identity,
        "model_ready": model.ready,
        "startup_ms": startup_ms,
        "torch_threads": 2,
        "torch_interop_threads": 1,
        "protocol": "serial closed-loop; one warmup; empty per-arm response cache",
        "command": f"python scripts/benchmark.py --repeats {args.repeats} --seed {args.seed} "
        f"--arms {args.arms} --output {args.output.as_posix()}",
        "gpu_nccl_rdma_multinode": "not measured",
        "summaries": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in raw), encoding="utf-8"
    )
    (args.output / "memory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in memory), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=Path(".cache/model"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--arms", choices=["baseline", "baseline,cache", "cache,baseline"], default="baseline,cache"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    asyncio.run(main(args))
