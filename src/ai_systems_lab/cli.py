from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from .dag import Journal
from .model import MODEL_ID, REVISION, LocalModel
from .pipeline import Pipeline
from .store import Store


async def demo(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.run_dir / "corpus.sqlite")
    journal = Journal(args.run_dir / "journal.sqlite")
    model = LocalModel(args.model_cache)
    try:
        pipeline = Pipeline(store, journal, model, cache=True)
        ingest = await pipeline.ingest(args.data / "documents.jsonl")
        result = await pipeline.query(args.question)
        report = {"model": model.identity, "ingest": ingest, "result": result}
        (args.run_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        await model.close()
        store.close()
        journal.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser(
        "download-model", help="Download pinned public weights; no inference API"
    )
    download.add_argument("--model-cache", type=Path, default=Path(".cache/model"))
    run = sub.add_parser("demo")
    run.add_argument("--model-cache", type=Path, default=Path(".cache/model"))
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--run-dir", type=Path, default=Path("runs/demo"))
    run.add_argument("--question", default="What is the purpose of a write ahead log?")
    args = parser.parse_args()
    if args.command == "download-model":
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            local_dir=str(args.model_cache),
            token=False,
            max_workers=1,
            allow_patterns=["*.json", "*.model", "model.safetensors", "README.md"],
        )
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.model_cache.iterdir()
            if p.is_file() and p.name != "manifest.json"
        }
        (args.model_cache / "manifest.json").write_text(
            json.dumps({"model": MODEL_ID, "revision": REVISION, "sha256": files}, indent=2),
            encoding="utf-8",
        )
        print(f"Downloaded {MODEL_ID}@{REVISION}")
    else:
        asyncio.run(demo(args))


if __name__ == "__main__":
    main()
