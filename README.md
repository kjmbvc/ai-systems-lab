# AI Systems Lab

A reproducible CPU RAG pipeline and resource-bounded asyncio DAG executor.
This is the first complete experiment in a systems engineering portfolio, not a finished
distributed AI platform. It runs real, pinned FLAN-T5-small weights locally without an API key.

The pipeline collects a synthetic JSONL fixture, validates and normalizes it, removes duplicates,
atomically builds a SQLite FTS5 index, retrieves evidence, follows a bounded rule policy,
generates an answer, and evaluates retrieval and answer quality. The agent policy can
reformulate once or abstain; it is not an autonomous LLM planner.

The executor rejects cycles before effects, admits tasks by concurrency and memory reservations,
supports timeouts/retries and pure-task replay, cancels failed graphs, records execution events,
and provides a bounded producer/consumer API. Model inference is isolated in a killable process.

## Reproduce

Python 3.12 recommended. Run from the repository root. About 1 GB of public packages/weights
is downloaded; inference then runs offline on CPU. No GPU or paid inference service is used.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,model]'
python -m ai_systems_lab.cli download-model
python -m ai_systems_lab.cli demo --question "What is the purpose of a write ahead log?"
```

The download command uses ordinary files, requiring no Windows symlink privileges.
The model revision and downloaded file hashes are recorded in `.cache/model/manifest.json`.
The worker verifies these hashes and loads safetensors with remote code disabled.
Local outputs go into ignored `runs/`. Only synthetic benchmark observations are committed.

```sh
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -q -m 'not model'
python scripts/scan_secrets.py
# Linux/macOS real model test:
RUN_MODEL_TESTS=1 python -m pytest -q -m model
# PowerShell real model test:
# $env:RUN_MODEL_TESTS='1'; python -m pytest -q -m model
python scripts/benchmark.py --arms baseline,cache --repeats 5 --output runs/repeated
python scripts/benchmark.py --arms cache,baseline --repeats 5 --output runs/reversed
python scripts/benchmark.py --arms baseline,cache --repeats 1 --output runs/unique
```

`requirements-lock.txt` captures the exact measured Windows Python environment; install its
CPU torch build from the PyTorch CPU index before installing the remaining pins. It is an
environment snapshot, not a cross-platform hash-locked build. CI installs only the development
extra and runs offline tests; it does not download model weights or claim real-model CI coverage.
Only `tests/test_model.py::test_real_model_rag` proves local real-model integration.

## Evidence and architecture

- [Experiment report and raw measurements](docs/RESEARCH.md): baseline, repeated-query cache,
  reversed-order comparison and unique-query ablation. Small-sample tail estimates are labeled.
- [Mathematical assumptions → CS architecture → implementation](docs/DESIGN.md).
- [한국어 설명](docs/README.ko.md).
- [First-party references, commits and licenses](docs/REFERENCES.md).
- [Acceptance gates](ACCEPTANCE.md) and [data provenance](data/README.md).

Source map: `dag.py` scheduling/journal/backpressure; `store.py` ingestion/FTS;
`pipeline.py` policy/cache/evaluation; `model.py` subprocess boundary;
`model_worker.py` CPU inference; `scripts/benchmark.py` raw measurement.

The experiment targets warmed p95 <= 1.5 s, exact match >= 90%, retrieval hit@2 >= 95%,
zero observed errors and sampled process-tree RSS <= 2 GiB on the declared synthetic workload.
This does not establish a production SLO. Reservations are not hard RSS enforcement, retrieval
source IDs are not proof of grounded answers, and input validation is not an alignment guarantee.

## Scope and license

No training, vector DB, distributed scheduler, GPU/NCCL/RDMA, multimodal input or production
HTTP server is implemented. C++/CUDA await evidence of a suitable low-level bottleneck.
Future work begins with open-loop load, larger held-out data and better evidence/abstention
evaluation; broad physical AI, accelerator and cooperative-agent research remains a roadmap.

Code: MIT. Synthetic fixtures: CC0-1.0. Third-party dependencies and model weights retain
their own licenses. Weights are downloaded explicitly and are not redistributed in this repository.
