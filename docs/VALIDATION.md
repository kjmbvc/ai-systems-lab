# Validation boundaries

Local environment: Windows 11, Intel Core 5 120U (10 physical / 12 logical CPUs),
16,764,178,432 bytes RAM, Python 3.12.10, SQLite 3.49.1. Package pins and model hashes
are recorded separately. No GPU, NCCL, RDMA or multiple nodes were measured.

The checked local suite has 31 tests: 30 offline tests and one opt-in real-model RAG test.
It checks cycles/missing dependencies/duplicate names/invalid resource declarations,
diamond dependencies, active task and memory bounds, retry and persistent pure replay,
timeout cleanup, sibling and external cancellation, queue backpressure and producer cleanup,
ingestion schema and size limits, normalization/deduplication, SQL transaction rollback,
query escaping, cache invalidation/TTL/LRU, abstention/reformulation, model process-tree
termination, and an end-to-end query with actual FLAN-T5 weights.

The model integration test checks nonempty generated output, retrieval and recorded model
timings. It does not assert a broad model-quality guarantee. The separate raw benchmark
reports exact-match and retrieval metrics for all 12 hand-authored questions.

Commands: `python -m ruff format --check .`, `python -m ruff check .`, `python -m mypy`,
`RUN_MODEL_TESTS=1 python -m pytest -q`, `python scripts/scan_secrets.py`.
On PowerShell set `$env:RUN_MODEL_TESTS='1'` before pytest. Static checking applies to the
seven local source modules; third-party library implementations are not type-checked.

The credential scan covers tracked files with high-confidence token/private-key patterns;
it does not prove complete secret absence. The working source was also inspected for local
account paths before publication. Downloaded weights, caches, environments and SQLite files
are ignored. The published dependency license inventory contains package metadata only.

CI runs on Ubuntu with Python 3.12, installing the development extra. Its model test is
explicitly deselected, not silently represented as real-model coverage. The workflow has
read-only repository permission, pinned action SHAs, no secrets and no persisted checkout
credentials. Local success, GitHub upload, remote workflow success and a GitHub Release are
separate states; see the completion report for the verified publication state.
