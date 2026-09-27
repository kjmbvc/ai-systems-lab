# CPU RAG response-cache experiment

## [Mathematical Rigor]

H1: avoid repeated model computation with a bounded response cache, preserving deterministic
answers. H2: without reuse, cache keying/insertion cannot provide a model-work reduction and
may add overhead. The objective, assumptions and provisional constraints are in `DESIGN.md`.
This is a small systems experiment, not a paper submission or a general model benchmark.

Seed 17; 12 unique synthetic questions, repeated five times and shuffled identically between
arms (60 requests/arm). Each arm starts with an empty cache. One model warmup precedes both
arms. No extra response-cache priming occurs. The unique-query ablation has one of each
question. Greedy answers are deterministic in these observed runs. All raw results are retained.

| Run / arm | n | Mean ms | p50 ms | p95 ms | p99 ms | Requests/s | Hits / model calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repeated / baseline | 60 | 610.45 | 562.41 | 876.94 | 946.00 | 1.638 | 0 / 60 |
| Repeated / cache | 60 | 122.63 | 1.55 | 690.35 | 886.42 | 8.154 | 48 / 12 |
| Reversed order / cache | 60 | 113.04 | 1.48 | 685.16 | 845.55 | 8.845 | 48 / 12 |
| Reversed order / baseline | 60 | 1384.35 | 362.96 | 787.80 | 59808.51 | 0.722 | 0 / 60 |
| Unique / baseline | 12 | 312.54 | 264.67 | 554.57 | 554.57 | 3.199 | 0 / 12 |
| Unique / cache | 12 | 273.72 | 230.37 | 557.27 | 557.27 | 3.653 | 0 / 12 |

Primary repeated-run mean latency fell 79.91% (4.98x throughput ratio approximately).
The cache avoided 48/60 model calls. Quality remained exact match 11/12 = 91.67% and
retrieval hit@2 12/12 = 100% in every arm. Observed query error rate was 0 in all arms.
The mismatch is `idempotency key` vs reference `An idempotency key`; the metric strips case
and punctuation, not articles. We report it as a mismatch rather than adjusting the metric.
Both repeated arms' sampled process-tree RSS peaks were about 705 MiB, below the 2 GiB
experimental target. This is not a hard memory cap or exclusive/private memory measure.

**Small tails and outlier:** each arm has far fewer than 1000 samples, so nearest-rank p99
is descriptive, not a reliable population tail estimate. The reversed baseline contains a
59,808.51 ms request at index 24 (`q05`), with 59,672.60 ms recorded in decoder generation.
The parent memory sampler had a simultaneous maximum gap of 59.1765 seconds. This suggests
a host pause or shared scheduling disruption, but its cause was not established. We preserve
the sample. No mean or p99 derived from this arm is used as a stable optimization estimate.
The sampled p95 targets passed, but a production tail guarantee would be unjustified.

**Ablation that did not resolve overhead:** the unique cache arm had zero hits and made
the same number of model calls, yet ran faster on average. That cannot be attributed to
avoided inference. Order, warmup and host variability dominate such a small comparison.
Its p95 was slightly worse. H2's isolated overhead was therefore not established; do not
claim a zero-reuse cache speedup. A next experiment should randomize interleaved treatments,
repeat independent runs and measure lookup/insertion CPU time explicitly.

## [CS Core & Backend Architecture]

The exploratory `benchmarks/profile` run preceded optimization selection and had overlapping
local validation activity. It is a bottleneck hint, not the controlled comparison: mean total
697.89 ms, tokenization 1.88 ms, encoder 194.54 ms, decoder generation 469.55 ms.
The model dominates this profile, motivating final-answer reuse instead of a Python/C++ rewrite.

The main runs used the same implementation commit, source digest and corpus. Database ingestion
is recorded separately and excluded from per-query latency. Per-request latency includes
retrieval, policy, journal work, cache keying and model IPC when applicable. The journal and
FTS database are in-memory for these benchmarks; the CLI also supports file-backed databases.
Do not extrapolate these measurements to durable disk I/O or concurrent service throughput.

Memory is the sum of parent and descendant process RSS, sampled nominally every 20 ms.
Shared pages can be counted more than once; short peaks or pauses can be missed. Raw samples
include startup; per-arm summaries report only the arm interval. Startup is explicitly separate
(15.67 seconds for the primary run, including process/import/checksum/model loading).
The model's own 1.10-second load measurement excludes imports and checksum verification.

## [Low-Level Code & Open Source Reference]

Actual model: FLAN-T5-small, 76,961,152 parameters, FP32 CPU, pinned revision
`0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`. CPU: Intel Core 5 120U, 10 physical / 12 logical
cores; host RAM 16,764,178,432 bytes. Python 3.12.10, Torch 2.8.0+cpu, Transformers 4.56.2,
SQLite 3.49.1. Two torch intra-op threads and one inter-op thread. No paid inference.

Measured implementation commit: `f5001ad0e91f432c91aa868eb9e9e69a74176856`.
Primary workload ID: `66f3e48b8b11ed0bf04f9112a32fe8adf944437e682d0facb7a1fc8b8d1d6da7`.
The source byte digest and complete environment are in each `summary.json`. Raw files:

- `benchmarks/repeated/{summary.json,requests.jsonl,memory.jsonl,events-*.json}`
- `benchmarks/reversed/{summary.json,requests.jsonl,memory.jsonl,events-*.json}`
- `benchmarks/unique/{summary.json,requests.jsonl,memory.jsonl,events-*.json}`
- `benchmarks/profile/` is exploratory, with a pre-commit source hash.

Reproduction commands are embedded in each summary and the main README. Model file hashes
are in `docs/model-manifest.json`. See `REFERENCES.md` for official source/license provenance.

## Limitations and next hypothesis

These are correlated repetitions of 12 synthetic questions, not 60 independent quality items.
No confidence interval, saturation limit, open-loop arrival process, TTFT, per-token latency,
CPU-energy metric, allocator-private memory metric or durable database benchmark is measured.
The RAG graph is linear, so its observed active node peak is one; branching/resource scheduling
is verified separately with unit tests, not demonstrated by an inference speedup claim.
No batching, quantization, retrieval cache or KV prefix cache was implemented or measured.
No GPU/NCCL/RDMA/multi-node results exist. C++/CUDA require a new profiling justification.

Next priority: larger held-out public data plus an open-loop bounded request driver. Test
whether arrival-rate pressure worsens cache-miss p95/p99 and abstention quality. Collect enough
samples, arrival timestamps, queue wait and service time, and separate host disruptions before
deciding on batching or a different retrieval/model architecture. Broader multimodal,
heterogeneous acceleration, distributed training and cooperative control remain future scope.
