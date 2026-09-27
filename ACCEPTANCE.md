# First increment: acceptance gates

1. Validate DAG structure before side effects; enforce active task and declared memory bounds.
2. Demonstrate retries, cooperative timeout, cancellation cleanup, downstream failure,
   bounded producer backpressure and versioned replay of pure tasks.
3. Ingest only checked-in synthetic data, validate, normalize, deduplicate and atomically
   replace the SQLite FTS5 corpus; retrieve, route, run a real CPU model and evaluate.
4. Distinguish deterministic test doubles from actual FLAN-T5 weights. Never call a paid API.
5. Profile a baseline before choosing an optimization; preserve raw samples, environment,
   seed, workload hash, versions, commands, quality and memory. Flag small tail samples.
6. Pass formatting, lint, typing, unit/integration/E2E/failure tests and a secret scan.
7. Publish only this new dedicated public repository; verify remote SHA, visibility and CI.
   Preserve local deliverables if publication is blocked. No force pushes.
8. Document assumptions, resource model, causal system design, citations, ablation and limits
   in English and Korean. No distributed/GPU performance or broad safety claims.
