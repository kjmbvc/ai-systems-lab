# First-party references and provenance

Verified 2026-09-27 UTC. Versioned source references precede framework adoption.
This project calls public APIs and contains original glue code; model weights and third-party
source trees are not vendored. `dependency-licenses.json` records installed package metadata;
`requirements-lock.txt` captures the measured environment. Metadata is not a legal audit.

| Component | Version / resolved commit | License / source structure used |
|---|---|---|
| CPython | Runtime recorded per benchmark (3.12) | PSF; [asyncio task cancellation](https://docs.python.org/3.12/library/asyncio-task.html), [bounded queues](https://docs.python.org/3.12/library/asyncio-queue.html), `Lib/asyncio` |
| SQLite | Runtime `sqlite` field per benchmark | [Public domain](https://www.sqlite.org/copyright.html); [FTS5](https://www.sqlite.org/fts5.html), `ext/fts5`, transactions + lexical ranking |
| Transformers | 4.56.2; `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.56.2/LICENSE); [`src/transformers/models/t5/modeling_t5.py`](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py), `generation/utils.py` |
| PyTorch | 2.8.0+cpu; `ba56102387ef21a3b04b357e5b183d48f0afefc7` | [BSD-style license with notices](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE); `torch`, `aten/src/ATen` CPU tensor operations |
| FLAN-T5-small | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` | [Model card / Apache-2.0](https://huggingface.co/google/flan-t5-small/tree/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab); explicit CPU support, pretrained encoder-decoder weights |
| Synthetic corpus | Corpus digest in benchmark | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/); authored fixture, `data/README.md` |

The Transformers tag object was `04a6c04a33fd1fab9a12f95ccb9c6be0004e487a`;
the table records the resolved commit, not the annotated tag object. Action SHAs are pinned
in `.github/workflows/ci.yml` from the official `actions/checkout` and `actions/setup-python`
repositories. A source commit does not prove wheel reproducibility; package versions and
downloaded model file SHA256 hashes provide additional, narrower provenance.

## Learning connections (not fabricated quotations)

- [DDIA official author site](https://dataintensive.net/): use reliability, scalability,
  maintainability and trade-off reasoning to motivate transaction/replay boundaries.
  Only the public overview was checked; no unseen page numbers or book passages are cited.
- [MIT 6.5840 official course](https://pdos.csail.mit.edu/6.824/): distributed systems learning
  reference for the next iteration's failures and coordination questions. This implementation
  is single-node and does not claim to implement its distributed labs or consensus algorithms.
- [CMU 15-445/645 official course](https://15445.courses.cs.cmu.edu/fall2026/): the published
  scope includes storage, indexing, transactions, recovery and query processing. Connect
  those topics to SQLite snapshot replacement and retrieval, without claiming a new DBMS.

Related research: [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
is linked by the FLAN-T5 model card. Its reported model results are not reproduced here.
This repository evaluates only a 12-question synthetic CPU task and a cache ablation.
