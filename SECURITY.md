# Security scope

This educational prototype processes public synthetic data locally. Do not expose it as
an unauthenticated service or put confidential data into the persistent result journal.
Model output is untrusted text and is never executed. The optional download step accesses
only public Hugging Face files; inference is offline. No credentials are required.

The offline credential scan checks selected high-confidence token/private-key patterns in
tracked files. It cannot prove absence of all secrets. Check staged files before publishing.
CI uses `contents: read`, SHA-pinned actions and no injected secrets. It uses `pull_request`,
not `pull_request_target`, and disables persisted checkout credentials.

Unsupported guarantees: prompt-injection resistance, safety alignment, hard process memory
limits, multi-tenant isolation, externally exactly-once effects, distributed fault tolerance.
Responsible AI work here is limited to provenance, an explicit abstention path, transparent
quality results and documented model/measurement limitations.
