"""Small offline high-confidence credential scan of tracked files; not a security guarantee."""

import re
import subprocess
from pathlib import Path

PATTERNS = [
    re.compile(rb"gh[pousr]_" + rb"[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_" + rb"[A-Za-z0-9_]{40,}"),
    re.compile(rb"AKIA" + rb"[A-Z0-9]{16}"),
    re.compile(rb"sk-" + rb"[A-Za-z0-9_-]{32,}"),
    re.compile(rb"-----BEGIN " + rb"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"hf_" + rb"[A-Za-z0-9]{30,}"),
]


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    findings = []
    for name in filter(None, files):
        data = Path(name).read_bytes()
        if any(pattern.search(data) for pattern in PATTERNS):
            findings.append(name)
    if findings:
        raise SystemExit("Potential credential in: " + ", ".join(findings))
    print(f"PASS: scanned {len(list(filter(None, files)))} tracked files; no credential patterns")


if __name__ == "__main__":
    main()
