"""One model, one request at a time. stdout is a JSONL protocol, not a log sink."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .model import GENERATION, MODEL_ID, REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    directory = Path(args.cache_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("model manifest revision mismatch")
    for name, expected in manifest["sha256"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("model file checksum mismatch")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    common = {
        "local_files_only": True,
        "trust_remote_code": False,
        "token": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.cache_dir, **common)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.cache_dir, use_safetensors=True, **common)
    model.to("cpu").eval()
    print(
        json.dumps(
            {
                "ready": True,
                "load_ms": (time.perf_counter() - started) * 1000,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            prompt = (
                f"Answer the question using the context.\nContext: {request['context']}\n"
                f"Question: {request['question']}\nAnswer:"
            )
            start = time.perf_counter()
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            tokenized = time.perf_counter()
            with torch.inference_mode():
                encoder = model.get_encoder()(**encoded)
                encoded_at = time.perf_counter()
                output = model.generate(
                    encoder_outputs=encoder, attention_mask=encoded["attention_mask"], **GENERATION
                )
            decoded_at = time.perf_counter()
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            value = {
                "answer": answer,
                "input_tokens": int(encoded["input_ids"].shape[1]),
                "output_tokens": int(output.shape[1] - 1),
                "tokenize_ms": (tokenized - start) * 1000,
                "encoder_ms": (encoded_at - tokenized) * 1000,
                "decode_ms": (decoded_at - encoded_at) * 1000,
            }
        except Exception as exc:
            value = {"error": type(exc).__name__}
        print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
