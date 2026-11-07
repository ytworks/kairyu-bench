#!/usr/bin/env python3
"""Generate public science answers; preview by default, no answer execution."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import urllib.request


def fingerprint(path: Path) -> dict:
    value = path.stat()
    return {
        key: getattr(value, "st_" + key)
        for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")
    }


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_cases(document: dict, split: str) -> list[dict]:
    groups = document.get("cases", document)
    rows = (
        groups
        if isinstance(groups, list)
        else [row for folds in groups.values() for fold in folds for row in fold]
    )
    if not rows or len(rows) > 256:
        raise ValueError("cases must contain 1..256 explicitly selected prompts")
    seen = set()
    public = []
    for row in rows:
        if row.get("split") != split:
            raise ValueError("all case splits must match --split")
        key, prompt = row.get("id"), row.get("prompt")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("case ID must be unique and nonempty")
        if not isinstance(prompt, str) or not prompt or len(prompt.encode()) > 65536:
            raise ValueError("missing or oversized public prompt")
        seen.add(key)
        public.append({"id": key, "prompt": prompt})
    return public


def save(path: Path, document: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def generate(row: dict, args, opener=urllib.request.urlopen) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    # Temperature/top_p deliberately omitted: match the server generation config.
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with opener(request, timeout=args.timeout) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("response exceeds bounded public-answer envelope")
    document = json.loads(raw)
    if len(document.get("choices", [])) != 1:
        raise ValueError("exactly one generated choice required")
    choice = document["choices"][0]
    answer = (choice.get("message") or {}).get("content")
    if not isinstance(answer, str) or len(answer.encode()) > 262144:
        raise ValueError("missing or oversized public answer")
    usage = document.get("usage") or {}
    counters = {}
    for target, source in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
    ):
        value = usage.get(source)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("invalid token counter")
        counters[target] = value
    request_id, finish = document.get("id"), choice.get("finish_reason")
    if any(
        value is not None and not isinstance(value, str)
        for value in (request_id, finish)
    ):
        raise ValueError("invalid response metadata")
    # Never return/log the complete response: reasoning fields are discarded.
    return {
        "id": row["id"],
        "public_answer": answer,
        "request_id": request_id,
        "finish_reason": finish,
        "usage": counters,
        "seconds": time.monotonic() - started,
    }


def main(argv=None, opener=urllib.request.urlopen):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "transfer"), required=True)
    parser.add_argument(
        "--endpoint", required=True, help="explicit OpenAI-compatible /v1 endpoint"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    url = urlsplit(args.endpoint)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("explicit HTTP endpoint without embedded credentials required")
    if not re.fullmatch("[0-9a-f]{64}", args.checkpoint_sha256):
        raise ValueError("checkpoint SHA256 required")
    if not (
        1 <= args.concurrency <= 32
        and 1 <= args.max_tokens <= 32768
        and math.isfinite(args.timeout)
        and 1 <= args.timeout <= 1200
    ):
        raise ValueError("invalid request bounds")
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    path = args.cases.resolve()
    state = fingerprint(path)
    if state["size"] > 8 * 1024 * 1024:
        raise ValueError("case file exceeds limit")
    data = path.read_bytes()
    if fingerprint(path) != state or len(data) != state["size"]:
        raise ValueError("case input changed during read")
    rows = public_cases(json.loads(data), args.split)
    source = Path(__file__).resolve()
    source_state, source_hash = fingerprint(source), digest(source.read_bytes())
    result = {
        "schema_version": 1,
        "split": args.split,
        "cases_sha256": digest(data),
        "generation": {
            "model_id": args.model,
            "checkpoint_sha256": args.checkpoint_sha256,
            "frame": "live-native-thinking-on-public-answer",
            "enable_thinking": True,
            "reasoning_saved": False,
            "sampling": "server_official_generation_config",
            "temperature": None,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "responses": [],
    }
    receipt = {
        "status": "preview",
        "cases": len(rows),
        "endpoint": args.endpoint,
        "case_path": str(path),
        "case_stat": state,
        "cases_sha256": digest(data),
        "source_sha256": source_hash,
        "source_stat": source_state,
        "checkpoint_identity": "caller-supplied; establish serving identity separately",
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "retries": 0,
        "hidden_reasoning_saved": False,
        "answer_execution": False,
    }
    args.output.mkdir(parents=True)
    save(args.output / "receipt.json", receipt)
    if not args.run:
        return receipt
    output = args.output / (args.split + "-responses.json")
    receipt["status"] = "running"
    save(args.output / "receipt.json", receipt)
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(generate, row, args, opener): row["id"] for row in rows
            }
            try:
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except Exception as error:
                        row = {
                            "id": futures[future],
                            "error_type": type(error).__name__,
                        }
                    result["responses"].append(row)
                    save(output, result)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if (
            fingerprint(path) != state
            or digest(path.read_bytes()) != digest(data)
            or fingerprint(source) != source_state
            or digest(source.read_bytes()) != source_hash
        ):
            raise ValueError("input or source changed during generation")
        receipt["status"] = "completed"
    except BaseException:
        receipt["status"] = "interrupted_or_error"
        raise
    finally:
        receipt["received"] = len(result["responses"])
        receipt["errors"] = sum("error_type" in row for row in result["responses"])
        save(args.output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    result = main()
    print(
        json.dumps(
            {key: result.get(key) for key in ("status", "cases", "received", "errors")}
        )
    )
