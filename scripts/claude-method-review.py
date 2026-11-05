#!/usr/bin/env python3
"""One explicit, anonymous-method Claude review; no repository tools or raw stream persistence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

SYSTEM_PROMPT = (
    "You are a methodological reviewer. Use only the anonymous design text supplied by the user. "
    "Do not access tools, files, repositories, memories or external resources. "
    "Reply with the requested JSON and no private data."
)
USAGE_FIELDS = (
    "inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens",
    "webSearchRequests", "costUSD", "contextWindow", "maxOutputTokens", "canonicalModel",
)
RESPONSE_FIELDS = ("reviews", "sequence", "evidence_limits", "fresh_reserve")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_prompt(raw: bytes) -> str:
    if not raw or len(raw) > 65536:
        raise ValueError("prompt must contain 1 through 65536 UTF-8 bytes")
    prompt = raw.decode("utf-8")
    forbidden = (
        r"https?://|file://|/(?:home|mnt|Users|private|tmp|etc)/|[A-Za-z]:\\",
        r"(?i)\b(?:sk-|ghp_|hf_)[A-Za-z0-9_-]{12,}",
        r"(?i)(?:authorization\s*:|bearer\s+\S+|(?:api[_-]?key|password|secret)\s*[:=])",
    )
    if "\x00" in prompt or any(re.search(pattern, prompt) for pattern in forbidden):
        raise ValueError("prompt contains a path, URL or credential-like content; review locally")
    return prompt


def parse_stream(raw: str, expected_model: str, labels: str) -> dict[str, Any]:
    if len(raw.encode()) > 8 * 1024**2:
        raise ValueError("unexpectedly large CLI stream")
    models: set[str] = set()
    kinds: dict[str, int] = {}
    tool_uses = 0
    final: dict[str, Any] | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("CLI event must be an object")
        kind = str(event.get("type", "unknown"))
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "assistant":
            message = event.get("message", {})
            model = message.get("model")
            if isinstance(model, str):
                models.add(model)
            tool_uses += sum(block.get("type") == "tool_use" for block in message.get("content", []))
            # Thinking/text blocks and machine-specific system events are never persisted.
        elif kind == "result":
            if final is not None:
                raise ValueError("multiple final results")
            final = event
    final = final or {}
    usage = final.get("modelUsage", final.get("model_usage", {}))
    if not isinstance(usage, dict):
        raise ValueError("invalid model usage")
    answer = final.get("result", "")
    marker = chr(96) * 3
    clean = re.sub(r"^\s*" + marker + r"(?:json)?\s*|\s*" + marker + r"\s*$", "", str(answer).strip())
    try:
        response = json.loads(clean)
    except json.JSONDecodeError:
        response = None
    reviews = response.get("reviews", []) if isinstance(response, dict) else []
    actual_labels = [row.get("label") for row in reviews if isinstance(row, dict)]
    label_match = (
        len(actual_labels) == len(reviews) == len(labels)
        and all(isinstance(label, str) for label in actual_labels)
        and sorted(actual_labels) == sorted(labels)
    )
    identity_match = (
        models == {expected_model}
        and set(usage) == {expected_model}
        and isinstance(usage.get(expected_model), dict)
        and usage[expected_model].get("canonicalModel", expected_model) == expected_model
    )
    verified = final.get("is_error") is False and identity_match and label_match and tool_uses == 0
    safe_usage = {
        model: {key: value for key, value in values.items() if key in USAGE_FIELDS}
        for model, values in usage.items() if isinstance(values, dict)
    }
    return {
        "assistant_models": sorted(models), "model_usage": safe_usage,
        "verified_requested_model": verified, "is_error": final.get("is_error"),
        "labels_match": label_match, "tool_uses": tool_uses, "event_types": kinds,
        # Preserve only final, requested review fields, never raw failed/error output.
        "response": {key: response[key] for key in RESPONSE_FIELDS if key in response}
        if isinstance(response, dict) else None,
    }


def review(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.prompt.read_bytes()
    prompt = validate_prompt(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if not re.fullmatch(r"[A-Z]{1,26}", args.labels) or len(set(args.labels)) != len(args.labels):
        raise ValueError("labels must be unique uppercase letters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{1,100}", args.model):
        raise ValueError("use an explicit model identifier")
    if not 1 <= args.timeout_seconds <= 900 or not 0 < args.max_budget_usd <= 100:
        raise ValueError("timeout or budget is outside the bounded range")
    preview = {
        "mode": "send" if args.send else "preview_only", "prompt_sha256": digest,
        "prompt_bytes": len(raw), "requested_model": args.model, "labels": args.labels,
        "anonymity": "pattern lint only; caller must review every prompt byte for private identifiers and measurements",
    }
    if not args.send:
        return preview
    if args.approved_prompt_sha256 != digest:
        raise ValueError("--send requires the locally reviewed prompt SHA256")
    # A distinct directory is a one-dispatch guard, including failed/timeout calls.
    args.output.mkdir(parents=True, exist_ok=False)
    for directory in ("empty-work", "tmp", "cache"):
        (args.output / directory).mkdir()
    (args.output / "prompt.txt").write_bytes(raw)
    (args.output / "dispatch.json").write_text(json.dumps(preview, indent=2) + "\n")
    command = [
        str(args.claude_binary), "--safe-mode", "--restricted", "--print", "--model", args.model,
        "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        "--system-prompt-snapshot", "off", "--tools", "", "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--no-chrome",
        "--permission-mode", "dontAsk", "--permission-prompts", "none", "--setting-sources", "",
        "--effort", "high", "--max-budget-usd", str(args.max_budget_usd),
        "--system-prompt", SYSTEM_PROMPT,
    ]
    env = os.environ.copy()
    env["TMPDIR"] = str((args.output / "tmp").resolve())
    env["XDG_CACHE_HOME"] = str((args.output / "cache").resolve())
    started = time.monotonic()
    receipt: dict[str, Any] = {
        **preview, "schema_version": 1, "started_at": timestamp(),
        "scope": "anonymous public-method design only; not an actual-candidate measurement review",
        "verified_requested_model": False, "fallback_model": None,
        "controls": {"safe_mode": True, "restricted": True, "tools": [], "mcp_servers": [],
                     "session_persistence": False, "custom_generic_system_prompt": True},
        "raw_stream_persisted": False, "thinking_blocks_persisted": False, "environment_persisted": False,
    }
    try:
        process = subprocess.run(
            command, input=prompt, text=True, capture_output=True,
            cwd=args.output / "empty-work", env=env, timeout=args.timeout_seconds, check=False,
        )
        receipt["returncode"] = process.returncode
        receipt["stderr_bytes"] = len(process.stderr.encode())
        receipt["stderr_sha256"] = hashlib.sha256(process.stderr.encode()).hexdigest()
        receipt.update(parse_stream(process.stdout, args.model, args.labels))
        receipt["verified_requested_model"] &= process.returncode == 0
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as error:
        # Do not persist exception messages: they can include command/output/authentication data.
        receipt["error_class"] = type(error).__name__
    receipt["finished_at"] = timestamp()
    receipt["duration_seconds"] = time.monotonic() - started
    temp = args.output / "receipt.tmp"
    temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output / "receipt.json")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--claude-binary", type=Path, default=Path("claude"))
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--labels", default="ABCDEFGH")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-budget-usd", type=float, default=2)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--approved-prompt-sha256")
    args = parser.parse_args()
    try:
        result = review(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: value for key, value in result.items() if key != "response"}, ensure_ascii=False))
    return 0 if not args.send or result["verified_requested_model"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
