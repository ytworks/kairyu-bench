"""Run one DeepSWE task with DeepSeek Harness inside the task container.

Pier uploads this file next to the hash-locked SDK venv. It may import only the
standard library and the pinned ``deepseek_harness`` package. Every outcome is
written to ``dsh-result.json``; the process exits 0 unless this script is broken.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

RESULT_FILE = "dsh-result.json"
EVENTS_FILE = "dsh-events.jsonl"
PROVIDER = "deepseek-official"
PROFILE = "sdk-minimal"
FINISHED = {"completed", "max-tokens"}


def classify(finish_reason: str | None, error: dict[str, Any] | None) -> str:
    if finish_reason in FINISHED:
        return str(finish_reason)
    if finish_reason == "error":
        code = (error or {}).get("code")
        if code == "CONTEXT_WINDOW_EXCEEDED":
            return "ContextWindowExceededError"
        return f"provider:{code if isinstance(code, str) and code else 'UNKNOWN'}"
    if finish_reason is None:
        return "MissingTurnEnd"
    return finish_reason


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    model_calls = 0
    error = None
    finish_reason = None
    for event in events:
        raw = event.get("data")
        data: dict[str, Any] = raw if isinstance(raw, dict) else {}
        kind = event.get("type")
        if kind == "assistant/message":
            model_calls += 1
        counts = data.get("usage")
        if kind in {"assistant/message", "compaction/summary"} and isinstance(counts, dict):
            usage["input_tokens"] += int(counts.get("inputTokens") or 0)
            usage["output_tokens"] += int(counts.get("outputTokens") or 0)
            usage["cache_read_tokens"] += int(counts.get("cacheReadTokens") or 0)
        reason = data.get("reason")
        if kind == "turn/end" and isinstance(reason, dict):
            finish_reason = reason.get("kind")
            reason_error = reason.get("error")
            error = reason_error if isinstance(reason_error, dict) else None
    if error is not None:
        error = {key: error[key] for key in ("code", "status", "message") if key in error}
    return {
        "finish_reason": finish_reason,
        "error": error,
        "model_calls": model_calls,
        "usage": usage,
    }


def _terminate(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--runtime-cwd", required=True)
    parser.add_argument("--patch", required=True)
    args = parser.parse_args(argv)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    dsh_home = args.logs_dir / "dsh-home"
    dsh_home.mkdir(exist_ok=True)
    effort = os.environ.get("KAIRYU_DSH_REASONING_EFFORT") or None
    events: list[dict[str, Any]] = []
    status = "MissingAgentResult"
    signal.signal(signal.SIGTERM, _terminate)
    with (args.logs_dir / EVENTS_FILE).open("a", encoding="utf-8") as stream:

        def record(notification: Any) -> None:
            if notification.method != "session.event":
                return
            event = notification.payload.get("event")
            if isinstance(event, dict):
                events.append(event)
                stream.write(json.dumps(notification.payload, sort_keys=True) + "\n")
                stream.flush()

        try:
            from deepseek_harness import DeepSeekHarness

            with DeepSeekHarness(
                provider=PROVIDER,
                model=os.environ["KAIRYU_DSH_MODEL"],
                reasoning_effort=effort,
                max_tokens=int(os.environ["KAIRYU_DSH_MAX_TOKENS"]),
                cwd=os.getcwd(),
                runtime_cwd=args.runtime_cwd,
                dsh_home=str(dsh_home),
                profile=PROFILE,
                patches=(args.patch,),
                initialize_timeout_seconds=120.0,
            ) as harness:
                harness.run(args.instruction, on_notification=record)
            outcome = summarize(events)
            status = classify(outcome["finish_reason"], outcome["error"])
        except SystemExit:
            status = "Terminated"
        except Exception as error:  # Recorded as an unscored runtime failure.
            status = f"RuntimeError:{type(error).__name__}"
            print(f"DeepSeek Harness runtime failed: {type(error).__name__}", file=sys.stderr)
    result = {"status": status, **summarize(events)}
    (args.logs_dir / RESULT_FILE).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"DeepSeek Harness finished: {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
