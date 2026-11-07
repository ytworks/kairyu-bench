#!/usr/bin/env python3
"""Bounded, opt-in vLLM serving counters. No benchmark scoring or content."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_BYTES = 2 * 1024**2
KINDS = {
    "vllm:num_requests_running": "gauge",
    "vllm:num_requests_waiting": "gauge",
    "vllm:generation_tokens_total": "counter",
    "vllm:prompt_tokens_total": "counter",
    "vllm:request_success_total": "counter",
    "process_start_time_seconds": "identity",
}
LIMITS = [
    "Counters cover the endpoint process and may include other clients, smoke tests and retries.",
    "No exclusive-client, per-item, retry-count, thought-token or benchmark-score attribution is proven.",
    "Container/model identity is caller-declared; optional attestation SHA is a reference, not verified here.",
    "Without process_start_time_seconds, monotonic counters do not rule out a process restart.",
    "Matching process start/counters cannot prove the absence of an unobserved reset.",
    "Missing series are unknown, never zero. Gauges have before/after values, not counter deltas.",
]


def endpoint_url(value: str) -> str:
    if any(ord(c) <= 32 or ord(c) == 127 for c in value):
        raise ValueError("endpoint contains whitespace/control characters")
    parts = urlsplit(value)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or "@" in parts.netloc):
        raise ValueError("endpoint must be HTTP(S), without credentials/query/fragment")
    port = parts.port
    host = parts.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    netloc = host + (":" + str(port) if port is not None else "")
    return urlunsplit((parts.scheme, netloc, parts.path or "/metrics", "", ""))


def identity(endpoint: str, model: str, container_id: str,
             attestation_sha256: str | None = None) -> dict:
    if not model or len(model) > 256 or any(ord(c) <= 32 or ord(c) == 127 for c in model):
        raise ValueError("expected model must be a bounded identifier")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", container_id):
        raise ValueError("container identity is required")
    if attestation_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", attestation_sha256):
        raise ValueError("attestation must be a SHA256 reference")
    return {"endpoint": endpoint_url(endpoint), "model": model, "container_id": container_id,
            "identity_kind": "caller_declared", "attestation_sha256": attestation_sha256}


def number(value: str) -> Decimal:
    if not isinstance(value, str) or len(value) > 128 or not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        raise ValueError("invalid numeric sample")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise ValueError("invalid numeric sample") from None
    if (not result.is_finite() or result < 0 or abs(int(result.as_tuple().exponent)) > 1024
            or not math.isfinite(float(result))):
        raise ValueError("sample must be finite and nonnegative")
    return result


LABEL = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\["\\n])*)"\s*')


def labels(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    pos = 0
    while pos < len(text):
        match = LABEL.match(text, pos)
        if match is None:
            raise ValueError("invalid Prometheus labels")
        key, raw = match.groups()
        if key in result:
            raise ValueError("duplicate label")
        result[key] = re.sub(r'\\(["\\n])', lambda m: "\n" if m[1] == "n" else m[1], raw)
        pos = match.end()
        if pos == len(text):
            break
        if text[pos] != ",":
            raise ValueError("invalid label separator")
        pos += 1
        if not text[pos:].strip():
            break  # Prometheus permits a trailing comma.
    return result


def series_key(row: dict) -> str:
    return json.dumps([row["metric"], row["labels"]], sort_keys=True, separators=(",", ":"))


def validate_row(row: dict, model: str) -> dict:
    metric = row["metric"]
    if metric not in KINDS or row.get("kind") != KINDS[metric]:
        raise ValueError("unknown metric or kind")
    lab = row["labels"]
    if not isinstance(lab, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in lab.items()):
        raise ValueError("invalid labels")
    allowed = set() if metric == "process_start_time_seconds" else {"model", "model_name", "engine"}
    if metric == "vllm:request_success_total":
        allowed.add("finished_reason")
        if lab.get("finished_reason") not in {"stop", "length", "abort", "error", "repetition"}:
            raise ValueError("unknown/missing finished reason")
    if set(lab) - allowed:
        raise ValueError("unsupported metric labels")
    if metric.startswith("vllm:"):
        model_labels = [lab[k] for k in ("model", "model_name") if k in lab]
        if not model_labels or any(v != model for v in model_labels):
            raise ValueError("missing/mixed model series")
    if "engine" in lab and not re.fullmatch(r"\d{1,16}", lab["engine"]):
        raise ValueError("unsupported engine label")
    value = number(row["value"])
    return {"metric": metric, "kind": KINDS[metric], "labels": dict(sorted(lab.items())), "value": str(value)}


def parse_metrics(body: bytes, model: str) -> list[dict]:
    if not body or len(body) > MAX_BYTES:
        raise ValueError("empty or oversized metrics")
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError("metrics must be UTF-8") from None
    found: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([A-Za-z_:][A-Za-z0-9_:]*)(.*)$", line)
        if match is None:
            raise ValueError("invalid metrics body")
        metric, rest = match.groups()
        if metric not in KINDS:
            if not rest or rest[0] not in "{ \t":
                raise ValueError("invalid metrics body")
            continue  # Unknown metric values/labels are never retained.
        lab = {}
        if rest.startswith("{"):
            quoted = escaped = False
            end = None
            for index, char in enumerate(rest[1:], 1):
                if escaped:
                    escaped = False
                elif char == "\\" and quoted:
                    escaped = True
                elif char == '"':
                    quoted = not quoted
                elif char == "}" and not quoted:
                    end = index
                    break
            if end is None:
                raise ValueError("unterminated labels")
            lab = labels(rest[1:end])
            rest = rest[end + 1:]
        fields = rest.split()
        if len(fields) not in (1, 2):
            raise ValueError("invalid sample fields")
        if len(fields) == 2:
            number(fields[1])  # Validate optional exposition timestamp; do not retain.
        row = validate_row({"metric": metric, "kind": KINDS[metric], "labels": lab,
                            "value": fields[0]}, model)
        key = series_key(row)
        if key in found:
            raise ValueError("duplicate metric series")
        found[key] = row
    if not found:
        raise ValueError("no allowed metric series")
    return [found[k] for k in sorted(found)]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("metrics redirects are refused")


def capture(source: dict, timeout: float = 5.0) -> dict:
    source = identity(source["endpoint"], source["model"], source["container_id"],
                      source.get("attestation_sha256"))
    if not math.isfinite(timeout) or not 0 < timeout <= 10:
        raise ValueError("timeout must be in (0,10] seconds")
    start = time.monotonic()
    request = Request(source["endpoint"], headers={"Accept": "text/plain"})
    with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        body = response.read(MAX_BYTES + 1)
    rows = parse_metrics(body, source["model"])
    return {"schema_version": 1, "kind": "serving_counter_snapshot",
            "captured_utc": datetime.now(timezone.utc).isoformat(), "source": source,
            "request_seconds": time.monotonic() - start,
            "http_requests": 1, "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
            "series": rows, "missing_metric_families": sorted(set(KINDS) - {r["metric"] for r in rows}),
            "limits": LIMITS}


def snapshot_rows(snapshot: dict) -> dict[str, dict]:
    if snapshot.get("schema_version") != 1 or snapshot.get("kind") != "serving_counter_snapshot":
        raise ValueError("unsupported snapshot")
    src = snapshot["source"]
    identity(src["endpoint"], src["model"], src["container_id"], src.get("attestation_sha256"))
    if not isinstance(snapshot["series"], list) or not snapshot["series"]:
        raise ValueError("missing snapshot series")
    result = {}
    for row in snapshot["series"]:
        checked = validate_row(row, src["model"])
        key = series_key(checked)
        if key in result:
            raise ValueError("duplicate snapshot series")
        result[key] = checked
    return result


def compare(before: dict, after: dict) -> dict:
    left, right = snapshot_rows(before), snapshot_rows(after)
    raw_source = before["source"]
    source = identity(raw_source["endpoint"], raw_source["model"], raw_source["container_id"],
                      raw_source.get("attestation_sha256"))
    for field in ("endpoint", "model", "container_id"):
        if before["source"][field] != after["source"][field]:
            raise ValueError("snapshot identities differ")
    start = datetime.fromisoformat(before["captured_utc"])
    end = datetime.fromisoformat(after["captured_utc"])
    if start.tzinfo is None or end.tzinfo is None or end < start:
        raise ValueError("snapshot times are invalid/reversed")
    missing_before, missing_after = sorted(set(right) - set(left)), sorted(set(left) - set(right))
    comparable = not missing_before and not missing_after
    for key in set(left) & set(right):
        a, b = number(left[key]["value"]), number(right[key]["value"])
        if left[key]["kind"] == "identity" and a != b:
            raise ValueError("process start changed; reset suspected")
        if left[key]["kind"] == "counter" and b < a:
            raise ValueError("counter decreased; reset suspected")
    rows = []
    for key in sorted(set(left) | set(right)):
        reference = left[key] if key in left else right[key]
        row_a, row_b = left.get(key), right.get(key)
        value = None
        if comparable and reference["kind"] == "counter":
            assert row_a is not None and row_b is not None
            left_value, right_value = number(row_a["value"]), number(row_b["value"])
            with localcontext() as context:
                context.prec = max(left_value.adjusted(), right_value.adjusted()) - min(
                    int(left_value.as_tuple().exponent), int(right_value.as_tuple().exponent)) + 2
                value = str(right_value - left_value)
        rows.append({"metric": reference["metric"], "kind": reference["kind"],
                     "labels": reference["labels"], "before": row_a["value"] if row_a else None,
                     "after": row_b["value"] if row_b else None, "counter_delta": value})
    return {"schema_version": 1, "kind": "serving_counter_diff", "source": source,
            "before_utc": before["captured_utc"], "after_utc": after["captured_utc"],
            "seconds": (end - start).total_seconds(), "comparable": comparable,
            "status": "counter_delta_only" if comparable else "unknown_missing_series",
            "missing_before": missing_before, "missing_after": missing_after,
            "process_start_available": any(r["kind"] == "identity" for r in rows),
            "series": rows, "limits": LIMITS}


def write_fresh(path: Path, report: dict, protected: tuple[Path, ...] = ()) -> None:
    if path.resolve() in {p.resolve() for p in protected} or os.path.lexists(path):
        raise ValueError("output must be a fresh path outside inputs")
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--endpoint", required=True, help="Explicit metrics URL; empty path defaults to /metrics")
    snap.add_argument("--model", required=True)
    snap.add_argument("--container-id", required=True, help="Caller-declared; not inspected automatically")
    snap.add_argument("--attestation-sha256")
    snap.add_argument("--timeout", type=float, default=5)
    snap.add_argument("--output", type=Path, required=True)
    snap.add_argument("--capture", action="store_true")
    diff = sub.add_parser("diff")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "snapshot":
            src = identity(args.endpoint, args.model, args.container_id, args.attestation_sha256)
            if not math.isfinite(args.timeout) or not 0 < args.timeout <= 10:
                raise ValueError("invalid timeout")
            if os.path.lexists(args.output):
                raise ValueError("output already exists")
            if not args.capture:
                print(json.dumps({"status": "preview", "source": src, "output": str(args.output),
                                  "http_requests": 0, "writes": 0, "max_bytes": MAX_BYTES}))
                return 0
            report = capture(src, args.timeout)
            write_fresh(args.output, report)
        else:
            before_bytes, after_bytes = args.before.read_bytes(), args.after.read_bytes()
            report = compare(json.loads(before_bytes), json.loads(after_bytes))
            report["input_sha256"] = {
                "before": hashlib.sha256(before_bytes).hexdigest(),
                "after": hashlib.sha256(after_bytes).hexdigest(),
            }
            write_fresh(args.output, report, (args.before, args.after))
        print(json.dumps({"output": str(args.output), "kind": report["kind"]}))
        return 0
    except Exception as error:
        # Do not echo HTTP error bodies, URLs from exceptions, labels or input text.
        print(json.dumps({"status": "refused", "error_type": type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
