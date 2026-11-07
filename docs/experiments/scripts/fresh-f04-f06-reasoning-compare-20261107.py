"""Read-only, append-safe F04/F06 WSP output comparison; no scoring or API requests."""
import argparse
import collections
import gzip
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
BASE = ROOT / "results/gemma4-fresh-f04-dev30-dp4-20261107-v1"
CAND = ROOT / "results/gemma4-fresh-f06-dev30-dp4-20261107-v1"
CODE_FIRST = re.compile(r"^(?:```|(?:async\s+)?def\s+\w+\s*\(|class\s+\w+|import\s+\w|from\s+[\w.]+\s+import\s|#!)")
FIELDS = ("reasoning_characters", "reasoning_utf8_bytes", "content_characters", "completion_tokens")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def take(path, output):
    with path.open("rb") as stream:
        before = path.stat()
        data = stream.read(before.st_size)
        after = path.stat()
    assert len(data) == before.st_size, "file shrank during snapshot"
    if path.suffix == ".jsonl":
        end = data.rfind(b"\n") + 1
        complete, tail = data[:end], data[end:]
    else:
        complete, tail = data, b""
    destination = output / (path.name + ".gz")
    with destination.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
            z.write(data)
    rows = [json.loads(line) for line in complete.splitlines()] if path.suffix == ".jsonl" else json.loads(data)
    return rows, {"path": str(path), "raw_prefix_sha256": digest(data), "parsed_prefix_sha256": digest(complete), "bytes": len(data), "unparsed_tail_bytes": len(tail), "snapshot": str(destination), "file_size_after": after.st_size, "mtime_ns_before": before.st_mtime_ns, "mtime_ns_after": after.st_mtime_ns}

def nodes(value, path="outcome"):
    result = {}
    if isinstance(value, dict):
        if "reasoning" in value or "reasoning_content" in value or ("content" in value and "completion_tokens" in value):
            reason = value.get("reasoning", value.get("reasoning_content"))
            content = value.get("content")
            first = content.lstrip().splitlines()[0] if isinstance(content, str) and content.strip() else None
            result[path] = {"reasoning_characters": len(reason) if isinstance(reason, str) else None, "reasoning_utf8_bytes": len(reason.encode()) if isinstance(reason, str) else None, "content_characters": len(content) if isinstance(content, str) else None, "finish_reason": value.get("finish_reason"), "completion_tokens": value.get("completion_tokens"), "code_first_appearance": bool(CODE_FIRST.match(first)) if first is not None else None}
        for key, child in value.items():
            result.update(nodes(child, path + "." + key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            result.update(nodes(child, path + "[" + str(i) + "]"))
    for key, row in result.items():
        row["role"] = "aggregate_with_descendant_nodes" if any(other.startswith(key + ".") or other.startswith(key + "[") for other in result if other != key) else "leaf"
    return result

def unique(rows):
    out = {r["id"]: r for r in rows}
    assert len(out) == len(rows), "duplicate item responses"
    return out

def stats(rows):
    result = {"joined_leaf_nodes": len(rows), "joined_item_count": len({r["id"] for r in rows})}
    for field in FIELDS:
        pairs = [(r["base"][field], r["candidate"][field]) for r in rows if isinstance(r["base"][field], (float, int)) and isinstance(r["candidate"][field], (float, int))]
        result[field] = {"observed_pairs": len(pairs), "unobserved_pairs": len(rows) - len(pairs), "base_sum": sum(a for a, b in pairs) if pairs else None, "candidate_sum": sum(b for a, b in pairs) if pairs else None, "base_median": statistics.median(a for a, b in pairs) if pairs else None, "candidate_median": statistics.median(b for a, b in pairs) if pairs else None, "candidate_less": sum(b < a for a, b in pairs), "equal": sum(b == a for a, b in pairs), "candidate_more": sum(b > a for a, b in pairs)}
    for field in ("finish_reason", "code_first_appearance"):
        result[field] = {"base": dict(collections.Counter(str(r["base"][field]) for r in rows)), "candidate": dict(collections.Counter(str(r["candidate"][field]) for r in rows)), "transitions": dict(collections.Counter(str(r["base"][field]) + " -> " + str(r["candidate"][field]) for r in rows))}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("partial", "completed"), required=True)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py.txt").write_bytes(Path(__file__).read_bytes())
    for name in ("base", "candidate"):
        (args.output / name).mkdir()
    # Candidate prefix is frozen first. Later output is deliberately excluded.
    candidate_rows, candidate_pin = take(CAND / "responses.jsonl", args.output / "candidate")
    candidate_run, run_pin = take(CAND / "run.json", args.output / "candidate")
    base_rows, base_pin = take(BASE / "responses.jsonl", args.output / "base")
    item_rows, item_pin = take(BASE / "items.jsonl", args.output / "base")
    items, base, candidate = unique(item_rows), unique(base_rows), unique(candidate_rows)
    assert len(items) == 2338 and sum(r.get("area") == "code" for r in items.values()) == 90
    assert set(candidate) <= set(items) and set(candidate) <= set(base)
    if args.phase == "completed":
        assert candidate_run["status"] == "completed" and set(candidate) == set(items) and candidate_pin["unparsed_tail_bytes"] == 0
    pairs, coverage = [], []
    for item_id, candidate_row in candidate.items():
        a, b = nodes(base[item_id].get("outcome")), nodes(candidate_row.get("outcome"))
        item = items[item_id]
        coverage.append({"id": item_id, "area": item.get("area"), "source": item.get("source"), "base_error": base[item_id].get("error"), "candidate_error": candidate_row.get("error"), "base_node_count": len(a), "candidate_node_count": len(b), "matched_paths": sorted(a.keys() & b.keys()), "base_only_paths": sorted(a.keys() - b.keys()), "candidate_only_paths": sorted(b.keys() - a.keys())})
        for path in sorted(a.keys() & b.keys()):
            pairs.append({"id": item_id, "area": item.get("area"), "source": item.get("source"), "path": path, "base": a[path], "candidate": b[path]})
    leaves = [r for r in pairs if r["base"]["role"] == r["candidate"]["role"] == "leaf"]
    summary = {"phase": args.phase, "started_epoch": started, "seconds": time.time() - started, "candidate_run_status_at_snapshot": candidate_run.get("status"), "full_selection": 2338, "full_code_selection": 90, "response_records_observed": len(candidate), "code_response_records_observed": sum(items[i].get("area") == "code" for i in candidate), "candidate_errors_observed": sum(r.get("error") is not None for r in candidate.values()), "all_matched_nodes": len(pairs), "aggregate_nodes_excluded_from_stats": len(pairs) - len(leaves), "all": stats(leaves), "code": stats([r for r in leaves if r["area"] == "code"]), "noncode": stats([r for r in leaves if r["area"] != "code"]), "input_pins": {"base_responses": base_pin, "base_items": item_pin, "candidate_responses": candidate_pin, "candidate_run": run_pin}, "code_first_regex": CODE_FIRST.pattern, "limitations": ["Partial snapshots have response-completion-order bias; counts apply only to observed matching nodes, not all 2338 or all 90 items.", "Exact item/sample/path match is structural, not proof of identical input prefix or equivalent agent trajectory.", "Aggregate nodes with descendant nodes are retained but excluded from additive summaries.", "Reasoning text measures returned output, not internal computation or competence. Separate reasoning token counts are unobserved.", "Total completion tokens cannot isolate reasoning tokens. Unknown/null values are not filled with zero.", "Code-first is a narrow code-fence/Python-definition/import/shebang appearance heuristic; no match does not prove prose and a match does not prove incorrectness.", "No scoring, model/API request, benchmark rerun, adoption or interruption is performed."], "new_model_calls": 0, "rescoring_calls": 0}
    for filename, value in (("summary.json", summary), ("all-observed-item-coverage.json", coverage), ("all-matched-node-pairs.json", pairs)):
        (args.output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("phase", "seconds", "candidate_run_status_at_snapshot", "response_records_observed", "code_response_records_observed", "candidate_errors_observed", "all_matched_nodes", "aggregate_nodes_excluded_from_stats", "all", "code", "noncode")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
