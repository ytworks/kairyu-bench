"""One-off saved-record taxonomy; no model requests, scoring, or generated-code exec."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
RUN = R / "results/gemma4-fresh-f04-dev30-dp4-20261107-v1"
N = R / "analysis/fresh-purpose-20261107-f04-failure-taxonomy-v1"
ARCHIVE = R / "runtime/fresh-purpose-20261107-av-live10-v1/runner-app"
WSP = Path("/mnt/nvme/kairyu/dev-logs/wsp")
RULES = {
    "precedence": ["length", "explicit_timeout", "missing_required_entrypoint",
                   "syntax_or_extraction", "assertion_stderr_hint", "unknown"],
    "length": "Only saved finish_reason=length; token count alone is insufficient.",
    "explicit_timeout": "Only saved timed_out=true or explicit timeout status.",
    "missing_required_entrypoint": "Expected callable absent after pinned sanitizer; AST only.",
    "syntax_or_extraction": "Empty selected code or AST SyntaxError; no generated-code execution.",
    "assertion_stderr_hint": "SciCode failed, returncode nonzero, timed_out=false, AssertionError hint. Medium confidence; not raw traceback proof.",
    "unknown": "Binary rejection, generic fail, or non-assertion exception hint cannot identify assertion vs runtime vs per-test timeout.",
    "not_inferred": ["No model-capability attribution from weights.", "No hidden-test inputs or reference answers read.",
                     "No benchmark-to-training-data transfer.", "No claimed timeout from long generation or algorithm complexity.",
                     "EvalPlus fail may include internal timeout; only separate timeout is unambiguous."],
}
NOTES = {
    "HumanEval/151": "Observed: type(num)==int excludes bool/subclasses. Plausible contract edge; actual failing input unavailable.",
    "Mbpp/801": "Observed: max frequency returns 1 when all three distinct. Sentinel semantics are a plausible functional issue; exact failing test unavailable.",
    "Mbpp/410": "Observed: numeric filtering followed by min without an empty guard. Edge contract/runtime cause remains unknown.",
    "Mbpp/576": "Observed: contiguous slicing, with empty sublist returning True. Exact rejected edge unavailable.",
    "Mbpp/129": "Observed: imposes a complete 1..n^2 set in addition to equal sums. Contract interpretation is a hypothesis.",
    "Mbpp/103": "Observed: explicit n=0/m<0/m>=n branches and quadratic DP. Boundary or resource cause not separable.",
    "Mbpp/639": "Observed: requires first character isupper, not merely non-lowercase. Non-letter edge is a hypothesis.",
    "Mbpp/126": "Expected entry point sum; saved public code defines sum_common_divisors. Pinned sanitizer retains only import math; no required callable.",
    "arc191_c": "Observed: bounded search for prime k*N+1 and A; unsuccessful search emits no line. Whether this triggered rejection is unknown.",
    "abc390_g": "Observed: nested coefficient convolution; recurrence changes. Complexity risk is not an observed timeout.",
    "arc196_a": "Observed: sorts odd/even position pools then pairs extremes. Feasibility reduction is an unverified algorithm hypothesis.",
    "arc190_c": "Observed: two markdown code blocks; official extractor selects the last, valid block. It recomputes full grid DP per query; no saved timeout evidence.",
    "abc392_d": "Observed: pairwise Counter overlap and 15-decimal output. Binary result gives no concrete cause.",
}

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def pin(p: Path) -> dict:
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}

def structure(code: str) -> dict:
    result = {"bytes": len(code.encode()), "sha256": sha(code.encode()), "empty": not code.strip()}
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        result.update(parse_ok=False, syntax_error={"type": type(error).__name__, "line": error.lineno})
        return result
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    result.update(parse_ok=True, top_level_definitions=names,
                  duplicate_top_level_definitions={k: v for k, v in Counter(names).items() if v > 1})
    return result

def main() -> None:
    started = time.time()
    N.mkdir(exist_ok=False)
    (N / "rules.json").write_text(json.dumps({"frozen_epoch": started, "rules": RULES}, indent=2) + "\n")
    from evalplus.sanitize import sanitize
    import evalplus.sanitize as sanitizer
    sys.path.insert(0, str(WSP / "src/livecodebench"))
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code
    import lcb_runner.utils.extraction_utils as extraction

    tasks_path = WSP / "cache/evalplus-tasks.json"
    tasks = json.loads(tasks_path.read_text())
    pins = [pin(Path(__file__)), pin(RUN / "items.jsonl"), pin(RUN / "report.json"),
            pin(tasks_path), pin(Path(sanitizer.__file__)), pin(Path(extraction.__file__))]
    for suffix in ["sources/code.py", "sources/__init__.py", "scorers/evalplus_score.py",
                   "scorers/livecodebench_score.py", "scorers/scicode_score.py"]:
        p = ARCHIVE / "src/kairyu_bench/wsp" / suffix
        q = B / "src/kairyu_bench/wsp" / suffix
        pins.append({**pin(p), "current_source_byte_equal": p.read_bytes() == q.read_bytes()})
    pins += [pin(R / "analysis/fresh-purpose-20261107-f04-dev30-dp4-v1/actual-runner.json"),
             pin(R / "analysis/fresh-purpose-20261107-f04-dev30-dp4-v1/launch-plan-approved.json")]
    rows, sci_all = [], []
    for lineno, raw in enumerate((RUN / "items.jsonl").read_bytes().splitlines(keepends=True), 1):
        item = json.loads(raw)
        if item["source"] not in {"humaneval-plus", "mbpp-plus", "livecodebench", "scicode"}:
            continue
        sample = item["outcome"]["samples"][0]
        common = {"item_id": item["id"], "source": item["source"], "official_score": item["score"],
                  "finish_reason": sample["finish_reason"], "completion_tokens": sample["completion_tokens"],
                  "api_error": item["error"], "line": lineno, "line_sha256": sha(raw),
                  "input_path": str(RUN / "items.jsonl")}
        if item["source"] == "scicode":
            files = sample["files"]
            assert len(files) == len(sample["calls"])
            calls = dict(zip(files, sample["calls"], strict=True))
            for diagnostic in item["detail"]["samples"][0]["step_diagnostics"]:
                step = diagnostic["step"]
                call = calls[step]
                record = {**common, "id": item["id"] + ":" + step,
                          "step": step, "diagnostic": diagnostic,
                          "step_finish_reason": call["finish_reason"],
                          "step_code": structure(files[step]),
                          "step_public_sha256": sha((call.get("content") or "").encode()),
                          "code_pointer": "outcome.samples[0].files." + step}
                sci_all.append(record)
                if diagnostic["passed"]:
                    continue
                if call["finish_reason"] == "length":
                    category, certainty = "length", "high"
                elif diagnostic["timed_out"]:
                    category, certainty = "timeout", "high"
                elif not record["step_code"]["parse_ok"]:
                    category, certainty = "format_extraction_syntax", "high_static"
                elif diagnostic["exception_class"] == "AssertionError":
                    category, certainty = "functional_assertion_hint", "medium_hint_only"
                else:
                    category, certainty = "unknown", "underdetermined"
                record.update(category=category, certainty=certainty,
                              causal_explanation="Unavailable: saved stderr contains only digest, size and allowlisted class hint; full stderr and appended tests were not retained.",
                              mechanism_note="Stored step includes dependencies and earlier generated helpers; failure may be in an earlier helper or current function. No blanket propagation claim.")
                rows.append(record)
            continue
        if item["score"] != 0:
            continue
        content = sample.get("content") or ""
        row = {**common, "id": item["id"], "detail": item["detail"],
               "public_content_sha256": sha(content.encode()),
               "public_content_bytes": len(content.encode()),
               "code_pointer": "outcome.samples[0].content"}
        expected = None
        if item["source"] == "livecodebench":
            code = extract_code(content, LMStyle.OpenAIChat)
            method = "cached official LiveCodeBench OpenAIChat extractor; last fenced block"
        else:
            dataset = "humaneval" if item["source"] == "humaneval-plus" else "mbpp"
            expected = tasks[dataset][item["id"].split(":", 1)[1]]["entry_point"]
            code = sanitize(content, expected)
            method = "cached evalplus sanitizer; pure string/tree parsing"
        parsed = structure(code)
        row.update(extracted_code=parsed, extraction_method=method, expected_entrypoint=expected)
        missing = expected and expected not in parsed.get("top_level_definitions", [])
        if sample["finish_reason"] == "length":
            category, certainty = "length", "high"
        elif any(v.get(k) == "timeout" for v in item["detail"]["samples"] for k in ("base", "plus")):
            category, certainty = "timeout", "high"
        elif missing:
            category, certainty = "format_extraction_syntax", "high_static"
        elif parsed["empty"] or not parsed["parse_ok"]:
            category, certainty = "format_extraction_syntax", "high_static"
        else:
            category, certainty = "unknown", "underdetermined"
        row.update(category=category, certainty=certainty,
                   missing_required_entrypoint=bool(missing),
                   observable_rejection="Saved official tests rejected the submitted code; no per-test failure detail or stderr retained.",
                   static_note=NOTES[item["id"].split(":", 1)[1]],
                   full_stderr_available=False, generated_code_executions=0)
        rows.append(row)
    assert len(rows) == 31 and len(sci_all) == 39
    assert Counter(r["source"] for r in rows) == Counter({"humaneval-plus": 1, "mbpp-plus": 7, "livecodebench": 5, "scicode": 18})
    report = {"status": "complete_saved_record_taxonomy", "created_epoch": time.time(),
              "scope": {"failed_standard_items": 13, "failed_scicode_steps": 18, "all_scicode_steps": 39},
              "rules": RULES, "counts": dict(Counter(r["category"] for r in rows)), "rows": rows,
              "all_scicode_steps": sci_all, "input_source_pins": pins,
              "method_limits": "AST/sanitizer/extractor parsing only. No scoring, code exec, model API, benchmark references, weight edit or training data creation. Cached extractor identity recorded; runner extraction payload was temporary and not independently retained.",
              "model_calls": 0, "scoring_calls": 0, "generated_code_executions": 0,
              "seconds": time.time() - started}
    (N / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"path": str(N / "report.json"), "sha256": sha((N / "report.json").read_bytes()),
                      "counts": report["counts"], "seconds": report["seconds"]}))
if __name__ == "__main__":
    main()
