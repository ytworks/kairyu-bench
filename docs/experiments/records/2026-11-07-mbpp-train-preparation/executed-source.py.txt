"""Fixed MBPP original TRAIN filtering recipe. CPU only; never executes examples."""
import argparse
import ast
import hashlib
import io
import json
import math
import re
import time
import tokenize
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_stat(path):
    s = path.stat()
    return {k: getattr(s, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


class StripDocs(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
        return node


def features(prompt, code):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", prompt).casefold())
    tree = None
    try:
        tree = StripDocs().visit(ast.parse(code))
        canonical = ast.dump(tree, include_attributes=False)
        normalized_code = ast.unparse(tree)
    except (SyntaxError, ValueError, TypeError):
        canonical = None
        normalized_code = code
    try:
        tokens = [t.string for t in tokenize.generate_tokens(io.StringIO(normalized_code).readline)
                  if t.type not in {tokenize.ENCODING, tokenize.COMMENT, tokenize.INDENT,
                                    tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = []
    return {"prompt": " ".join(words), "ast": canonical,
            "prompt_tokens": len(words), "code_tokens": len(tokens),
            "prompt_shingles": {tuple(words[i:i+3]) for i in range(len(words)-2)},
            "code_shingles": {tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}}


def matches(left, right, policy):
    found = []
    if left["prompt"] and left["prompt"] == right["prompt"]:
        found.append({"rule": "exact_prompt"})
    if left["ast"] and left["ast"] == right["ast"]:
        found.append({"rule": "exact_code_ast"})
    for kind in ["prompt", "code"]:
        rule = policy["near_" + kind]
        if min(left[kind+"_tokens"], right[kind+"_tokens"]) < rule["minimum_shorter_tokens"]:
            continue
        a, b = left[kind+"_shingles"], right[kind+"_shingles"]
        if not a or not b:
            continue
        common = len(a & b)
        jaccard = common / len(a | b)
        containment = common / min(len(a), len(b))
        if jaccard >= rule["jaccard_gte"] or containment >= rule["containment_of_shorter_gte"]:
            found.append({"rule": "near_"+kind, "jaccard": jaccard, "shorter_containment": containment})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    policy_path = args.input_root / "filter-policy.json"
    policy = json.loads(policy_path.read_text())
    source_path = args.input_root / "upstream-mbpp.jsonl"
    download = json.loads((args.input_root / "dataset-download-receipt.json").read_text())
    assert sha(source_path) == download["sha256"] and sha(policy_path) == download["filter_policy_frozen_before_download_sha256"]
    assert policy["original_train_task_ids_inclusive"] == [601, 974]
    pins = {str(p): sha(p) for p in [source_path, policy_path, args.manifest, Path(__file__)]}
    raw_rows = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    by_id = {row["task_id"]: row for row in raw_rows}
    assert len(by_id) == len(raw_rows) == 974 and set(by_id) == set(range(1, 975))
    manifest = json.loads(args.manifest.read_text())["items"]
    official = {int(x["id"].split("/")[-1]) for x in manifest if x["source"] == "mbpp-plus"}
    probes = {int(x["id"].split(":")[-1]) for x in manifest if x["source"] == "probe-code"}
    references, reference_meta, pool_ids = [], [], set()

    def add(source, identifier, prompt, code):
        item = dict(source=source, id=str(identifier), **features(prompt, code))
        references.append(item)

    for row in raw_rows:
        if row["task_id"] < 601:
            add("original_mbpp_nontrain", row["task_id"], row["text"], row["code"])
    for dataset, filename in [("mbppplus", "MbppPlus-v0.2.0.jsonl"), ("humanevalplus", "HumanEvalPlus-v0.1.10.jsonl")]:
        path = args.cache / "xdg/evalplus" / filename
        pins[str(path)] = sha(path)
        count = 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                code = row["prompt"] + row["canonical_solution"]
                try:
                    tree = ast.parse(code)
                    prompt = ast.get_docstring(tree) or next((ast.get_docstring(n) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(n)), row["prompt"])
                except SyntaxError:
                    code, prompt = row["canonical_solution"], row["prompt"]
                add(dataset, row["task_id"], prompt, code)
                if dataset == "mbppplus":
                    pool_ids.add(int(row["task_id"].split("/")[-1]))
                count += 1
        reference_meta.append({"source": dataset, "rows": count, "full_file_sha256": pins[str(path)]})
    assert len(pool_ids) == 378 and len(official | probes) == 228
    arrow_pins = {}
    for category in ["SciCode1___sci_code", "json"]:
        root = args.cache / "hf/wsp/datasets" / category
        for path in sorted(root.rglob("*.arrow")):
            before = stable_stat(path)
            projected = []
            with pa.memory_map(str(path), "r") as mapped:
                for batch in ipc.open_stream(mapped):
                    columns = ["problem_id", "problem_description_main", "sub_steps", "general_solution"] if category != "json" else ["question_id", "question_content"]
                    projected.extend(batch.select(columns).to_pylist())
            for row in projected:
                if category == "json":
                    add("livecodebench_prompt_only", row["question_id"], row["question_content"], "")
                else:
                    add("scicode_"+path.stem, row["problem_id"], row["problem_description_main"], row["general_solution"])
                    for step in row["sub_steps"]:
                        add("scicode_"+path.stem, row["problem_id"]+":"+step["step_number"], step["step_description_prompt"], step["ground_truth_code"])
            assert stable_stat(path) == before
            encoded = json.dumps(projected, sort_keys=True, ensure_ascii=False).encode()
            arrow_pins[str(path)] = {"stat": before, "projected_columns": columns, "projected_rows": len(projected), "projected_bytes": len(encoded), "projected_sha256": hashlib.sha256(encoded).hexdigest(), "full_file_hash": None}
    retained, ledger = [], []
    for identifier in range(601, 975):
        row = by_id[identifier]
        reasons = []
        if identifier in official:
            reasons.append({"rule": "official_WSP_code_id"})
        if identifier in probes:
            reasons.append({"rule": "official_WSP_probe_id"})
        if identifier in pool_ids:
            reasons.append({"rule": "complete_MBPPplus_pool_id"})
        if not row.get("text", "").strip() or not row.get("code", "").strip() or len(row.get("test_list", [])) != 3 or not all(isinstance(t, str) and t.strip() for t in row.get("test_list", [])):
            reasons.append({"rule": "missing_contract_fields_or_not_three_tests"})
        for label, code in [("code", row.get("code", "")), ("setup", row.get("test_setup_code", ""))] + [("test_"+str(i), t) for i, t in enumerate(row.get("test_list", []))]:
            try:
                ast.parse(code)
            except (SyntaxError, TypeError, ValueError) as error:
                reasons.append({"rule": "syntax_invalid", "field": label, "error_type": type(error).__name__})
        feature = features(row["text"], row["code"])
        for ref in references:
            for match in matches(feature, ref, policy):
                reasons.append(dict(match, reference_source=ref["source"], reference_id=ref["id"]))
        for previous in retained:
            for match in matches(feature, previous["features"], policy):
                reasons.append(dict(match, reference_source="earlier_retained_train", reference_id=previous["row"]["task_id"]))
        status = "excluded" if reasons else "retained_before_holdout"
        ledger.append({"task_id": identifier, "original_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "status": status, "reasons": reasons})
        if not reasons:
            retained.append({"row": row, "features": feature})
    seed = policy["holdout"]["seed"]
    ordered = sorted((x["row"]["task_id"] for x in retained), key=lambda i: hashlib.sha256((seed+":"+str(i)).encode()).hexdigest())
    holdout = set(ordered[:min(64, math.ceil(len(ordered)*.2))])
    training = set(ordered) - holdout
    for row in ledger:
        if row["status"] != "excluded":
            row["status"] = "holdout" if row["task_id"] in holdout else "train"
    for name, selected in [("train", training), ("holdout", holdout)]:
        (args.output/(name+".jsonl")).write_text("".join(json.dumps(by_id[i], ensure_ascii=False)+"\n" for i in sorted(selected)))
    summary = {"status": "completed", "upstream_rows": 974, "original_train_rows": 374, "official_id_hits": len(set(range(601, 975)) & (official|probes)), "whole_pool_id_hits": len(set(range(601, 975)) & pool_ids), "excluded_rows": sum(x["status"]=="excluded" for x in ledger), "retained_unique_rows": len(retained), "train_rows": len(training), "holdout_rows": len(holdout), "training_ids": sorted(training), "holdout_ids": sorted(holdout), "exclusion_rule_rows": dict(Counter(rule for x in ledger for rule in {r["rule"] for r in x["reasons"]})), "references_by_source": dict(Counter(x["source"] for x in references)), "reference_metadata": reference_meta, "arrow_projection_evidence": arrow_pins, "input_sha256": pins, "seconds": time.time()-started, "model_calls": 0, "training_steps": 0, "reference_execution_calls": 0, "quality_status": "syntax and contamination-filter eligible; reference execution and semantic validity untested", "semantically_disjoint_proven": False}
    assert not training & holdout and not (training|holdout) & (official|probes|pool_ids)
    assert all(sha(Path(p)) == v for p, v in pins.items())
    assert all(stable_stat(Path(p)) == v["stat"] for p, v in arrow_pins.items())
    (args.output/"all374-ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False)+"\n")
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"training_ids","holdout_ids","input_sha256","arrow_projection_evidence","reference_metadata"}}))


if __name__ == "__main__":
    main()
