"""CPU tokenizer-only verification of all204 fixed body-target frames."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from transformers import AutoTokenizer

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def sha(value):
    return hashlib.sha256(value).hexdigest()

def save(name, value):
    with (Path("/out") / name).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write(chr(10))

started = time.monotonic()
assert not list(Path("/dev").glob("nvidia*"))
default = module("default_frames", "/source/default_prepare.py")
helper = module("body_frames", "/source/body_helper.py")
tokenizer = AutoTokenizer.from_pretrained("/checkpoint", local_files_only=True, trust_remote_code=False)
rows = [json.loads(line) for line in Path("/inputs/train.jsonl").read_text().splitlines()]
assert len(rows) == 204 and len({row["task_id"] for row in rows}) == 204
prior = json.loads(Path("/prior/all255-frames.json").read_text())[:204]
frames, proof = helper.prepare_frames(tokenizer, rows, default.prepare_training_frames)
save("body-frame-proof.json", proof)
records = []
for row, frame, evidence, original in zip(rows, frames, proof["rows"], prior, strict=True):
    assert frame["id"] == original["task_id"] == row["task_id"]
    exact_ids = frame["ids"] == original["full_ids"]
    exact_prefix = frame["prefix_tokens"] == original["prefix_tokens"]
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["text"]}, {"role": "assistant", "content": row["code"]}],
        tokenize=False, enable_thinking=True, add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["text"]}],
        tokenize=False, enable_thinking=True, add_generation_prompt=True)
    offsets = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    terminal = tokenizer.encode("<turn|>" + chr(10), add_special_tokens=False)
    positions = frame["body_target_positions"]
    start, end = evidence.get("body_character_start"), evidence.get("body_character_end")
    intervals_valid = bool(positions) and all(
        offsets[i][0] >= start and offsets[i][1] > offsets[i][0] and offsets[i][1] <= end
        for i in positions)
    prefix_terminal_excluded = bool(positions) and all(
        frame["prefix_tokens"] <= i < len(frame["ids"]) - len(terminal) for i in positions)
    tree = ast.parse(row["code"])
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    first = functions[0]
    following = tree.body[tree.body.index(first) + 1:]
    subsequent_offsets = []
    lines = row["code"].splitlines(keepends=True)
    for node in following:
        offset_bytes = sum(len(line.encode()) for line in lines[:node.lineno - 1]) + node.col_offset
        subsequent_offsets.append(len(prefix) + len(row["code"].encode()[:offset_bytes].decode()))
    following_target_tokens = sum(
        any(offsets[i][0] >= offset for offset in subsequent_offsets) for i in positions)
    record = {
        "id": frame["id"], "reference_code_sha256": sha(row["code"].encode()),
        "complete_tokens": len(frame["ids"]), "prefix_tokens": frame["prefix_tokens"],
        "body_target_tokens": len(positions), "body_positions_sha256": sha(json.dumps(positions).encode()),
        "full_ids_equal_prior": exact_ids, "prefix_equal_prior": exact_prefix,
        "all_target_intervals_inside_declared_body_to_end": intervals_valid,
        "prefix_and_official_terminator_excluded": prefix_terminal_excluded,
        "terminal_ids": terminal, "body_start_character": start, "body_end_character": end,
        "first_function": first.name, "following_top_level_types": [type(node).__name__ for node in following],
        "following_top_level_target_tokens": following_target_tokens,
        "first_function_docstring_present": ast.get_docstring(first) is not None,
        "eligible": evidence["eligible"], "unsupported": frame["id"] in proof["unsupported_ids"],
    }
    records.append(record)
save("all204-boundary-checks.json", records)
failed = [x["id"] for x in records if not all(x[k] for k in [
    "full_ids_equal_prior", "prefix_equal_prior", "all_target_intervals_inside_declared_body_to_end",
    "prefix_and_official_terminator_excluded", "eligible"]) or x["unsupported"]]
summary = {
    "status": "passed" if not failed else "failed",
    "rows": len(records), "failed_ids": failed, "unsupported_ids": proof["unsupported_ids"],
    "ineligible_ids": proof["ineligible_ids"], "complete_tokens_total": sum(x["complete_tokens"] for x in records),
    "complete_tokens_maximum": max(x["complete_tokens"] for x in records),
    "body_target_tokens_total": sum(x["body_target_tokens"] for x in records),
    "body_target_tokens_minimum": min(x["body_target_tokens"] for x in records),
    "body_target_tokens_maximum": max(x["body_target_tokens"] for x in records),
    "following_definition_ids": [x["id"] for x in records if x["following_top_level_types"]],
    "following_definition_target_tokens": {x["id"]: x["following_top_level_target_tokens"] for x in records if x["following_top_level_types"]},
    "scope": "First top-level function's first non-docstring statement through assistant-code end. Later function headers remain targets in nine rows; this is not the union of function-body-only AST spans.",
    "model_forwards": 0, "teacher_forwards": 0, "gradient_calls": 0, "generated_or_reference_code_execution": 0,
    "truncation": False, "holdout_tokenized_here": 0, "seconds": time.monotonic() - started,
}
save("summary.json", summary)
print(json.dumps(summary))
assert not failed and not proof["unsupported_ids"] and not proof["ineligible_ids"]
assert summary["complete_tokens_maximum"] <= 1024
