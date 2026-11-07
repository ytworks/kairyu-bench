"""CPU tokenizer-only audit of unchanged public MBPP train/holdout assistant frames."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

started = time.monotonic()
assert not list(Path("/dev").glob("nvidia*"))
tokenizer = AutoTokenizer.from_pretrained("/checkpoint", local_files_only=True, trust_remote_code=False)
rows, input_sha = [], {}
for split in ("train", "holdout"):
    path = Path("/inputs") / (split + ".jsonl")
    raw = path.read_bytes()
    input_sha[split] = hashlib.sha256(raw).hexdigest()
    cases = [json.loads(line) for line in raw.splitlines()]
    assert len(cases) == {"train": 204, "holdout": 51}[split]
    for case in cases:
        user = {"role": "user", "content": case["text"]}
        assistant = {"role": "assistant", "content": case["code"]}
        prefix = tokenizer.apply_chat_template([user], tokenize=False, enable_thinking=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template([user, assistant], tokenize=False, enable_thinking=True, add_generation_prompt=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        exact = full_ids[:len(prefix_ids)] == prefix_ids
        suffix_ids = full_ids[len(prefix_ids):] if exact else None
        rows.append({
            "task_id": case["task_id"], "split": split,
            "raw_prompt_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
            "raw_code_sha256": hashlib.sha256(case["code"].encode()).hexdigest(),
            "prefix_text": prefix, "full_text": full,
            "prefix_ids": prefix_ids, "full_ids": full_ids, "assistant_target_ids": suffix_ids,
            "prefix_tokens": len(prefix_ids), "full_tokens": len(full_ids),
            "target_tokens": len(suffix_ids) if suffix_ids is not None else None,
            "exact_token_prefix": exact, "exact_text_prefix": full.startswith(prefix),
            "contains_artificial_empty_thought": "<|channel>thought\n<channel|>" in prefix,
            "last_16_token_ids": full_ids[-16:],
            "last_16_decoded": tokenizer.decode(full_ids[-16:], skip_special_tokens=False),
            "last_token_id": full_ids[-1], "eos_token_id_occurrences": full_ids.count(tokenizer.eos_token_id),
            "over_limit": len(full_ids) > 1024,
        })
out = Path("/out")
with (out / "all255-frames.json").open("x") as stream:
    json.dump(rows, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write("\n")
summary = {
    "rows": len(rows), "input_sha256": input_sha,
    "all_exact_prefix": all(x["exact_token_prefix"] and x["exact_text_prefix"] for x in rows),
    "prefix_mismatch_ids": [x["task_id"] for x in rows if not x["exact_token_prefix"]],
    "maximum_full_tokens": max(x["full_tokens"] for x in rows),
    "maximum_target_tokens": max(x["target_tokens"] or 0 for x in rows),
    "over_limit_ids": [x["task_id"] for x in rows if x["over_limit"]],
    "artificial_empty_thought_ids": [x["task_id"] for x in rows if x["contains_artificial_empty_thought"]],
    "max_tokens": 1024, "truncated": False, "enable_thinking": True,
    "add_special_tokens": False, "assistant_code_transformation": None,
    "loss_scope": "All official assistant suffix tokens after exact user generation prefix; no tests, artificial thought, or manually appended EOS.",
    "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
    "distinct_last_token_ids": sorted({x["last_token_id"] for x in rows}),
    "last_16_example": rows[0]["last_16_decoded"],
    "versions": {x: version(x) for x in ("transformers", "tokenizers")},
    "tokenizer_files_sha256": {x: hashlib.sha256((Path("/checkpoint") / x).read_bytes()).hexdigest() for x in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")},
    "all_frames_sha256": hashlib.sha256((out / "all255-frames.json").read_bytes()).hexdigest(),
    "model_imports": 0, "model_forwards": 0, "gradient_calls": 0,
    "elapsed_seconds": time.monotonic() - started,
}
with (out / "summary.json").open("x") as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write("\n")
print(json.dumps(summary))
