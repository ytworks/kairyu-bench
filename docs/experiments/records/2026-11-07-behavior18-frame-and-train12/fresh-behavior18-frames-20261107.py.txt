"""Fixed eighteen-case tokenizer-only length evidence; no model imports or forwards."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from transformers import AutoTokenizer
from kairyu_bench.wsp.weights.finite_replay import prepare_frames, validate_frame_rows

started = time.monotonic()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    with (Path("/out") / name).open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


inputs = {
    "train-cases.json": "b42990b0caac24ccc7843e40718c0c5e7ad61dd5baa131ed6852c126bf37f66d",
    "transfer-cases.json": "4951c70a686ebcd7135919c48d4038c6aa6d359abc018730aee73c1afe9f4b55",
}
for name, digest in inputs.items():
    assert sha(Path("/inputs") / name) == digest
train = json.loads(Path("/inputs/train-cases.json").read_text())
transfer = json.loads(Path("/inputs/transfer-cases.json").read_text())
assert transfer["split"] == "transfer" and transfer["gradient_allowed"] is False
assert list(train) == list("ABCDEF")
assert all(len(folds) == 2 and all(len(fold) == 1 for fold in folds) for folds in train.values())
tr = [r for folds in train.values() for fold in folds for r in fold]
te = transfer["cases"]
assert [r["id"] for r in tr] == [f"behavior-{a}{i}" for a in "ABCDEF" for i in (1, 2)]
assert [r["id"] for r in te] == [f"behavior-{a}3" for a in "ABCDEF"]
assert all(r["split"] == "train" for r in tr) and all(r["split"] == "transfer" for r in te)
assert len({r["id"] for r in tr + te}) == 18
assert not list(Path("/dev").glob("nvidia*"))
tokenizer = AutoTokenizer.from_pretrained("/checkpoint", local_files_only=True, trust_remote_code=False)
prepared, evidence = prepare_frames(
    tokenizer, {**train, "transfer_length_only": [[r] for r in te]}, max_frame_tokens=None
)
rows = [r for folds in prepared.values() for fold in folds for r in fold]
evidence["lengths"] = [
    {"id": r[0], "prefix": len(r[1]), "correct": len(r[2]), "foil": len(r[3]),
     "correct_complete": len(r[1]) + len(r[2]), "foil_complete": len(r[1]) + len(r[3])}
    for r in rows
]
evidence["over_limit_ids"] = [r[0] for r in rows if len(r[1]) + max(len(r[2]), len(r[3])) > 1024]
evidence.update(expected_count=18, model_forwards=0, generation_calls=0, gradient_calls=0,
                max_frame_tokens=1024, input_sha256=inputs,
                maximum=max(len(r[1]) + max(len(r[2]), len(r[3])) for r in rows),
                versions={name: version(name) for name in ("transformers", "tokenizers", "numpy", "torch")},
                elapsed_seconds=time.monotonic()-started,
                boundary_whitespace_policy="Existing shared frame strips prompt/correct/foil boundary whitespace; no token truncation")
save("frame-evidence.json", evidence)
# Exact subset of shared prepare_frames return, with no ID/fold/row relabelling.
train_evidence = {k: v for k, v in evidence.items() if k not in {
    "lengths", "over_limit_ids", "expected_count", "model_forwards", "generation_calls",
    "gradient_calls", "max_frame_tokens", "input_sha256", "maximum", "versions",
    "elapsed_seconds", "boundary_whitespace_policy"}}
train_evidence["frames"] = {group: evidence["frames"][group] for group in train}
save("train-frame-evidence.json", train_evidence)
save("train-frame-subset-proof.json", {
    "all18_evidence_sha256": sha("/out/frame-evidence.json"),
    "train_evidence_sha256": sha("/out/train-frame-evidence.json"),
    "train_cases_sha256": inputs["train-cases.json"],
    "ids": [r["id"] for r in tr], "fold_sizes": {g: [len(f) for f in train[g]] for g in train},
    "shared_frame_rows_equal": all(train_evidence["frames"][g] == evidence["frames"][g] for g in train),
    "original_train_rows_sha256": hashlib.sha256(json.dumps(train, sort_keys=True).encode()).hexdigest(),
    "transfer_removed_only_from_derived_train_view": True, "source_envelopes_unchanged": True})
for name, digest in inputs.items():
    assert sha(Path("/inputs") / name) == digest
validate_frame_rows(rows, 1024)
print(json.dumps({"cases": 18, "maximum": evidence["maximum"], "over_limit_ids": evidence["over_limit_ids"],
                  "model_forwards": 0, "elapsed_seconds": evidence["elapsed_seconds"]}))
