#!/usr/bin/env python3
"""One-time CPU preparation of a pinned F04 MBPP LoRA job; never starts Docker."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
N = R / "analysis/fresh-purpose-20261107-f04-mbpp-lora-training-v1"
PREPARATION = R / "analysis/fresh-purpose-20261107-train-lora-owner-preparation-v1"
PY = R / "infra/weight-tools-ci/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-sha256", required=True)
    args = parser.parse_args()
    owner = B / "docs/experiments/scripts/fresh-av-native-gpu4-20261107.py"
    assert sha(owner) == "3e7e9c58b68a679d87623e1507e6508adcc731c3da36d51dd390f9bffcdb2b07"
    recipe = B / "scripts/experiments/gemma4-purpose"
    assert sha(recipe / "train_lora.py") == args.trainer_sha256
    assert not N.exists()
    N.mkdir()
    shutil.copyfile(__file__, N / "prepare-source.py.txt")
    source = N / "source"
    for original in (B / "src").rglob("*.py"):
        destination = source / original.relative_to(B)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    for name in ("native.py", "recipe_common.py", "objective_variants.py", "train_lora.py"):
        destination = source / "scripts/experiments/gemma4-purpose" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recipe / name, destination)
    source_pins = {str(p): sha(p) for p in source.rglob("*.py")}
    source_pins.update({str(p): sha(p) for p in (owner, owner.with_name("fresh-f01-finite.py"))})
    save(N / "source-pins.json", source_pins)
    F = R / "analysis/fresh-purpose-20261107-mbpp-train-preparation-v1/filter-v1"
    Q = R / "analysis/fresh-purpose-20261107-mbpp-reference-quality-v1"
    frames = R / "analysis/fresh-purpose-20261107-mbpp255-trainer-frames-v1/trainer-train204-frame-proof.json"
    split = read(F / "summary.json")
    quality = read(Q / "summary.json")
    assert quality["status"] == "completed" and quality["container_returncode"] == 0
    assert quality["input_unchanged"] and quality["protected27_unchanged"] and quality["owned_cleanup_completed"]
    assert quality["splits"]["train"]["eligible_ids"] == split["training_ids"]
    assert quality["splits"]["train"]["passed"] == split["train_rows"] == 204
    assert quality["splits"]["train"]["unknown"] == 0
    identity = dict(status="completed", training_data_sha256=sha(F / "train.jsonl"),
                    holdout_data_sha256=sha(F / "holdout.jsonl"), train_ids=split["training_ids"],
                    passed_train_ids=quality["splits"]["train"]["eligible_ids"], gradient_holdout_ids=[],
                    reference_quality_summary=pin(Q / "summary.json"),
                    all255_reference_outcomes=pin(Q / "all255-quality.json"),
                    split_summary=pin(F / "summary.json"), policy="all 204 TRAIN references pass; all 51 HOLDOUT stay outside gradients",
                    no_new_quality_execution=True)
    save(N / "input-quality-identity.json", identity)
    attestation = pin(R / "analysis/fresh-purpose-20261107-f04-dev30-dp4-v1/http-trace/capture.json")
    checkpoint = read(attestation["path"])["checkpoint"]
    task = dict(schema_version=1, kind="gemma4_lora_sft_v1",
                checkpoint=dict(path=checkpoint["path"], tree_sha256=checkpoint["sha256"], attestation=attestation),
                data=dict(**pin(F / "train.jsonl"), split="train", ids=split["training_ids"], split_summary=pin(F / "summary.json")),
                input_quality=pin(Q / "summary.json"), frame_evidence=pin(frames),
                hyperparameters=dict(rank=8, alpha=8, learning_rate=.0001, weight_decay=0.,
                                     grad_clip=1., accumulation=4, seed=0, max_frame_tokens=1024,
                                     relative_delta_cap=.005, pilot_steps=4, total_steps=64),
                maximum_seconds=1800, continuation_wait_seconds=300)
    save(N / "trainer-plan.json", task)
    snapshot = read(PREPARATION / "safe27-snapshot.json")
    assert snapshot["container_count"] == 27 and snapshot["temporary_sandbox_ids"] == []
    plan = dict(schema_version=1, mode="train-lora", approved_for_execution=False,
                checkpoint=checkpoint["path"], checkpoint_tree_sha256=checkpoint["sha256"],
                attestation=attestation, source_root=str(source), source_pins=source_pins,
                output_root=str(N), container_prefix="fresh-f04-mbpp-lora-v1",
                maximum_training_seconds=2400, training_startup_allowance_seconds=300,
                target=snapshot["target"], protected=snapshot["protected"],
                training=dict(plan=pin(N / "trainer-plan.json"),
                              input_quality=pin(N / "input-quality-identity.json"),
                              holdout=pin(F / "holdout.jsonl"),
                              restoration_proof=pin(R / "analysis/fresh-purpose-20261107-f05-rejected-f04-restore-v1/restoration-completion.json")),
                authorization="CPU metadata preview only; root must approve exact separate plan before --run",
                continuation="Only root may write work/train-lora/continue.json binding pilot-ready SHA; otherwise pilot_only",
                snapshot_evidence=pin(PREPARATION / "safe27-snapshot.json"))
    arguments = [str(source / "scripts/experiments/gemma4-purpose/train_lora.py"), "--plan", str(N / "trainer-plan.json"),
                 "--output", str(N / "work/train-lora")]
    start = time.time()
    result = subprocess.run([str(PY), "-B", *arguments], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(source / "src"), TMPDIR=str(R / "tmp")))
    (N / "trainer-preview.stdout").write_text(result.stdout)
    (N / "trainer-preview.stderr").write_text(result.stderr)
    preview = dict(command=[str(PY), "-B", *arguments], trainer_arguments=arguments,
                   returncode=result.returncode, elapsed_seconds=time.time() - start,
                   model_calls=0, service_changes=0, source_pins=source_pins)
    save(N / "trainer-preview.json", preview)
    assert result.returncode == 0, result.stderr
    plan["training"]["cpu_preview"] = pin(N / "trainer-preview.json")
    save(N / "owner-plan.json", plan)
    start = time.time()
    command = [str(PY), "-B", str(owner), "--plan", str(N / "owner-plan.json")]
    result = subprocess.run(command, capture_output=True, text=True)
    (N / "owner-preview.stdout").write_text(result.stdout)
    (N / "owner-preview.stderr").write_text(result.stderr)
    save(N / "owner-preview.json", dict(command=command, returncode=result.returncode,
         elapsed_seconds=time.time()-start, model_calls=0, service_changes=0,
         plan=pin(N / "owner-plan.json"), source_unchanged=all(sha(p)==h for p,h in source_pins.items())))
    assert result.returncode == 0, result.stderr
    print(json.dumps(dict(plan=pin(N / "owner-plan.json"), trainer_plan=pin(N / "trainer-plan.json"),
                          preview=pin(N / "owner-preview.json"), mode="preview_no_execution")))


if __name__ == "__main__":
    main()
