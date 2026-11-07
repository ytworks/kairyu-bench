"""Bind completed dense F06 evidence to existing generic deployment previews; no apply."""
from pathlib import Path
import ast
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
D = R / "analysis/fresh-purpose-20261107-f06-lora-deployment-plan-v1"
V = D / "bound-preview-v1"
M = R / "analysis/fresh-purpose-20261107-f04-mbpp-lora-training-v1/materialization-f06-v2"
H = B / "docs/experiments/scripts/redeploy-replica.py"
sys.path.insert(0, str(B / "src"))
module = importlib.util.spec_from_file_location("g", H)
g = importlib.util.module_from_spec(module)
module.loader.exec_module(g)
old_source = R / "analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1/remove-completed-observer-v2.py"
assert g.sha(old_source) == "99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce"
tree = ast.parse(old_source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
    and n.name in {"safe", "inventory"}], type_ignores=[]), str(old_source), "exec"), globals())


def save(name, value):
    p = V / name
    assert not p.exists()
    g.save(p, value)


def pin(path):
    return dict(path=str(path), sha256=g.sha(path))


def main():
    plan = g.read(D / "plan-unbound.json")
    assert all(g.sha(p) == h for p, h in plan["source_pins"].items())
    assert g.sha(M / "fullcandidateidentity.json") == "50aa8056b9e48257b48c8621396030b4796c2e72d1c94fd8f64ad84a6867bdf7"
    identity = g.read(M / "fullcandidateidentity.json")
    for key in ("manifest", "verify", "completion"):
        p = identity[key]
        assert g.sha(p["path"]) == p["sha256"]
    manifest = g.read(identity["manifest"]["path"])
    verified = g.read(identity["verify"]["path"])
    assert manifest["kind"] == verified["kind"] == "dense_lora_recipe_v1"
    assert verified["verified"] and manifest["candidate_tree_sha256"] == verified["candidate_tree_sha256"]
    candidate = identity["candidate"]
    assert candidate["path"] == plan["candidate"] and candidate["files"] == manifest["candidate_files"]
    assert candidate["tree_sha256"] == manifest["candidate_tree_sha256"] == "0fc0ef29346c4d8a06bea8570dee96d8e0ba7038fee0d5aded8db57852474ec5"
    assert len(candidate["files"]) == len(candidate["inventory_stat"]) == 9
    stats = {name: dict(device=row["dev"], **{k: v for k, v in row.items() if k != "dev"})
             for name, row in candidate["inventory_stat"].items()}
    assert g.inventory(Path(plan["candidate"])) == stats
    assert g.digest(sorted(candidate["files"], key=lambda r: r["path"])) == candidate["tree_sha256"]
    review = R / "analysis/fresh-purpose-20261107-f06-lora-review-v1"
    decision = g.read(review / "root-review-decision.json")
    receipt = g.read(review / "actual-opus55-v1/receipt.json")
    assert g.sha(review / "root-review-decision.json") == "0bb1c680e96f368306c91a371ec5a51b5819966cddc77f438ed5e3b946063e82"
    assert receipt["verified_requested_model"] and receipt["assistant_models"] == ["claude-opus-5-5"]
    assert decision["status"] == "proceed_to_one_full_dev30_after_verified_artifact_and_serving_identity"
    started = time.time()
    with (R / "candidate-evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        V.mkdir()
        before, raw = inventory()
        assert len(before) == 27
        targets = {}
        for service in plan["service_order"]:
            matches = [cid for cid, d in raw.items() if d["Config"].get("Labels", {}).get("com.docker.compose.project") == "gemma-4-31b-8gpu"
                       and d["Config"].get("Labels", {}).get("com.docker.compose.service") == service]
            assert len(matches) == 1
            cid = matches[0]
            assert cid == plan["expected_previous_cids"][service]
            old = g.snapshot(raw[cid])
            assert old["tree"] == identity["source"]["tree_sha256"] and old["checkpoint_path"] == identity["source"]["path"]
            assert old["running"] and old["health"] == "healthy"
            g.idle(raw[cid])
            targets[service] = cid
        protected = sorted(set(before) - set(targets.values()))
        assert len(protected) == 23
        save("protected-before.json", before)
        save("target-started-at.json", {s: raw[cid]["State"]["StartedAt"] for s, cid in targets.items()})
        freeze = dict(checkpoint_path=plan["candidate"], all_checkpoint_files=candidate["files"],
                      candidate_inventory_stat=stats, control_tree_sha256=candidate["tree_sha256"],
                      full_identity=pin(M / "fullcandidateidentity.json"), actual_manifest=identity["manifest"],
                      actual_verify=identity["verify"], stat_evidence=identity["stat_evidence"],
                      no_additional_payload_hash=True)
        save("f06-freeze.json", freeze)
        save("generic-verification.json", dict(verified=True, tree_sha256=candidate["tree_sha256"],
             kind="verified_dense_lora_recipe_adapter", actual_verify=identity["verify"],
             manifest=identity["manifest"], materialization_completion=identity["completion"]))
        previews = {}
        for service in plan["service_order"]:
            command = [plan["python"], str(H), "--control", plan["control"], "--service", service,
                       "--candidate", plan["candidate"], "--freeze", str(V / "f06-freeze.json"),
                       "--baseline", str(V / "all-chat-before.json"), "--output", str(V / (service + "-preview")), "--timeout", "1200"]
            t = time.time()
            with (V / (service + ".stdout")).open("x") as out, (V / (service + ".stderr")).open("x") as err:
                result = subprocess.run(command, cwd=B, env=dict(os.environ, PYTHONPATH=str(B / "src"),
                    PYTHONDONTWRITEBYTECODE="1"), stdout=out, stderr=err, timeout=90)
            save(service + "-command.json", dict(argv=command, returncode=result.returncode, seconds=time.time()-t))
            assert result.returncode == 0
            p = V / (service + "-preview/receipt.json")
            value = g.read(p)
            assert value["status"] == "preview" and not value["apply_requested"]
            assert value["before"]["id"] == targets[service]
            assert value["candidate_identity"]["tree_sha256"] == candidate["tree_sha256"]
            previews[service] = pin(p)
        assert inventory()[0] == before and g.inventory(Path(plan["candidate"])) == stats
        plan.update(status="all_four_generic_previews_passed_awaiting_root_apply",
                    freeze=pin(V / "f06-freeze.json"), verification=pin(V / "generic-verification.json"),
                    baseline=pin(V / "all-chat-before.json"), protected_before=pin(V / "protected-before.json"),
                    target_ids=targets, protected_ids=protected, previews=previews,
                    output=str(D / "parallel-apply-v1"), candidate_identity=pin(M / "fullcandidateidentity.json"),
                    materialization=identity["completion"], opus_receipt=pin(review / "actual-opus55-v1/receipt.json"),
                    root_review_decision=pin(review / "root-review-decision.json"))
        save("plan-bound-unapproved.json", plan)
        save("four-preview-summary.json", dict(status="all_four_previews_passed", seconds=time.time()-started,
             plan=pin(V / "plan-bound-unapproved.json"), freeze=pin(V / "f06-freeze.json"), previews=previews,
             target_ids=targets, all27_unchanged=True, protected_count=23, all9_stat_continuity=True,
             source=identity["source"]["path"], source_tree=identity["source"]["tree_sha256"],
             candidate_tree=candidate["tree_sha256"], stat_evidence=identity["stat_evidence"],
             model_calls=0, service_mutations=0, fresh_payload_hashes=0))
    save("preview-lock-released.json", dict(pid=os.getpid(), epoch=time.time(), normal_release=True))
    print(json.dumps(g.read(V / "four-preview-summary.json")))


if __name__ == "__main__":
    main()
