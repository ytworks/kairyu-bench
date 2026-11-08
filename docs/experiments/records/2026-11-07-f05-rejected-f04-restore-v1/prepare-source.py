"""Restore rejected F05 deployment to best complete F04; existing previews only, no weight editing."""

import argparse
import ast
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
D = R / "analysis/fresh-purpose-20261107-f05-rejected-f04-restore-v1"
PR = B / "docs/experiments/records/2026-11-07-f05-rejected-f04-restore-v1"
CANDIDATE = R / "checkpoints/fresh-native-relative-f04-faststore-v1/checkpoint"
PARENT = R / "checkpoints/fresh-native-contract-f05-faststore-v1/checkpoint"
PARENT_MANIFEST = PARENT.parent / "manifest.json"
PARENT_TREE = "fcc4681343c1a0c338157bee767df2024f37290469e55dbc417723717f4ac176"
GATE = R / "analysis/fresh-purpose-20261107-f05-behavior19-native-v1/score-v1/gate.json"
DECISION = B / "docs/experiments/records/2026-11-07-f04-official-comparison/parent-selection.json"
TEMPLATE = R / "analysis/fresh-purpose-20261107-f05-native-contract-v1/deployment-dp4-v1/plan-root-approved.json"
H = B / "docs/experiments/scripts/redeploy-replica.py"
OWNER = B / "docs/experiments/scripts/fresh-four-parallel-redeploy-20261107.py"
INVENTORY_SOURCE = R / "analysis/fresh-purpose-20261107-av-dev30-deployment-plan-v1/remove-completed-observer-v2.py"
TARGETS = {
    "gemma-0": "19a1344a05b79dbc5ac3f5f1c2540126847586226a002e042714158fe408b7aa",
    "gemma-1": "c49e6eb95a1609b547e74fa32706ea36068f60bb95b5e23136b66a1e73bf6490",
    "gemma-2": "abb3b4d5103df28984f34e784c757fd26b093b48a426824d817b271c8cdbc1d0",
    "gemma-3": "9ef25f4abd6e5bf027a50f3f147636f5e15def3d4bc4666006396c3671e0030e",
}
STATIC_PINS = {
    str(GATE): "77548bc4f71fcb74afafc3e964fd8a39293255e649e02373001a5ffdec3c1d2f",
    str(H): "2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae",
    str(OWNER): "19ae025add4f2f08705f33f0146f48cc9fa8e92bdb14a8383f46fb7d531e53bc",
    str(PARENT_MANIFEST): "09ea569a08f138efd438055d7149f0697dc1d6edd08332f2b6c7c09b1da7c24c",
    str(DECISION): "50a48edce45c69e7f06e35566dcdbbaa93ed18d2c27368d2f14c3c52d759157b",
    str(TEMPLATE): "ec377967cf4e93015dacd8396c84fa0dcb17e87ed59cc6aadc3ce4e7b441b719",
    str(INVENTORY_SOURCE): "99a9cc25006a988d1d7cfe11b8f579ea0bbb857c4de34c39f21bf0769b5a12ce",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "spec", "verify"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    assert os.getuid() == 1003
    assert all(p.is_absolute() for p in (args.manifest, args.spec, args.verify))
    assert Path(__file__).resolve() == D / "prepare-source.py"
    assert not (D / "preview-start.json").exists(), "One invocation only; retain any prior failure."
    assert not (D / "preview-failure.json").exists()
    sys.path.insert(0, str(B / "src"))
    module = importlib.util.spec_from_file_location("guard", H)
    g = importlib.util.module_from_spec(module)
    module.loader.exec_module(g)

    def save(name, value):
        path = D / name
        assert not path.exists()
        g.save(path, value)

    started = time.time()
    stage = "input_validation"
    lock_held = False
    try:
        assert all(g.sha(p) == h for p, h in STATIC_PINS.items())
        proof, verification, spec = map(g.read, (args.manifest, args.verify, args.spec))
        assert proof["checkpoint_path"] == str(CANDIDATE)
        assert proof["spec_file"]["path"] == str(args.spec)
        assert proof["spec_file"]["sha256"] == g.sha(args.spec)
        assert proof["native_manifest"]["spec"] == spec
        assert verification["verified"] is True
        assert verification["tree_sha256"] == proof["control_tree_sha256"]
        assert verification["no_shared_link_mutation"] is True
        assert verification["private_original_byte_restoration"] is True
        assert verification["shared_payload_bytes_read"] == 0
        assert verification["fresh_full_model_hash"] is False
        assert len(proof["all_checkpoint_files"]) == 9
        for key in ("changed_elements", "changed_tensors", "delta_l2"):
            assert verification[key] == proof[key]
        parent = g.read(PARENT_MANIFEST)
        assert parent["checkpoint_path"] == str(PARENT)
        assert parent["control_tree_sha256"] == PARENT_TREE
        gate = g.read(GATE)
        assert gate["go"] is False
        assert gate["decision"] == "REJECT_AT_NATIVE_SCREEN_OFFICIALLY_UNSCORED"
        assert gate["candidate_tree_sha256"] == PARENT_TREE
        logical = {
            "kind": "deployment_only_restore_rejected_candidate",
            "current_checkpoint": str(PARENT),
            "current_tree": PARENT_TREE,
            "restore_checkpoint": str(CANDIDATE),
            "restore_tree": proof["control_tree_sha256"],
            "new_weight_edit": False,
            "rejection_gate": {"path": str(GATE), "sha256": g.sha(GATE)},
        }
        assert g.inventory(PARENT) == parent["candidate_inventory_stat"]
        assert g.inventory(CANDIDATE) == proof["candidate_inventory_stat"]
        shared = parent["candidate_inventory_stat"]["shared.safetensors"]
        assert shared["nlink"] == 17
        assert proof["candidate_inventory_stat"]["shared.safetensors"] == shared
        decision = g.read(DECISION)
        assert decision["selected_next_parent"] == "F04"
        assert decision["selected_tree_sha256"] == proof["control_tree_sha256"]
        plan = g.read(TEMPLATE)
        for key in ("authorization_epoch", "authorization_reason"):
            plan.pop(key, None)
        plan.update(
            approved_for_apply=False,
            parent_gate="Restore deployment from officially unscored rejected F05 to selected best-complete F04. No weight edit or new benchmark. Root approval of exact restore plan required.",
        )
        assert list(TARGETS) == plan["service_order"]
        assert plan["owner_sha256"] == g.sha(OWNER) and plan["helper_sha256"] == g.sha(H)
        assert all(g.sha(p) == h for p, h in plan["source_pins"].items())
        input_pins = {str(p): g.sha(p) for p in (args.manifest, args.spec, args.verify)}
        input_pins.update(STATIC_PINS)
        input_pins[str(Path(__file__))] = g.sha(Path(__file__))
        tree = ast.parse(INVENTORY_SOURCE.read_text())
        namespace = {"g": g, "json": json}
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"safe", "inventory"}]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(INVENTORY_SOURCE), "exec"), namespace)
        inventory = namespace["inventory"]
        stage = "global_lock"
        with (R / "candidate-evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_held = True
            stage = "live_metadata"
            before, raw = inventory()
            assert len(before) == 27 and set(TARGETS.values()) <= set(before)
            for cid in TARGETS.values():
                g.idle(raw[cid])
                snapshot = g.snapshot(raw[cid])
                assert snapshot["running"] and snapshot["health"] == "healthy"
                assert snapshot["checkpoint_path"] == str(PARENT)
                assert snapshot["tree"] == PARENT_TREE and snapshot["checkpoint_readonly"]
                assert snapshot["image_id"] == "sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe"
            protected = sorted(set(before) - set(TARGETS.values()))
            assert len(protected) == 23
            save("protected-before.json", before)
            save("preview-start.json", {
                "pid": os.getpid(), "epoch": started, "global_lock_held": True,
                "input_pins": input_pins, "argv": sys.argv,
                "source_pins": plan["source_pins"], "logical_parent": logical,
                "parent_shared_stat": shared, "model_calls": 0, "serving_mutations": 0,
            })
            previews = {}
            for service in plan["service_order"]:
                stage = service + "_preview"
                cmd = [plan["python"], str(H), "--control", plan["control"],
                       "--service", service, "--candidate", str(CANDIDATE),
                       "--freeze", str(args.manifest), "--baseline", str(D / "all-chat-before.json"),
                       "--output", str(D / (service + "-preview")), "--timeout", "1200"]
                save(service + "-preview-command.json", {"argv": cmd, "apply": False})
                with (D / (service + "-preview.stdout")).open("x") as out, (D / (service + "-preview.stderr")).open("x") as err:
                    result = subprocess.run(
                        cmd, cwd=B, env=dict(os.environ, PYTHONPATH=str(B / "src"), PYTHONDONTWRITEBYTECODE="1"),
                        stdout=out, stderr=err, timeout=90,
                    )
                save(service + "-preview-execution.json", {"returncode": result.returncode})
                assert result.returncode == 0
                path = D / (service + "-preview/receipt.json")
                record = g.read(path)
                assert record["status"] == "preview" and not record["apply_requested"]
                assert record["before"]["id"] == TARGETS[service]
                assert record["before"]["checkpoint_path"] == str(PARENT) and record["before"]["tree"] == PARENT_TREE
                assert record["candidate_identity"]["tree_sha256"] == proof["control_tree_sha256"]
                assert not record["compose_remove_orphans"]
                previews[service] = {"path": str(path), "sha256": g.sha(path)}
            stage = "postcheck"
            assert inventory()[0] == before
            assert g.inventory(PARENT) == parent["candidate_inventory_stat"]
            assert g.inventory(CANDIDATE) == proof["candidate_inventory_stat"]
            assert all(g.sha(p) == h for p, h in input_pins.items())
            assert all(g.sha(p) == h for p, h in plan["source_pins"].items())
            def ref(path):
                return {"path": str(path), "sha256": g.sha(path)}
            plan.update(
                status="four_previews_passed_awaiting_root_apply", candidate=str(CANDIDATE),
                freeze=ref(args.manifest), verification=ref(args.verify),
                baseline=ref(D / "all-chat-before.json"), protected_before=ref(D / "protected-before.json"),
                target_ids=TARGETS, protected_ids=protected, previews=previews,
                output=str(D / "parallel-apply-v1"), materialization=ref(args.verify),
                parent_decision=ref(DECISION), spec=ref(args.spec),
                preparation_source=ref(Path(__file__)), parent_manifest=ref(PARENT_MANIFEST),
                operation=logical, rejected_candidate_gate=ref(GATE),
            )
            save("plan-bound-unapproved.json", plan)
            save("four-preview-summary.json", {
                "status": "all_four_previews_passed", "seconds": time.time() - started,
                "previews": previews, "plan_sha256": g.sha(D / "plan-bound-unapproved.json"),
                "tree": proof["control_tree_sha256"], "input_pins": input_pins,
                "all27_unchanged": True, "protected_count": 23, "all9_stats_unchanged": True,
                "parent_fullstat_unchanged": True, "shared_stat_unchanged": True, "shared_nlink": 17,
                "model_calls": 0, "serving_mutations": 0, "payload_rehashed": False,
                "approved_for_apply": False,
            })
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock_held = False
        save("preview-lock-released.json", {"normal_release": True, "pid": os.getpid(), "epoch": time.time()})
    except BaseException as error:
        save("preview-failure.json", {
            "status": "failed", "stage": stage, "exception_type": type(error).__name__,
            "epoch": time.time(), "elapsed_seconds": time.time() - started,
            "lock_was_acquired": lock_held, "model_calls": 0, "serving_mutations": 0,
            "retry_performed": False,
        })
        raise
    finally:
        for path in D.rglob("*"):
            if path.is_file() and path.suffix != ".lock" and path.name != "prepare-source.py":
                target = PR / path.relative_to(D)
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists()
                shutil.copyfile(path, target)
    print((D / "four-preview-summary.json").read_text())


if __name__ == "__main__":
    main()
