#!/usr/bin/env python3
"""Fixed scalar archive release; unchanged F02 engine, one explicit byte-budget AST substitution."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import stat
import sys
import types

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
C = R / "checkpoints/scalar-l41-minus4"
A = R / "analysis/fresh-purpose-20261107-scalar-l41-minus4-lossless-v1"
D = R / "analysis/fresh-purpose-20261107-scalar-l41-minus4-release-plan-v1"
H = Path(__file__).with_name("fresh-f02-payload-release-20261107.py")
HSHA = "435e01109887123db12519c100e304fa7bd4f5a71b772e6c96eedce1b130f894"
OLD_BYTES, NEW_BYTES = 62546338264, 62546338248


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_engine():
    assert sha(H) == HSHA
    tree = ast.parse(H.read_text())
    before = ast.dump(tree, include_attributes=False)
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == OLD_BYTES:
            changes.append(dict(line=node.lineno, before=OLD_BYTES, after=NEW_BYTES))
            node.value = NEW_BYTES
    assert len(changes) == 1 and changes[0]["after"] == 62546338248
    after = ast.dump(tree, include_attributes=False)
    # Only this literal differs; control flow and unlink operations remain exact.
    assert before.replace("Constant(value=" + str(OLD_BYTES) + ")",
                          "Constant(value=" + str(NEW_BYTES) + ")") == after
    module = types.ModuleType("scalar_release_original_engine")
    module.__file__ = str(H)
    exec(compile(tree, str(H), "exec"), module.__dict__)
    proof = dict(original_source_sha256=HSHA, replacements=changes,
                 original_ast_sha256=hashlib.sha256(before.encode()).hexdigest(),
                 transformed_ast_sha256=hashlib.sha256(after.encode()).hexdigest(),
                 only_byte_literal_changed=True)
    return module, proof


def configure(module, plan, plan_path, plan_sha, wrapper_sha):
    base = Path(plan["mandatory_base"])
    assert base == Path("/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/models/gemma-4-31b-it")
    module.BASE, module.CANDIDATE, module.ARCHIVE = base, C, A
    module.PLAN, module.PLAN_SHA = plan_path, plan_sha

    def archive_check(value):
        assert sha(A / "SHA256SUMS") == value["archive_checksum_list_sha256"]
        assert sha(A / "archive.json") == value["archive_json_sha256"]
        before = module.inventory(A)
        checks = []
        for line in (A / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            p = A / name
            assert p.resolve().is_relative_to(A) and not p.is_symlink() and module.sha(p) == digest
            checks.append(dict(path=name, sha256=digest))
        assert len(checks) == value["archive_checksum_entries"]
        assert len({v["path"] for v in checks}) == len(checks)
        assert set(before) == {v["path"] for v in checks} | {"SHA256SUMS"}
        assert module.inventory(A) == before
        archive = json.loads((A / "archive.json").read_text())
        verified = json.loads((A / "fresh-verify.json").read_text())
        validation = json.loads((A / "archive-validation.json").read_text())
        assert archive["mandatory_base_path"] == str(base) and archive["candidate_path"] == str(C)
        assert archive["streamed_restoration_equals_original_base"] and archive["files"] == verified["files"]
        assert verified["verified"] and verified["legacy_full_hashes_missing"]
        assert len(verified["files"]) == 12 and verified["changed_elements"] == verified["changed_tensors"] == 1
        assert validation["status"] == "passed" and validation["all12_fresh_hashes"]
        return archive, before, checks

    def checkpoint_guard(archive, released, marker_present):
        original, candidate = archive["source_stats"], archive["candidate_stats"]
        assert len(original) == len(candidate) == 12
        original_current = module.inventory(base)
        assert original_current == plan["complete_original_inventory"]
        assert {name: original_current[name] for name in original} == original
        current = module.inventory(C)
        if marker_present:
            current.pop("ARCHIVED.json")
        assert current == {k: v for k, v in candidate.items() if k not in released}
        for name in module.NAMES:
            s, b = candidate[name], original[name]
            assert stat.S_ISREG(s["mode"]) and s["uid"] == s["gid"] == 1003 and s["nlink"] == 1
            assert (s["device"], s["inode"]) != (b["device"], b["inode"])
        for root, expected in plan["protected_checkpoint_inventories"].items():
            assert module.inventory(Path(root)) == expected
        return original, candidate

    original_mounts = module.mounts
    original_references = module.references

    def guarded_mounts():
        assert sha(H) == HSHA and sha(__file__) == wrapper_sha and sha(plan_path) == plan_sha
        return original_mounts()

    def references(targets):
        result = original_references(targets)
        assert plan["unknown_process_visibility_accepted_by_root"] is True
        result["inaccessible_is_unknown_not_zero"] = True
        result["root_accepted_fixed_candidate_scope"] = True
        return result

    module.archive_check = archive_check
    module.checkpoint_guard = checkpoint_guard
    module.mounts = guarded_mounts
    module.references = references


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    args, remaining = parser.parse_known_args()
    wrapper_sha = sha(__file__)
    assert wrapper_sha == args.expected_wrapper_sha256 and os.getuid() == 1003
    assert args.plan.parent == D and sha(args.plan) == args.plan_sha256
    plan = json.loads(args.plan.read_text())
    assert plan["wrapper_sha256"] == wrapper_sha and plan["helper_sha256"] == HSHA
    module, proof = load_engine()
    assert proof == plan["engine_ast_transformation"]
    configure(module, plan, args.plan, args.plan_sha256, wrapper_sha)
    sys.argv = [str(H), *remaining]
    module.main()


if __name__ == "__main__":
    main()
