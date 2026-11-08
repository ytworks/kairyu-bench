#!/usr/bin/env python3
"""Compose disjoint verified tensor bytes; preview by default, no model or GEMM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from kairyu_bench.wsp.weights.checkpoint import Checkpoint
from kairyu_bench.wsp.weights import editor, checkpoint

_COMMON = Path(__file__).parent / "gemma4_lora/verify_saved_checkpoint.py"
_spec = importlib.util.spec_from_file_location("compose_proof_common", _COMMON)
assert _spec is not None and _spec.loader is not None
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
CHUNK = 8 * 1024**2
KIND = "verified_tensor_byte_composition_v1"


def overlap(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def pin(record, pins):
    path = Path(record["path"]).resolve()
    value, state = common.captured(path, record["sha256"])
    pins[path] = (record["sha256"], state)
    return value


def manifest_files(manifest):
    if manifest.get("schema_version") == 1 and manifest.get("kind") is None:
        editor.validate_manifest(manifest)
        source = [
            {"path": r["path"], "size": r["size"], "sha256": r["source_sha256"]}
            for r in manifest["files"]
        ]
        candidate = [
            {"path": r["path"], "size": r["size"], "sha256": r["candidate_sha256"]}
            for r in manifest["files"]
        ]
    elif (
        manifest.get("schema_version") == 2
        and manifest.get("kind") == "dense_lora_recipe_v1"
    ):
        source, candidate = manifest["source_files"], manifest["candidate_files"]
    else:
        raise ValueError(
            "only native schema1 and dense schema2 component manifests are supported"
        )
    return common.file_rows(source), common.file_rows(candidate)


def layout(reader):
    return {
        name: (t.relative_shard, t.offset, t.nbytes, t.dtype, t.shape)
        for name, t in reader.tensors.items()
    }


def prepare(plan_path, plan_sha, *, creating):
    pins = {}
    plan = pin({"path": str(plan_path), "sha256": plan_sha}, pins)
    if plan.get("schema_version") != 1 or plan.get("kind") != KIND:
        raise ValueError("unsupported composition plan")
    raw_outputs = [
        Path(plan[key]).expanduser().absolute()
        for key in ("candidate", "manifest", "freeze")
    ]
    if any(path.is_symlink() for path in raw_outputs) or (
        creating and any(os.path.lexists(path) for path in raw_outputs)
    ):
        raise ValueError(
            "candidate, manifest and freeze must be fresh nonsymlink paths"
        )
    for attestation in plan.get("attestations", []):
        pin(attestation, pins)
    source = Path(plan["source"]["path"]).resolve()
    candidate = Path(plan["candidate"]).resolve()
    outputs = [Path(plan[key]).resolve() for key in ("manifest", "freeze")]
    if not source.is_dir() or Path(plan["source"]["path"]).is_symlink():
        raise ValueError("source must be a real directory")
    components = plan["components"]
    if not isinstance(components, list) or len(components) < 2:
        raise ValueError("at least two explicitly pinned components required")
    roots = [source] + [Path(c["path"]).resolve() for c in components]
    if len(set(roots)) != len(roots) or any(
        overlap(a, b) for i, a in enumerate(roots) for b in roots[i + 1 :]
    ):
        raise ValueError("component/source roots must be separate")
    if any(overlap(candidate, p) for p in roots) or any(
        overlap(p, q) for p in outputs for q in roots + [candidate]
    ):
        raise ValueError("output overlaps an input checkpoint")
    if len(set(outputs)) != 2 or candidate in outputs:
        raise ValueError("outputs collide")
    if not candidate.parent.is_dir() or any(not p.parent.is_dir() for p in outputs):
        raise ValueError("output parents must exist")
    if creating and (
        os.path.lexists(candidate) or any(os.path.lexists(p) for p in outputs)
    ):
        raise ValueError("candidate, manifest and freeze must be fresh")
    base = Checkpoint(source)
    base_layout = layout(base)
    states = {root: common.inventory(root) for root in roots}
    selected = {}
    expected_source = None
    prepared = []
    for number, component in enumerate(components):
        root = roots[number + 1]
        if Path(component["path"]).is_symlink():
            raise ValueError("component symlink refused")
        manifest = pin(component["manifest"], pins)
        if (
            Path(manifest["source"]).resolve() != source
            or Path(manifest["candidate"]).resolve() != root
        ):
            raise ValueError("component manifest identity mismatch")
        source_files, files = manifest_files(manifest)
        if expected_source is None:
            expected_source = source_files
        if source_files != expected_source or set(files) != set(source_files):
            raise ValueError("components do not share exact original files")
        for folder, expected in ((source, source_files), (root, files)):
            if set(states[folder]) != set(expected) or any(
                states[folder][n]["size"] != r["size"] for n, r in expected.items()
            ):
                raise ValueError("full file inventory/size differs")
        reader = Checkpoint(root)
        if layout(reader) != base_layout:
            raise ValueError("all tensor headers/layouts/dtypes must match")
        mode = component.get("input_verification", "fresh_hash")
        if mode not in ("fresh_hash", "frozen_stat"):
            raise ValueError("unknown input verification mode")
        freeze = None
        if component.get("freeze"):
            freeze = pin(component["freeze"], pins)
            if (
                freeze.get("status")
                not in (
                    "verified_dense_cpu_recipe",
                    "verified_saved_bf16_against_ram",
                    "verified_tensor_byte_splice",
                )
                or freeze.get("source_tree_sha256") != plan["source"]["tree_sha256"]
            ):
                raise ValueError(
                    "recognized verification proof and matching source tree required"
                )
            if (
                Path(freeze["checkpoint_path"]).resolve() != root
                or common.file_rows(freeze["all_checkpoint_files"]) != files
                or freeze["control_tree_sha256"]
                != common.canonical([files[n] for n in sorted(files)])
                or freeze.get("manifest_sha256") != component["manifest"]["sha256"]
            ):
                raise ValueError("component freeze binding mismatch")
            if (
                mode == "frozen_stat"
                and freeze["candidate_inventory_stat"] != states[root]
            ):
                raise ValueError("frozen component stat changed")
        if mode == "frozen_stat" and freeze is None:
            raise ValueError(
                "prior hash reuse requires a pinned freeze with exact full stat"
            )
        seen = set()
        for entry in manifest["edits"]:
            name = entry["tensor"]
            if name in seen or name in selected or name not in base.tensors:
                raise ValueError("overlapping or unknown edited tensors")
            seen.add(name)
            t = base.tensors[name]
            if (
                entry["shard"],
                entry["offset"],
                entry["size"],
                entry["dtype"],
                tuple(entry["shape"]),
            ) != base_layout[name]:
                raise ValueError("entry must name the complete exact tensor span")
            if t.dtype != "BF16":
                raise ValueError(
                    "initial composition contract requires BF16 edited tensors"
                )
            for key in ("source_tensor_sha256", "candidate_tensor_sha256"):
                if not isinstance(entry[key], str) or len(entry[key]) != 64:
                    raise ValueError("tensor SHA required")
                int(entry[key], 16)
            if (
                type(entry["changed_count"]) is not int
                or not 0 <= entry["changed_count"] <= t.numel
                or not math.isfinite(entry["delta_l2"])
                or entry["delta_l2"] < 0
            ):
                raise ValueError("invalid verified component metrics")
            selected[name] = {
                k: entry[k]
                for k in (
                    "tensor",
                    "shard",
                    "offset",
                    "size",
                    "dtype",
                    "shape",
                    "source_tensor_sha256",
                    "candidate_tensor_sha256",
                    "changed_count",
                    "delta_l2",
                )
            }
            selected[name]["component"] = number
        if (
            not seen
            or manifest["changed_tensors"] != len(seen)
            or manifest["changed_elements"]
            != sum(selected[n]["changed_count"] for n in seen)
            or not math.isclose(
                manifest["delta_l2"],
                math.hypot(*(selected[n]["delta_l2"] for n in seen)),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("component metric totals differ")
        prepared.append({"path": root, "files": files, "mode": mode, "reader": reader})
    assert expected_source is not None
    if (
        common.canonical([expected_source[n] for n in sorted(expected_source)])
        != plan["source"]["tree_sha256"]
    ):
        raise ValueError("original tree pin mismatch")
    for p in [candidate, *outputs]:
        if p in pins or any(overlap(p, q) for q in pins):
            raise ValueError("output overlaps pinned input")
    for path in (
        Path(__file__).resolve(),
        _COMMON.resolve(),
        Path(editor.__file__).resolve(),
        Path(checkpoint.__file__).resolve(),
    ):
        pins[path] = (common.sha(path), common.stamp(path))
    return {
        "plan": plan,
        "pins": pins,
        "source": source,
        "candidate": candidate,
        "outputs": outputs,
        "states": states,
        "source_files": expected_source,
        "components": prepared,
        "edits": selected,
    }


def stable(p):
    for root, old in p["states"].items():
        if common.inventory(root) != old:
            raise ValueError("input checkpoint changed")
    for path, (digest, state) in p["pins"].items():
        if common.stamp(path) != state or common.sha(path) != digest:
            raise ValueError("pinned metadata/implementation changed")


def hash_inputs(p):
    # The original is always freshly hashed; old native inputs without freezes too.
    for name, row in p["source_files"].items():
        if common.sha(p["source"] / name) != row["sha256"]:
            raise ValueError("original full-file hash mismatch")
    for component in p["components"]:
        if component["mode"] == "fresh_hash":
            for name, row in component["files"].items():
                if common.sha(component["path"] / name) != row["sha256"]:
                    raise ValueError("component full-file hash mismatch")
    stable(p)


def copy_component(p, owned):
    component = p["components"][0]
    for name, expected in component["files"].items():
        target = p["candidate"] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with target.open("xb") as out:
            s = common.stamp(target)
            owned[target] = (s["device"], s["inode"])
            with (component["path"] / name).open("rb") as inp:
                while block := inp.read(CHUNK):
                    out.write(block)
                    digest.update(block)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("copied component hash mismatch")


def splice(p):
    for entry in p["edits"].values():
        if entry["component"] == 0:
            continue
        component = p["components"][entry["component"]]["path"]
        with (
            (component / entry["shard"]).open("rb") as inp,
            (p["candidate"] / entry["shard"]).open("r+b") as out,
        ):
            inp.seek(entry["offset"])
            out.seek(entry["offset"])
            remaining = entry["size"]
            while remaining:
                block = inp.read(min(CHUNK, remaining))
                if not block:
                    raise ValueError("short component tensor")
                out.write(block)
                remaining -= len(block)
            out.flush()
            os.fsync(out.fileno())


def verify_bytes(p):
    dest = p["candidate"]
    before = common.inventory(dest)
    if set(before) != set(p["source_files"]) or layout(Checkpoint(dest)) != layout(
        Checkpoint(p["source"])
    ):
        raise ValueError("candidate inventory/layout mismatch")
    for name, state in before.items():
        if (
            state["uid"] != os.getuid()
            or state["nlink"] != 1
            or any(os.path.samefile(dest / name, root / name) for root in p["states"])
        ):
            raise ValueError("candidate must contain owned independent private files")
    rows = []
    component_rows = [[] for _ in p["components"]]
    for name, original in p["source_files"].items():
        entries = sorted(
            (e for e in p["edits"].values() if e["shard"] == name),
            key=lambda e: e["offset"],
        )
        current, restored = hashlib.sha256(), hashlib.sha256()
        component_hashes = [hashlib.sha256() for _ in p["components"]]
        with (dest / name).open("rb") as saved, (p["source"] / name).open("rb") as base:
            position = 0
            for entry in entries:
                left = entry["offset"] - position
                if left < 0:
                    raise ValueError("overlapping spans")
                while left:
                    raw = saved.read(min(CHUNK, left))
                    if not raw:
                        raise ValueError("short candidate")
                    current.update(raw)
                    restored.update(raw)
                    for digest in component_hashes:
                        digest.update(raw)
                    left -= len(raw)
                base.seek(entry["offset"])
                sh, ch = hashlib.sha256(), hashlib.sha256()
                left = entry["size"]
                while left:
                    raw = saved.read(min(CHUNK, left))
                    old = base.read(len(raw))
                    if not raw or len(old) != len(raw):
                        raise ValueError("short tensor")
                    current.update(raw)
                    restored.update(old)
                    for index, digest in enumerate(component_hashes):
                        digest.update(raw if index == entry["component"] else old)
                    sh.update(old)
                    ch.update(raw)
                    left -= len(raw)
                if (
                    sh.hexdigest() != entry["source_tensor_sha256"]
                    or ch.hexdigest() != entry["candidate_tensor_sha256"]
                ):
                    raise ValueError(
                        "selected tensor differs from verified source/component bytes"
                    )
                position = entry["offset"] + entry["size"]
            while raw := saved.read(CHUNK):
                current.update(raw)
                restored.update(raw)
                for digest in component_hashes:
                    digest.update(raw)
        if (
            restored.hexdigest() != original["sha256"]
            or before[name]["size"] != original["size"]
        ):
            raise ValueError("undeclared candidate byte difference")
        for index, digest in enumerate(component_hashes):
            if digest.hexdigest() != p["components"][index]["files"][name]["sha256"]:
                raise ValueError(
                    "component virtual restoration differs from its full-file hash"
                )
            component_rows[index].append(
                {
                    "path": name,
                    "size": before[name]["size"],
                    "sha256": digest.hexdigest(),
                }
            )
        rows.append(
            {"path": name, "size": before[name]["size"], "sha256": current.hexdigest()}
        )
    stable(p)
    if common.inventory(dest) != before:
        raise ValueError("candidate changed during verification")
    virtual = [
        {
            "path": str(component["path"]),
            "retained_component_index": index,
            "files": sorted(component_rows[index], key=lambda r: r["path"]),
            "tree_sha256": common.canonical(
                sorted(component_rows[index], key=lambda r: r["path"])
            ),
        }
        for index, component in enumerate(p["components"])
    ]
    return sorted(rows, key=lambda r: r["path"]), before, virtual


def publish(path, value, owned):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=".compose-", delete=False
        ) as out:
            temporary = Path(out.name)
            json.dump(value, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        s = common.stamp(temporary)
        owned[path] = (s["device"], s["inode"])
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def owns(path, identity):
    if path.is_symlink() or not path.exists():
        return False
    s = path.stat()
    return (s.st_dev, s.st_ino) == identity


def compose(plan_path, plan_sha, *, apply=False):
    p = prepare(plan_path, plan_sha, creating=True)
    plan = p["plan"]
    dest = p["candidate"]
    manifest, freeze = p["outputs"]
    need = sum(r["size"] for r in p["source_files"].values())
    reserve = plan.get("reserve_bytes", 0)
    if type(reserve) is not int or reserve < 0:
        raise ValueError("invalid reserve")
    if shutil.disk_usage(dest.parent).free < need + reserve:
        raise ValueError("insufficient free bytes")
    summary = {
        "candidate": plan["id"],
        "changed_tensors": len(p["edits"]),
        "changed_elements": sum(e["changed_count"] for e in p["edits"].values()),
        "delta_l2": math.hypot(*(e["delta_l2"] for e in p["edits"].values())),
        "copy_bytes": need,
        "reserve_bytes": reserve,
    }
    stable(p)
    if not apply:
        return {
            "status": "preview",
            **summary,
            "checkpoint_writes": 0,
            "payload_hashed": False,
        }
    hash_inputs(p)
    owned = {}
    publications = {}
    dest.mkdir()
    dest_id = (dest.stat().st_dev, dest.stat().st_ino)
    try:
        copy_component(p, owned)
        splice(p)
        rows, states, virtual = verify_bytes(p)
        record = {
            "schema_version": 1,
            "kind": KIND,
            "status": "verified_tensor_byte_splice",
            **summary,
            "source": str(p["source"]),
            "candidate_path": str(dest),
            "source_tree_sha256": plan["source"]["tree_sha256"],
            "candidate_files": rows,
            "candidate_tree_sha256": common.canonical(rows),
            "edits": list(p["edits"].values()),
            "plan_path": str(Path(plan_path).resolve()),
            "plan_sha256": plan_sha,
            "plan": plan,
            "input_sha256": {str(path): value[0] for path, value in p["pins"].items()},
            "input_inventory_stat": {
                str(path): value for path, value in p["states"].items()
            },
            "candidate_inventory_stat": states,
            "component_virtual_restorations": virtual,
            "input_verification": [
                {"path": str(c["path"]), "mode": c["mode"]} for c in p["components"]
            ],
            "proof_method": "Exact full-tensor byte splice; component tensor SHA matches; full candidate hash and original-byte restoration.",
            "metrics_origin": "Verified component actual stored-byte metrics, combined over disjoint tensor sets; no new floating weight arithmetic.",
            "verification": {
                "full_original_hash": True,
                "all_selected_tensor_hashes": True,
                "untouched_spans": True,
                "component_virtual_restorations": True,
                "private_files": True,
            },
            "model_calls": 0,
            "gemm_calls": 0,
            "weight_rounding_operations": 0,
        }
        publish(manifest, record, publications)
        frozen = {
            "schema_version": 1,
            "status": "verified_tensor_byte_splice",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(dest),
            "candidate": plan["id"],
            "candidate_inventory_stat": states,
            "all_checkpoint_files": rows,
            "control_tree_sha256": common.canonical(rows),
            "source_tree_sha256": plan["source"]["tree_sha256"],
            "manifest_path": str(manifest),
            "manifest_sha256": common.sha(manifest),
            "input_sha256": record["input_sha256"],
            "proof_method": record["proof_method"],
            "changed_elements": summary["changed_elements"],
            "changed_tensors": summary["changed_tensors"],
            "delta_l2": summary["delta_l2"],
            "fresh_candidate_full_hash": True,
            "ram_comparison_performed": False,
            "dense_recipe_regenerated": False,
            "component_virtual_restorations": virtual,
            "model_calls": 0,
            "gemm_calls": 0,
        }
        publish(freeze, frozen, publications)
        return {
            "status": "completed",
            **summary,
            "manifest": str(manifest),
            "freeze": str(freeze),
            "tree_sha256": frozen["control_tree_sha256"],
        }
    finally:
        # Publishing our own verified manifest is the commit boundary. A later
        # freeze failure preserves the completed candidate for explicit recovery.
        committed = manifest in publications and owns(manifest, publications[manifest])
        if not committed and owns(dest, dest_id):
            for path, ino in owned.items():
                if owns(path, ino):
                    path.unlink()
            for path in sorted(
                (x for x in dest.rglob("*") if x.is_dir() and not x.is_symlink()),
                key=lambda x: len(x.parts),
                reverse=True,
            ):
                try:
                    path.rmdir()
                except OSError:
                    pass
            try:
                dest.rmdir()
            except OSError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(compose(args.plan, args.plan_sha256, apply=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
