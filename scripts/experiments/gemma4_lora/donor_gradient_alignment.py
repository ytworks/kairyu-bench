#!/usr/bin/env python3
"""Compare saved matrix gradients with stored BF16 donor edits, on sampled cells only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile

import numpy as np

from kairyu_bench.wsp.weights.checkpoint import CONFIG_FILES, Checkpoint

MAX_COORDINATES = 1_000_000
MAX_ARRAY_BYTES = 64 * 1024**2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if path.stat().st_size > MAX_ARRAY_BYTES:
        raise ValueError("metadata exceeds bounded input size")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def state(identity: dict) -> dict:
    return {row["path"]: {"size": row["size"], "stat": row["stat"]}
            for row in identity["files"]}


def metrics(gradient: np.ndarray, delta: np.ndarray) -> dict:
    g, d = gradient.astype(np.float64), delta.astype(np.float64)
    gn, dn = float(np.linalg.norm(g)), float(np.linalg.norm(d))
    dot = float(g @ d)
    active = (g != 0) & (d != 0)
    return {
        "coordinates": len(g), "changed_coordinates": int(np.count_nonzero(d)),
        "gradient_l2": gn, "stored_delta_l2": dn, "dot": dot,
        "cosine": dot / (gn * dn) if gn and dn else None,
        "sign_agreement": float(np.mean(np.sign(g[active]) == np.sign(d[active]))) if active.any() else None,
        "joint_nonzero_coordinates": int(active.sum()),
    }


def analyze(collection: Path, manifest_path: Path) -> dict:
    collection, manifest_path = collection.resolve(), manifest_path.resolve()
    files = [collection / n for n in ("summary.json", "coverage.json", "provenance.json", "sample-plan.npz")]
    files += [manifest_path, Path(__file__).resolve()]
    hashes = {str(p): sha(p) for p in files}
    summary, coverage, provenance = [read_json(collection / n) for n in ("summary.json", "coverage.json", "provenance.json")]
    manifest = read_json(manifest_path)
    if not isinstance(summary.get("mode"), str) or summary["mode"] != provenance.get("mode"):
        raise ValueError("summary/provenance objective mode mismatch")
    if (summary.get("status") != "completed" or not summary.get("source_values_unchanged")
            or summary.get("coverage") != coverage or not coverage.get("actual_values_crosschecked_against_safetensors")):
        raise ValueError("requires a completed, source-preserving matrix collection")
    if manifest.get("schema_version") != 2 or manifest.get("kind") != "dense_lora_recipe_v1":
        raise ValueError("requires the stored dense donor recipe manifest")
    base, merged = Checkpoint(manifest["source"]), Checkpoint(manifest["candidate"])
    declared = provenance["checkpoint"]
    base_identity = base.identity(hash_files=False)
    if Path(declared["path"]).resolve() != base.path or state(declared) != state(base_identity):
        raise ValueError("gradient source checkpoint state does not match current base")
    # Verify stored payloads, not a reimplementation of the historical merge.
    # Manifest and reader inventory conventions differ; compare per-file hashes.
    merged_identity = merged.identity(hash_files=True, chunk_bytes=1024**2)
    expected = {x["path"]: x for x in manifest["candidate_files"]}
    recognized = {name for name in expected if Path(name).suffix == ".safetensors"
                  or name == "model.safetensors.index.json" or name in CONFIG_FILES}
    if recognized != {row["path"] for row in merged_identity["files"]}:
        raise ValueError("stored merged recognized-file inventory differs from its recipe manifest")
    for row in merged_identity["files"]:
        old = expected.get(row["path"])
        if old is None or any(old[k] != row[k] for k in ("size", "sha256")):
            raise ValueError("stored merged checkpoint differs from its recipe manifest")
    with zipfile.ZipFile(collection / "sample-plan.npz") as archive:
        if sum(x.file_size for x in archive.infolist()) > MAX_ARRAY_BYTES:
            raise ValueError("sample arrays exceed bounded input size")
    with np.load(collection / "sample-plan.npz", allow_pickle=False) as arrays:
        coordinates = arrays["coordinates"]
        source_values = arrays["source_values"]
    n = len(coordinates)
    if (not 0 < n <= MAX_COORDINATES or coordinates.shape != (n, 2) or coordinates.dtype.kind not in "iu"
            or source_values.shape != (n,) or source_values.dtype != np.dtype("float32")
            or not np.isfinite(source_values).all() or coverage["sampled_coordinates"] != n):
        raise ValueError("invalid coordinate/source-value arrays")
    delta = np.empty(n, dtype=np.float64)
    seen, cursor, represented = set(), 0, 0
    groups: dict[str, list[tuple[int, int]]] = {}
    mapping = {x["tensor"]: x for x in manifest["edits"] if x.get("entry_kind") == "lora_matrix"}
    for entry in coverage["entries"]:
        name, start, count = entry["tensor"], entry["start"], entry["count"]
        if (name in seen or type(start) is not int or type(count) is not int
                or start != cursor or count <= 0 or start + count > n or name not in mapping):
            raise ValueError("duplicate, non-contiguous or non-donor coordinate entry")
        match = re.fullmatch(
            r"model\.language_model\.layers\.(\d+)\.(?:self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight",
            name,
        )
        if match is None or int(match[1]) != entry["layer"] or match[2].removesuffix("_proj") != entry["family"]:
            raise ValueError("matrix layer/family tuple mismatch")
        seen.add(name)
        info, target = base.tensors[name], merged.tensors[name]
        if (info.dtype != "BF16" or target.dtype != "BF16" or len(info.shape) != 2
                or info.shape != target.shape or list(info.shape) != entry["shape"]
                or list(info.shape) != mapping[name]["shape"]):
            raise ValueError("matrix tuple/shape/dtype mismatch")
        rc = coordinates[start:start + count]
        if np.any(rc < 0) or np.any(rc[:, 0] >= info.shape[0]) or np.any(rc[:, 1] >= info.shape[1]):
            raise ValueError("matrix coordinate outside tensor")
        indices = (rc[:, 0] * info.shape[1] + rc[:, 1]).tolist()
        if len(set(indices)) != count:
            raise ValueError("duplicate coordinates in matrix")
        original = base.read_indices(name, indices).astype(np.float32)
        saved = source_values[start:start + count]
        if not np.array_equal(original.view(np.uint32), saved.view(np.uint32)):
            raise ValueError("sampled base bytes do not match gradient source values")
        values = merged.read_indices(name, indices).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite stored merged values")
        delta[start:start + count] = values.astype(np.float64) - original.astype(np.float64)
        for group in (f"layer:{entry['layer']}", f"family:{entry['family']}"):
            groups.setdefault(group, []).append((start, start + count))
        cursor += count
        represented += info.numel
    if cursor != n or len(seen) != coverage["sampled_matrices"]:
        raise ValueError("coverage inventory mismatch")
    folds, fold_ids = [], set()
    if not 1 <= len(summary["folds"]) <= 64:
        raise ValueError("unsupported fold count")
    for fold in summary["folds"]:
        identity = (fold["purpose"], fold["fold"])
        if identity in fold_ids:
            raise ValueError("duplicate fold")
        fold_ids.add(identity)
        path = (collection / fold["file"]).resolve()
        if not path.is_relative_to(collection) or path.stat().st_size > MAX_ARRAY_BYTES:
            raise ValueError("gradient must be a bounded local array")
        hashes[str(path)] = sha(path)
        gradient = np.load(path, mmap_mode="r", allow_pickle=False)
        if gradient.shape != (n,) or gradient.dtype != np.dtype("float32") or not np.isfinite(gradient).all():
            raise ValueError("invalid gradient array")
        grouped = {}
        for group, spans in groups.items():
            idx = np.concatenate([np.arange(a, b) for a, b in spans])
            grouped[group] = metrics(gradient[idx], delta[idx])
        folds.append({"purpose": identity[0], "fold": identity[1],
                      "sampled": metrics(gradient, delta), "groups": grouped})
    base.assert_stable()
    merged.assert_stable()
    for path, digest in hashes.items():
        if sha(Path(path)) != digest:
            raise ValueError("analysis input changed while reading")
    return {
        "schema_version": 1, "status": "completed", "mode": summary["mode"],
        "delta_source": "actual_stored_BF16_checkpoint_bytes",
        "coefficient": manifest["recipe"]["coefficient"],
        "donor_repo": manifest["recipe"].get("donor_repo"),
        "donor_revision": manifest["recipe"].get("donor_revision"),
        "donor_sha256": manifest["recipe"]["donor_sha256"],
        "input_sha256": hashes, "base_identity": base_identity, "merged_identity": merged_identity,
        "manifest_candidate_tree_sha256": manifest["candidate_tree_sha256"],
        "sampled_coordinates": n, "sampled_matrices": len(seen), "represented_matrix_elements": represented,
        "sampled_element_fraction": n / represented, "sampled_delta_l2": float(np.linalg.norm(delta)),
        "stored_nonzero_delta_coordinates": int(np.count_nonzero(delta)), "folds": folds,
        "new_checkpoints": 0, "model_calls": 0, "benchmark_calls": 0,
        "limits": [
            "Inner products predict only the sparse edit at these sampled coordinates, not the full donor direction.",
            "Stratified cells are not an unbiased estimator of the full LoRA dot product; no extrapolation is made.",
            "Continuous local prediction is not a finite nonlinear outcome or a capability score.",
            "Native norm edits occupy other tensors; their interaction still requires a joint finite check.",
            "Historical donor training-base and training/evaluation overlap uncertainty remain.",
            "Source identity uses stored stat continuity plus exact sampled values; source full payload is not rehashed.",
            "Merged recognized files are fully hashed against the manifest; this reader's tree digest is not the recipe tree digest.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = read_json(args.merged_manifest)
        output = args.output.resolve()
        protected = [args.collection.resolve(), Path(manifest["source"]).resolve(), Path(manifest["candidate"]).resolve()]
        if args.output.exists() or args.output.is_symlink() or any(output.is_relative_to(p) for p in protected):
            raise ValueError("output must be fresh and outside all checkpoint/collection inputs")
        result = analyze(args.collection, args.merged_manifest)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "sampled_coordinates": result["sampled_coordinates"],
                      "delta_source": result["delta_source"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
