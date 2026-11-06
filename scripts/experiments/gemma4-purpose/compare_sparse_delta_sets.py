#!/usr/bin/env python3
"""Compare realized sparse BF16 deltas from one pinned source snapshot; never compose edits."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
from kairyu_bench.wsp.weights import composition
from kairyu_bench.wsp.weights.composition import _bf16


def geometry(base, variants):
    """The existing composition diagnostic's pairwise arithmetic, without a merged vector."""
    source = _bf16(base, "base")
    names = list(variants)
    if not 2 <= len(names) <= 8:
        raise ValueError("two through eight named variants required")
    values = {name: _bf16(variants[name], name) for name in names}
    if any(v.shape != source.shape for v in values.values()):
        raise ValueError("all variants must share the exact coordinate scope")
    delta = {name: value.astype(np.float64) - source.astype(np.float64)
             for name, value in values.items()}
    components = {name: dict(changed=int(np.count_nonzero(v)),
                            stored_bit_changes=int(np.count_nonzero(
                                values[name].view(np.uint32) != source.view(np.uint32))),
                            raw_delta_l2=float(np.linalg.norm(v)))
                  for name, v in delta.items()}
    pairs = []
    for left, right in itertools.combinations(names, 2):
        a, b = delta[left], delta[right]
        active_a, active_b = a != 0, b != 0
        overlap, union = active_a & active_b, active_a | active_b
        same = overlap & (np.signbit(a) == np.signbit(b))
        opposite = overlap & ~same
        na, nb = components[left]["raw_delta_l2"], components[right]["raw_delta_l2"]
        common_a, common_b = float(np.linalg.norm(a[overlap])), float(np.linalg.norm(b[overlap]))
        dot = float(a @ b)
        pairs.append(dict(
            left=left, right=right, overlap=int(overlap.sum()), union=int(union.sum()),
            left_only=int((active_a & ~active_b).sum()), right_only=int((active_b & ~active_a).sum()),
            support_jaccard=float(overlap.sum()/union.sum()) if union.any() else None,
            same_sign=int(same.sum()), opposite_sign=int(opposite.sum()),
            exact_opposite_cancellation=int((opposite & ((a+b) == 0)).sum()),
            identical_target_on_overlap=int(np.count_nonzero(overlap & (
                values[left].view(np.uint32) == values[right].view(np.uint32)))),
            raw_dot=dot, raw_cosine=dot/(na*nb) if na and nb else None,
            left_common_raw_l2=common_a, right_common_raw_l2=common_b,
            left_common_norm2_fraction=(common_a/na)**2 if na else None,
            right_common_norm2_fraction=(common_b/nb)**2 if nb else None,
            common_raw_cosine=dot/(common_a*common_b) if common_a and common_b else None,
            delta_difference_l2=float(np.linalg.norm(a-b))))
    return dict(components=components, pairs=pairs, selected_scope=len(source))


def compare(postcheck_path, expected_sha, output):
    pins, stamps = {}, {}
    def load(path, expected):
        p = Path(path).resolve()
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError("input hash differs")
        pins[str(p)] = digest
        s = p.stat()
        stamps[str(p)] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        return data
    postcheck_path, output = Path(postcheck_path).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("fresh output required")
    for implementation in (Path(__file__).resolve(), Path(composition.__file__).resolve()):
        load(implementation, hashlib.sha256(implementation.read_bytes()).hexdigest())
    post = json.loads(load(postcheck_path, expected_sha))
    rows = post["results"]
    if not 2 <= len(rows) <= 8 or any(r["status"] != "exact_spec_postchecked" for r in rows):
        raise ValueError("two through eight exact postchecked proposals required")
    plans, specs = [], []
    for row in rows:
        plans.append(json.loads(load(row["plan"], row["plan_sha256"])))
        specs.append(json.loads(load(row["spec"], row["spec_sha256"])))
    first = plans[0]
    if any(p["checkpoint"] != first["checkpoint"] or p["collections"] != first["collections"]
           for p in plans):
        raise ValueError("proposals have different source snapshots")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique proposal IDs required")
    roots = [Path(first["checkpoint"]["path"]).resolve()]
    collections = first["collections"]
    native_dir = Path(collections["native"]["correct-logprob"]).resolve()
    matrix_dir = Path(collections["matrix"]["correct-logprob"]).resolve()
    roots.extend(Path(path).resolve() for group in collections.values() for path in group.values())
    selected, needed_native = {}, False
    for row, spec in zip(rows, specs, strict=True):
        if (spec["id"] != row["id"] or spec["source_checkpoint"] != first["checkpoint"]["path"]
                or spec["base_tree_sha256"] != first["checkpoint"]["tree_sha256"]):
            raise ValueError("spec source identity differs")
        edits = {}
        for edit in spec["edits"]:
            native = "indices" in edit
            needed_native |= native
            positions = edit["indices"] if native else edit["coordinates"]
            for position, value in zip(positions, edit["set_values"], strict=True):
                key = (edit["tensor"], (int(position),) if native else tuple(position))
                if key in edits:
                    raise ValueError("duplicate edited coordinate")
                edits[key] = value
        selected[row["id"]] = edits
    keys = sorted(set().union(*(set(v) for v in selected.values())))
    if not 1 <= len(keys) <= 2000000:
        raise ValueError("bounded nonempty sparse union required")
    sources = {}
    if needed_native:
        p = native_dir / "source-native-values.npz"
        load(p, post["input_sha256"][str(p)])
        with np.load(p, allow_pickle=False) as arrays:
            for tensor, position in keys:
                if len(position) == 1:
                    sources[(tensor, position)] = float(arrays[tensor].reshape(-1)[position[0]])
    coverage_path, sample_path = matrix_dir / "coverage.json", matrix_dir / "sample-plan.npz"
    coverage = json.loads(load(coverage_path, post["input_sha256"][str(coverage_path)]))
    load(sample_path, post["input_sha256"][str(sample_path)])
    needed = set(keys)
    with np.load(sample_path, allow_pickle=False) as arrays:
        for entry in coverage["entries"]:
            start, count = entry["start"], entry["count"]
            for rc, value in zip(arrays["coordinates"][start:start+count],
                                 arrays["source_values"][start:start+count], strict=True):
                key = (entry["tensor"], tuple(map(int, rc)))
                if key in needed:
                    if key in sources:
                        raise ValueError("duplicate source coordinate")
                    sources[key] = float(value)
    if set(sources) != needed:
        raise ValueError("source snapshot lacks selected coordinates")
    base = np.asarray([sources[key] for key in keys], dtype=np.float64)
    variants = {name: np.asarray([edits.get(key, sources[key]) for key in keys])
                for name, edits in selected.items()}
    result = geometry(base, variants)
    for row in rows:
        actual = result["components"][row["id"]]
        if (actual["changed"] != row["actual_changed"]
                or not np.isclose(actual["raw_delta_l2"], row["raw_delta_l2"], rtol=1e-12, atol=1e-12)):
            raise ValueError("postchecked count or raw norm differs")
    families = {}
    for index, (tensor, position) in enumerate(keys):
        match = re.search(r"\.(q|k|v|o|gate|up|down)_proj\.weight$", tensor)
        family = match[1] if len(position) == 2 and match else "native"
        families.setdefault(family, []).append(index)
    result["families"] = {family: geometry(base[ix], {n: v[ix] for n, v in variants.items()})
                          for family, ix in families.items()}
    target = output.resolve()
    if any(target == root or root in target.parents or target in root.parents for root in roots):
        raise ValueError("output overlaps a source root")
    if any(target == Path(p) or target in Path(p).parents for p in pins):
        raise ValueError("output overlaps a pinned input")
    for name, digest in pins.items():
        p = Path(name)
        s = p.stat()
        if ((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) != stamps[name]
                or hashlib.sha256(p.read_bytes()).hexdigest() != digest):
            raise ValueError("input changed during analysis")
    result.update(schema_version=1, source_checkpoint=first["checkpoint"], parent=post["parent"],
                  input_sha256=pins, model_calls=0, checkpoint_payload_reads=0,
                  candidate_or_spec_writes=0, merged_values_computed=False,
                  limits=["Raw weight geometry is not functional additivity or capability complementarity.",
                          "Absolute old set_values must not overwrite a future adopted parent's changes.",
                          "Future composition requires source-word conflict checks and fresh-parent observation."])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postcheck", required=True, type=Path)
    parser.add_argument("--postcheck-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.postcheck, args.postcheck_sha256, args.output)
    print(json.dumps(dict(path=str(args.output), components=result["components"], pairs=result["pairs"])))


if __name__ == "__main__":
    main()
