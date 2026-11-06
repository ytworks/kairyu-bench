#!/usr/bin/env python3
"""Numeric postcheck of fixed proposal artifacts against their recorded current-parent source arrays."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True, type=Path)
    ap.add_argument("--observation-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    obs = args.observation_root.resolve()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or obs == args.output.resolve()
        or obs in args.output.resolve().parents
    ):
        raise ValueError("fresh output outside source observations required")
    roster = json.loads(args.roster.read_text())
    inputs = {
        args.roster.resolve(),
        obs / "native-correct-logprob/source-native-values.npz",
        obs / "matrix-correct-logprob/coverage.json",
        obs / "matrix-correct-logprob/sample-plan.npz",
    }
    for item in roster["rows"]:
        inputs.add(Path(item["plan"]).resolve())
        planned_input = json.loads(Path(item["plan"]).read_text())
        if "prospective_method" in planned_input:
            inputs.add(Path(planned_input["prospective_method"]["path"]).resolve())
        report_path = Path(item["output"]) / "report.json"
        if report_path.is_file():
            inputs.add(report_path.resolve())
            for ident in item["ids"]:
                spec_path = Path(item["output"]) / (ident + ".json")
                if spec_path.is_file():
                    inputs.add(spec_path.resolve())
    pins = {str(p): sha(p) for p in inputs}
    stamps = {
        str(p): (
            p.stat().st_dev,
            p.stat().st_ino,
            p.stat().st_size,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
        )
        for p in inputs
    }
    if any(
        args.output.resolve() == p or args.output.resolve() in p.parents for p in inputs
    ):
        raise ValueError("output overlaps input")

    with np.load(
        obs / "native-correct-logprob/source-native-values.npz", allow_pickle=False
    ) as f:
        native = {n: np.asarray(f[n], dtype=np.float64).reshape(-1) for n in f.files}
    cov = json.loads((obs / "matrix-correct-logprob/coverage.json").read_text())
    matrix = {}
    with np.load(
        obs / "matrix-correct-logprob/sample-plan.npz", allow_pickle=False
    ) as f:
        vals = np.asarray(f["source_values"], dtype=np.float64)
        coords = f["coordinates"]
        for e in cov["entries"]:
            sl = slice(e["start"], e["start"] + e["count"])
            matrix[e["tensor"]] = {
                tuple(map(int, c)): float(v)
                for c, v in zip(coords[sl], vals[sl], strict=True)
            }
    results = []
    for item in roster["rows"]:
        plan_path = Path(item["plan"]).resolve()
        if pins[str(plan_path)] != item["plan_sha256"]:
            raise ValueError("roster plan SHA mismatch")
        plan = json.loads(plan_path.read_text())
        checkpoint_path = Path(plan["checkpoint"]["path"]).resolve()
        if (
            checkpoint_path == args.output.resolve()
            or checkpoint_path in args.output.resolve().parents
        ):
            raise ValueError("output overlaps checkpoint")
        if "prospective_method" in plan:
            prospective = plan["prospective_method"]
            policy_path = Path(prospective["path"])
            if pins[str(policy_path.resolve())] != prospective["sha256"]:
                raise ValueError("prospective policy pin differs")
            policy = json.loads(policy_path.read_text())["common"]
        else:
            policy = plan["eight_method_policy"]
            if policy["protocol"] != "eight_dependency_sets_v1":
                raise ValueError("declared eight-set fixed policy required")
        if (
            policy["actual_declared_weighted_l2_cap"] != 1.0
            or policy["relative_radius"] != 0.025
            or policy["absolute_radius"] != 1e-6
        ):
            raise ValueError("declared fixed budget required")
        for candidate in plan["candidates"]:
            if candidate["rounding"] != {
                "relative_radius": 0.025,
                "absolute_radius": 1e-6,
                "rounds": 16,
            }:
                raise ValueError("candidate rounding differs from policy")

        out = Path(item["output"])
        rp = out / "report.json"
        if not rp.is_file():
            results.extend(
                {"id": i, "status": "no_report", "eligible_for_finite": False}
                for i in item["ids"]
            )
            continue
        report = json.loads(rp.read_text())
        source_hashes = report["source_sha256"]
        if source_hashes[str(plan_path)] != item["plan_sha256"]:
            raise ValueError("report plan SHA mismatch")
        for source_path in (
            obs / "native-correct-logprob/source-native-values.npz",
            obs / "matrix-correct-logprob/coverage.json",
            obs / "matrix-correct-logprob/sample-plan.npz",
        ):
            expected = source_hashes.get(str(source_path))
            required = (
                plan["schema_version"] == 2
                or source_path.name == "source-native-values.npz"
            )
            if (required or expected is not None) and expected != pins[
                str(source_path)
            ]:
                raise ValueError("observation source differs from proposal report")

        for method in report["methods"]:
            ident = method["id"]
            sp = out / (ident + ".json")
            planned = next(c for c in plan["candidates"] if c["id"] == ident)
            if (
                planned["cost_metric"] != method["cost_metric"]
                or planned["constraints"] != method["constraints"]
            ):
                raise ValueError("report objective or cost differs from pinned plan")
            row = {
                "id": ident,
                "plan": item["plan"],
                "plan_sha256": sha(item["plan"]),
                "report": str(rp),
                "report_sha256": sha(rp),
                "solver_seconds": report["seconds"],
                "chosen_round": method["chosen_round"],
                "finite_measured": False,
                "adopted": False,
            }
            pred = np.asarray(method["prediction"])
            lower = np.asarray(method["minimum_delta"])
            short = np.maximum(lower - pred, 0)
            rows = [
                dict(c, predicted_delta=float(p), shortfall=float(s))
                for c, p, s in zip(method["constraints"], pred, short, strict=True)
            ]
            purposes = {}
            for obj in ("correct-logprob", "margin"):
                purposes[obj] = {
                    p: float(
                        np.mean(
                            [
                                r["predicted_delta"]
                                for r in rows
                                if r["objective"] == obj and r["purpose"] == p
                            ]
                        )
                    )
                    for p in sorted({r["purpose"] for r in rows})
                }
            row.update(
                prediction_rows=rows,
                predicted_purpose_means=purposes,
                predicted_misses=int(np.count_nonzero(short > 1e-10)),
                max_prediction_shortfall=float(short.max()),
                predicted_constraints_satisfied=method[
                    "predicted_constraints_satisfied"
                ],
            )
            if not sp.is_file():
                row.update(status="no_spec", eligible_for_finite=False)
                results.append(row)
                continue
            spec = json.loads(sp.read_text())
            metric = method["cost_metric"]
            scales = metric.get("block_scales", {"native": 1.0, "matrix": 1.0})
            counts = {"native": 0, "matrix": 0}
            family = {}
            raw2 = weighted2 = relative2 = 0.0
            excess = 0
            maxexcess = 0.0
            for edit in spec["edits"]:
                if "indices" in edit:
                    if plan.get("eligible_blocks") == ["matrix"]:
                        raise ValueError("native edit in matrix-only plan")
                    if (
                        plan.get("native_tensors") is not None
                        and edit["tensor"] not in plan["native_tensors"]
                    ):
                        raise ValueError("protected native tensor changed")
                elif plan.get("matrix_families") is not None:
                    match = re.search(
                        r"\.(q|k|v|o|gate|up|down)_proj\.weight$", edit["tensor"]
                    )
                    if match is None or match.group(1) not in plan["matrix_families"]:
                        raise ValueError("ineligible matrix family changed")
                n = edit["tensor"]
                isnative = "indices" in edit
                block = "native" if isnative else "matrix"
                if isnative:
                    base = native[n][np.asarray(edit["indices"], dtype=np.int64)]
                else:
                    base = np.asarray(
                        [matrix[n][tuple(c)] for c in edit["coordinates"]],
                        dtype=np.float64,
                    )
                v = np.asarray(edit["set_values"], dtype=np.float64)
                v32 = v.astype(np.float32)
                if not np.all(np.isfinite(v)) or np.any(
                    v32.view(np.uint32) & np.uint32(65535)
                ):
                    raise ValueError("non BF16 spec value")
                if not np.array_equal(v, v32.astype(np.float64)):
                    raise ValueError("nonexact float32 spec value")
                delta = v - base
                if np.any(delta == 0):
                    raise ValueError("declared unchanged coordinate")
                rel = delta / np.maximum(np.abs(base), 1e-6)
                cost = (
                    np.ones_like(base)
                    if metric["type"] == "euclidean"
                    else 1 / np.maximum(np.abs(base), metric["floor"])
                )
                weighted = delta * cost * scales[block]
                radius = np.maximum(np.abs(base) * 0.025, 1e-6)
                ex = np.maximum(np.abs(delta) - radius, 0)
                excess += int(np.count_nonzero(ex > 0))
                maxexcess = max(maxexcess, float(ex.max(initial=0)))
                r2 = float(delta @ delta)
                w2 = float(weighted @ weighted)
                b2 = float(rel @ rel)
                raw2 += r2
                weighted2 += w2
                relative2 += b2
                counts[block] += len(delta)
                f = (
                    "native"
                    if isnative
                    else re.search(r"\.(q|k|v|o|gate|up|down)_proj\.weight$", n).group(
                        1
                    )
                )
                a = family.setdefault(
                    f,
                    {
                        "changed_elements": 0,
                        "changed_tensors": 0,
                        "raw_l2_squared": 0.0,
                        "declared_weighted_l2_squared": 0.0,
                        "unscaled_relative_l2_squared": 0.0,
                    },
                )
                a["changed_elements"] += len(delta)
                a["changed_tensors"] += 1
                a["raw_l2_squared"] += r2
                a["declared_weighted_l2_squared"] += w2
                a["unscaled_relative_l2_squared"] += b2
            raw = math.sqrt(raw2)
            weighted = math.sqrt(weighted2)
            if (
                sum(counts.values()) != method["changed_elements"]
                or len(spec["edits"]) != method["changed_tensors"]
            ):
                raise ValueError("count differs")
            if not math.isclose(
                raw, method["actual_delta_l2"], rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("raw norm differs")
            if not math.isclose(
                weighted, method["weighted_delta_l2"], rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("weighted norm differs")
            for a in family.values():
                for k in ("raw", "declared_weighted", "unscaled_relative"):
                    a[k + "_l2"] = math.sqrt(a.pop(k + "_l2_squared"))
            row.update(
                status="exact_spec_postchecked",
                spec=str(sp),
                spec_sha256=sha(sp),
                actual_changed=sum(counts.values()),
                changed_tensors=len(spec["edits"]),
                changed_by_block=counts,
                family_support=family,
                raw_delta_l2=raw,
                declared_weighted_delta_l2=weighted,
                unscaled_relative_delta_l2=math.sqrt(relative2),
                metric=metric,
                continuous_box_excess_count=excess,
                maximum_continuous_box_excess=maxexcess,
                eligible_for_finite=weighted <= 1.0 and sum(counts.values()) > 0,
                metric_cap=1.0,
                both_blocks_changed=all(v > 0 for v in counts.values()),
                block_collapse=(
                    plan.get("eligible_blocks") == ["native", "matrix"]
                    and not all(v > 0 for v in counts.values())
                ),
            )
            results.append(row)
    output = args.output
    for p in inputs:
        current = (
            p.stat().st_dev,
            p.stat().st_ino,
            p.stat().st_size,
            p.stat().st_mtime_ns,
            p.stat().st_ctime_ns,
        )
        if current != stamps[str(p)] or sha(p) != pins[str(p)]:
            raise ValueError("numeric postcheck input changed")

    value = {
        "status": "completed",
        "parent": roster["parent"],
        "model_calls": 0,
        "adopted": False,
        "scope": "Independent exact selected-source BF16 counts/norms. 84 predictions are retained from the pinned existing solver; no new solve or model.",
        "limitations": [
            "A block-scaled metric cap is not comparable to unscaled relative cap.",
            "Family masks and eligible-dependent targets do not imply dominance or functional independence.",
            "Continuous-box excess after nearest BF16 is reported, not silently clipped.",
        ],
        "roster_sha256": sha(args.roster),
        "input_sha256": pins,
        "input_stat_unchanged": True,
        "results": results,
    }
    with output.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha(output),
                "results": [
                    {
                        "id": r["id"],
                        "status": r["status"],
                        "weighted_l2": r.get("declared_weighted_delta_l2"),
                        "changed": r.get("actual_changed"),
                        "misses": r.get("predicted_misses"),
                        "eligible": r["eligible_for_finite"],
                    }
                    for r in results
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
