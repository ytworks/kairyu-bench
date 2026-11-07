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


def observation_sources(plan, obs):
    matrix_only = plan.get("observed_blocks") == ["matrix"]
    if "observed_blocks" in plan and (plan.get("schema_version") != 2
            or plan["observed_blocks"] not in (["native", "matrix"], ["matrix"])):
        raise ValueError("unsupported observed blocks")
    if matrix_only and (plan.get("eligible_blocks") != ["matrix"]
            or "native_tensors" in plan or set(plan["collections"]) != {"matrix"}):
        raise ValueError("matrix-only observations require frozen native")
    paths = [obs / "matrix-correct-logprob/coverage.json",
             obs / "matrix-correct-logprob/sample-plan.npz"]
    if not matrix_only:
        paths.insert(0, obs / "native-correct-logprob/source-native-values.npz")
    return paths


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
        obs / "matrix-correct-logprob/coverage.json",
        obs / "matrix-correct-logprob/sample-plan.npz",
    }
    for item in roster["rows"]:
        inputs.add(Path(item["plan"]).resolve())
        planned_input = json.loads(Path(item["plan"]).read_text())
        inputs.update(observation_sources(planned_input, obs))
        for key in ("analytic_report", "reference_plan"):
            if key in planned_input:
                ref = planned_input[key]
                path = Path(ref["path"]).resolve()
                if sha(path) != ref["sha256"]:
                    raise ValueError("analytic input pin differs")
                inputs.add(path)
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

    native = {}
    native_source = obs / "native-correct-logprob/source-native-values.npz"
    if native_source in inputs:
        with np.load(native_source, allow_pickle=False) as f:
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
        cap_metric = "declared_weighted"
        functional = False
        if "prospective_method" in plan:
            prospective = plan["prospective_method"]
            policy_path = Path(prospective["path"])
            if pins[str(policy_path.resolve())] != prospective["sha256"]:
                raise ValueError("prospective policy pin differs")
            document = json.loads(policy_path.read_text())
            if document.get("protocol") == "functional_targets_v1":
                functional = True
                cap_metric = "raw"
                cap = document["realized_raw_cap"]
                if (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0
                        or document["parent_state"] != plan.get("parent_state")
                        or document.get("native_frozen") is not True
                        or plan.get("eligible_blocks") != ["matrix"]):
                    raise ValueError("functional policy parent, native scope or raw cap differs")
                for candidate in plan["candidates"]:
                    declared = [m for m in document["methods"] if m["id"] == candidate["id"]]
                    if (len(declared) != 1 or declared[0]["target_mode"] != "functional"
                            or declared[0]["cost_metric"] != candidate["cost_metric"]
                            or declared[0]["matrix_families"] != plan["matrix_families"]):
                        raise ValueError("functional candidate differs from declared method")
                    rounding = declared[0]["rounding"]
                    if "analytic_report" in plan:
                        ref = plan["analytic_report"]
                        report_path = Path(ref["path"]).resolve()
                        if pins[str(report_path)] != ref["sha256"]:
                            raise ValueError("analytic report changed")
                        analytic = json.loads(report_path.read_text())
                        original = plan["reference_plan"]
                        original_path = Path(original["path"]).resolve()
                        if pins[str(original_path)] != original["sha256"]:
                            raise ValueError("analytic reference plan changed")
                        reference = json.loads(original_path.read_text())
                        for key in ("checkpoint", "parent_state", "cases", "collections",
                                    "matrix_families", "matrix_tensors", "prospective_method",
                                    "observed_blocks", "eligible_blocks"):
                            if reference.get(key) != plan.get(key):
                                raise ValueError("analytic reference scope differs")
                        original_method = reference["candidates"][0]
                        for key in ("id", "constraints", "cost_metric"):
                            if candidate[key] != original_method[key]:
                                raise ValueError("analytic target or metric differs")
                        expected_names = [
                            e["tensor"] for e in cov["entries"]
                            if re.search(r"\.(q|k|v|o|gate|up|down)_proj\.weight$",
                                         e["tensor"]).group(1) in plan["matrix_families"]
                            and (plan.get("matrix_tensors") is None
                                 or e["tensor"] in plan["matrix_tensors"])]
                        observed_names = [e["tensor"] for e in analytic["layout"]]
                        case_path = str(Path(plan["cases"]["path"]).resolve())
                        if (analytic["cost_metric"] != candidate["cost_metric"]
                                or observed_names != expected_names
                                or analytic["source_sha256"].get(str(original_path)) != original["sha256"]
                                or analytic["source_sha256"].get(case_path) != plan["cases"]["sha256"]):
                            raise ValueError("analytic metric, matrix mask or reference source differs")
                        rows = [{k: r[k] for k in ("objective", "purpose", "fold", "minimum_delta")}
                                for r in analytic["rows"]]
                        if (analytic["status"] != "analytic_calibration_complete"
                                or analytic["continuous_calls"] != 1
                                or analytic["source_checkpoint"] != plan["checkpoint"]
                                or analytic["parent_state"] != plan["parent_state"]
                                or rows != candidate["constraints"]):
                            raise ValueError("analytic objective binding differs")
                        rounding = dict(relative_radius=analytic["derived_relative_radius"],
                                        absolute_radius=analytic["derived_absolute_radius"], rounds=16)
                    if rounding != candidate["rounding"]:
                        raise ValueError("functional rounding differs from declared or derived box")
                policy = dict(actual_declared_weighted_l2_cap=cap)
            elif "realized_weighted_cap" in document:
                if (document["parent_state"] != plan.get("parent_state")
                        or document.get("native_frozen") is not True
                        or plan.get("eligible_blocks") != ["matrix"]):
                    raise ValueError("shared-code policy parent or native scope differs")
                cap = document["realized_weighted_cap"]
                if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
                    raise ValueError("positive finite prospective cost cap required")
                for candidate in plan["candidates"]:
                    declared = [row for row in document["methods"] if row["id"] == candidate["id"]]
                    if (len(declared) != 1 or declared[0]["rounding"] != candidate["rounding"]
                            or declared[0]["matrix_families"] != plan["matrix_families"]
                            or candidate["cost_metric"] != dict(type="relative",
                                floor=declared[0]["relative_floor"],
                                block_scales=dict(native=1., matrix=1.))):
                        raise ValueError("shared-code candidate differs from prospective policy")
                policy = dict(actual_declared_weighted_l2_cap=cap, relative_radius=.025,
                              absolute_radius=1e-6)
            else:
                policy = document["common"]
                if policy["actual_declared_weighted_l2_cap"] != 1.:
                    raise ValueError("legacy fixed cost cap differs")
        else:
            policy = plan["eight_method_policy"]
            if (policy["protocol"] != "eight_dependency_sets_v1"
                    or policy["actual_declared_weighted_l2_cap"] != 1.):
                raise ValueError("declared eight-set fixed policy required")
        cap = policy["actual_declared_weighted_l2_cap"]
        if not functional:
            if policy["relative_radius"] != 0.025 or policy["absolute_radius"] != 1e-6:
                raise ValueError("declared fixed rounding budget required")
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
        if plan.get("observed_blocks") == ["matrix"] and (
                report.get("observed_blocks") != ["matrix"]
                or report.get("native_parameters") != "frozen_not_observed"
                or any(m["native_scope"]["observed_columns"] != 0
                       or m["column_scope"].get("cross_family_baseline_parity") != "not_measured"
                       for m in report["methods"])):
            raise ValueError("matrix-only report claims native observation or parity")
        if source_hashes[str(plan_path)] != item["plan_sha256"]:
            raise ValueError("report plan SHA mismatch")
        for source_path in observation_sources(plan, obs):
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
            if cov.get("sampling_mode") == "mlp-full-units":
                expected_declaration = dict(schema_version=1, support_plan_sha256=cov["unit_support_sha256"],
                    layer_prefix="model.language_model.layers.", units=cov["unit_support"]["units"])
                if spec.get("matrix_full_units") != expected_declaration or plan.get("observed_blocks") != ["matrix"]:
                    raise ValueError("full-unit specification/support binding differs")
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
                if ("coordinates" in edit and plan.get("matrix_tensors") is not None
                        and edit["tensor"] not in plan["matrix_tensors"]):
                    raise ValueError("ineligible exact matrix tensor changed")
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
                radius = np.maximum(np.abs(base) * planned["rounding"]["relative_radius"],
                                    planned["rounding"]["absolute_radius"])
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
                eligible_for_finite=(raw if cap_metric == "raw" else weighted) <= cap
                                    and sum(counts.values()) > 0,
                metric_cap=cap,
                cap_metric=cap_metric,
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
