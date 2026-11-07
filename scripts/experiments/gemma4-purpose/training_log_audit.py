"""Read-only audit of complete Gemma4 LoRA training logs, never a capability score."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def distribution(values):
    return {
        "n": len(values),
        "minimum": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "maximum": max(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.study / "work/train-lora"
    if not (work / "summary.json").is_file():
        print(json.dumps({"status": "pending", "reason": "trainer summary absent; no analysis performed"}))
        return 75
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = json.loads((args.study / "trainer-plan.json").read_text())
    summary = json.loads((work / "summary.json").read_text())
    order_record = json.loads((work / "training-order.json").read_text())
    steps = [json.loads(line) for line in (work / "steps.jsonl").read_text().splitlines()]
    micro = [json.loads(line) for line in (work / "microbatches.jsonl").read_text().splitlines()]
    evidence_files = [args.study / "trainer-plan.json"]
    evidence_files += [p for p in work.rglob("*.json") if p.is_file()]
    evidence_files += [work / "steps.jsonl", work / "microbatches.jsonl"]
    pins = {str(p): sha(p) for p in sorted(evidence_files)}
    save(args.output / "input-pins.json", pins)
    complete = summary["status"] == "completed"
    if not complete:
        save(args.output / "summary.json", {
            "status": "partial", "trainer_status": summary["status"],
            "completed_steps": len(steps), "presentations": len(micro),
            "trainer_summary": summary, "capability_claim": False,
            "analysis_deferred": True, "input_sha256": pins,
        })
        return 75

    assert len(steps) == summary["completed_steps"] == 64
    assert len(micro) == summary["model_forwards"] == summary["forward_attempts"] == 256
    assert summary["base_unchanged"] and not summary["holdout_gradient_ids"]
    assert summary["error"] is None
    training_ids = plan["data"]["ids"]
    assert len(training_ids) == len(set(training_ids)) == 204
    expected_order = list(training_ids)
    random.Random(0).shuffle(expected_order)
    assert order_record["ids"] == expected_order and order_record["seed"] == 0
    assert [r["id"] for r in micro] == (expected_order * 2)[:256]
    assert [r["position"] for r in micro] == list(range(1, 257))
    assert [r["step"] for r in steps] == list(range(1, 65))
    assert all(r["backward_completed"] and math.isfinite(r["CE"]) for r in micro)
    by_id = defaultdict(list)
    for row in micro:
        by_id[row["id"]].append(row)
    assert set(by_id) == set(training_ids)
    assert Counter(len(v) for v in by_id.values()) == {1: 152, 2: 52}
    repeated, coverage = [], []
    for identifier in training_ids:
        rows = by_id[identifier]
        coverage.append({
            "id": identifier, "presentations": len(rows), "positions": [r["position"] for r in rows],
            "steps": [r["step"] for r in rows], "CE": [r["CE"] for r in rows],
            "target_tokens": [r["target_tokens"] for r in rows],
        })
        if len(rows) == 2:
            first, second = rows
            assert first["position"] <= 52 and second["position"] == first["position"] + 204
            assert first["target_tokens"] == second["target_tokens"]
            repeated.append({
                "id": identifier, "first_position": first["position"], "second_position": second["position"],
                "first_step": first["step"], "second_step": second["step"],
                "target_tokens": first["target_tokens"], "first_CE": first["CE"], "second_CE": second["CE"],
                "delta_CE_second_minus_first": second["CE"] - first["CE"],
            })
    repeated.sort(key=lambda r: r["first_position"])
    names = set(steps[0]["modules"])
    assert len(names) == 410
    projected_steps, observed, lifetime = [], {}, {}
    for name in sorted(names):
        match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(mlp|self_attn)\.(\w+)_proj\.weight", name)
        assert match, name
        lifetime[name] = {
            "layer": int(match[1]), "family": match[2] + "." + match[3] + "_proj",
            "base_full_matrix_frobenius": steps[0]["modules"][name]["base_frobenius"],
            "projection_count": 0, "projection_steps": [], "zero_A_gradient_steps": [], "zero_B_gradient_steps": [],
            "maximum_relative_before": 0.0, "maximum_relative_after": 0.0,
            "maximum_delta_after": 0.0,
        }
    for step in steps:
        index = step["step"]
        cases = micro[(index - 1) * 4:index * 4]
        assert step["case_ids"] == [r["id"] for r in cases]
        assert step["example_CE"] == [r["CE"] for r in cases]
        assert step["answer_EOS_tokens"] == sum(r["target_tokens"] for r in cases)
        assert math.isclose(step["mean_CE"], statistics.mean(r["CE"] for r in cases), abs_tol=1e-12)
        assert set(step["modules"]) == names
        selected = {}
        for name, row in step["modules"].items():
            record = lifetime[name]
            assert row["calls"] == 4 and row["base_frobenius"] == record["base_full_matrix_frobenius"]
            assert all(math.isfinite(row[k]) for k in ("A_grad_l2", "B_grad_l2", "relative_after", "relative_before", "delta_l2_after_projection", "B_multiplier"))
            assert row["A_zero_grad"] == (row["A_grad_l2"] == 0.0)
            assert row["B_zero_grad"] == (row["B_grad_l2"] == 0.0)
            if row["A_zero_grad"]:
                record["zero_A_gradient_steps"].append(index)
            if row["B_zero_grad"]:
                record["zero_B_gradient_steps"].append(index)
            if row["B_multiplier"] < 1:
                record["projection_count"] += 1
                record["projection_steps"].append(index)
            record["maximum_relative_before"] = max(record["maximum_relative_before"], row["relative_before"])
            record["maximum_relative_after"] = max(record["maximum_relative_after"], row["relative_after"])
            record["maximum_delta_after"] = max(record["maximum_delta_after"], row["delta_l2_after_projection"])
            if index == 64:
                record["final"] = dict(row)
            if index in (1, 4, 64):
                assert set(row["activations"]) == {"input", "output"}
                selected[name] = dict(row)
                selected[name]["derived_activation_statistics"] = {}
                for kind, values in row["activations"].items():
                    n = values["elements"]
                    assert n > 0 and values["sum_squares"] >= 0
                    selected[name]["derived_activation_statistics"][kind] = {
                        "mean": values["sum"] / n,
                        "rms": math.sqrt(values["sum_squares"] / n),
                        "elements": n, "max_abs": values["max_abs"],
                    }
            else:
                assert "activations" not in row
        if selected:
            observed[str(index)] = {"case_ids": step["case_ids"], "modules": selected}
        rows = list(step["modules"].values())
        denominator = math.sqrt(sum(r["base_frobenius"] ** 2 for r in rows))
        delta = math.sqrt(sum(r["delta_l2_after_projection"] ** 2 for r in rows))
        reconstructed_grad = math.sqrt(sum(r["A_grad_l2"] ** 2 + r["B_grad_l2"] ** 2 for r in rows))
        projected_steps.append({
            "step": index, "case_ids": step["case_ids"], "mean_CE": step["mean_CE"],
            "target_tokens": step["answer_EOS_tokens"], "seconds": step["seconds"],
            "preclip_global_factor_gradient_l2": step["grad_l2_before_clip"],
            "preclip_reconstructed_from_module_norms": reconstructed_grad,
            "gradient_norm_reconstruction_difference": reconstructed_grad - step["grad_l2_before_clip"],
            "global_clip_active": step["grad_l2_before_clip"] > 1,
            "projection_modules": sum(r["B_multiplier"] < 1 for r in rows),
            "strict_relative_after_over_0_005": sum(r["relative_after"] > 0.005 for r in rows),
            "relative_after_over_implementation_tolerance": sum(r["relative_after"] > 0.005 * (1 + 1e-7) for r in rows),
            "max_relative_before": max(r["relative_before"] for r in rows),
            "max_relative_after": max(r["relative_after"] for r in rows),
            "all410_continuous_delta_l2": delta,
            "all410_complete_matrix_denominator_l2": denominator,
            "all410_relative_l2": delta / denominator,
            "zero_A_gradient_modules": sum(r["A_zero_grad"] for r in rows),
            "zero_B_gradient_modules": sum(r["B_zero_grad"] for r in rows),
        })
    save(args.output / "all204-coverage.json", coverage)
    save(args.output / "all52-repeated-case-CE.json", repeated)
    save(args.output / "all410-step1-4-64.json", observed)
    save(args.output / "all410-lifetime.json", lifetime)
    save(args.output / "all64-step-summary.json", projected_steps)
    columns = list(repeated[0])
    with (args.output / "all52-repeated-case-CE.tsv").open("x") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in repeated:
            stream.write("\t".join(str(row[key]) for key in columns) + "\n")
    factors = []
    for phase in ("pilot", "final"):
        directory = work / phase
        receipt = json.loads((directory / "training-receipt.json").read_text())
        payload = directory / "adapter_model.safetensors"
        stat = payload.stat()
        factors.append({
            "phase": phase, "path": str(payload), "bytes": stat.st_size,
            "sha256_from_export_receipt": receipt["adapter_sha256"],
            "receipt_sha256": sha(directory / "training-receipt.json"),
            "stat": {k: getattr(stat, "st_" + k) for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns", "nlink")},
            "payload_rehashed_here": False, "payload_copied_to_PR": False,
        })
    save(args.output / "factor-identities.json", factors)
    deltas = [r["delta_CE_second_minus_first"] for r in repeated]
    tokens = sum(r["target_tokens"] for r in repeated)
    result = {
        "status": "completed", "trainer_status": summary["status"], "completed_steps": 64,
        "training_ids": 204, "presentations": 256, "repeated_cases": 52,
        "all_original_ids_covered": True, "holdout_gradient_ids": [],
        "same_case_CE": {
            "first": distribution([r["first_CE"] for r in repeated]),
            "second": distribution([r["second_CE"] for r in repeated]),
            "delta_second_minus_first": distribution(deltas),
            "lower": sum(x < 0 for x in deltas), "equal": sum(x == 0 for x in deltas), "higher": sum(x > 0 for x in deltas),
            "token_weighted_first": sum(r["first_CE"] * r["target_tokens"] for r in repeated) / tokens,
            "token_weighted_second": sum(r["second_CE"] * r["target_tokens"] for r in repeated) / tokens,
        },
        "modules": 410, "observed_activation_steps": [1, 4, 64],
        "total_module_projection_events": sum(r["projection_count"] for r in lifetime.values()),
        "modules_projected_at_least_once": sum(r["projection_count"] > 0 for r in lifetime.values()),
        "global_gradient_clip_steps": sum(r["global_clip_active"] for r in projected_steps),
        "strict_radius_excess_events": sum(r["strict_relative_after_over_0_005"] for r in projected_steps),
        "implementation_tolerance_radius_excess_events": sum(r["relative_after_over_implementation_tolerance"] for r in projected_steps),
        "selected_step_summary": [r for r in projected_steps if r["step"] in (1, 4, 64)],
        "sum_logged_training_step_seconds": sum(r["seconds"] for r in steps),
        "trainer_elapsed_including_load_export_and_pilot_wait": summary["elapsed_seconds"],
        "base_unchanged": summary["base_unchanged"], "capability_claim": False,
        "additional_model_forwards": 0, "GPU_service_operations": 0,
        "activation_limit": "Each module's saved input/output aggregate spans four full-sequence calls in one step; different steps use different cases. No per-case/token/rank attribution or cross-step causal claim.",
        "gradient_limit": "A/B factor-coordinate accumulated preclip gradients, not dense-weight derivatives. Continuous F32 BA norms precede any BF16 materialization.",
        "CE_limit": "Teacher-forced CE on the same 52 training examples during optimization; no independent capability, generation, holdout or WSP conclusion.",
        "source_sha256": sha(Path(__file__)), "analysis_seconds": time.monotonic() - started,
    }
    assert all(sha(Path(path)) == digest for path, digest in pins.items())
    save(args.output / "summary.json", result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
