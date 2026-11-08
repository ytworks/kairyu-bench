#!/usr/bin/env python3
"""Freeze one native F03 CPU proposal from measured AV behavior; no solve/model call."""
import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
OUT = ROOT / "analysis/fresh-purpose-20261107-f03-native-v1"
MEASUREMENT = ROOT / "analysis/fresh-purpose-20261107-av-behavior12-native-gpu4-v2"
DEPENDENCY = ROOT / "analysis/fresh-purpose-20261107-native-dependency-v2/report.json"


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert not OUT.exists()
    measured = json.loads((MEASUREMENT / "root-approved-plan.json").read_text())
    dependency_pin = pin(DEPENDENCY)
    assert dependency_pin["sha256"] == "271ee1afef4afd855c3aa6d589891bef6ad2b144b9499df9646039894835fad7"
    report = json.loads(DEPENDENCY.read_text())
    rows = {row["id"]: row for row in report["case_rows"]}
    families = {"post_attention_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar"}
    names = sorted(row["tensor"] for row in report["tensors"] if row["family"] in families)
    assert len(names) == 180
    constraints = []
    for objective in ("correct-logprob", "margin"):
        for purpose in "ABCDEF":
            for fold in range(2):
                short = purpose + str(fold + 1)
                identifier = "behavior-" + short
                assert identifier in rows
                if objective == "correct-logprob":
                    bound = .05 if short in ("A1", "F2") else 0. if short in ("A2", "F1") else -.02
                else:
                    bound = .02 - rows[identifier]["margin"] if short == "A1" else -.10 if short == "F2" else -.05
                constraints.append(dict(objective=objective, purpose=purpose, fold=fold, minimum_delta=bound))
    OUT.mkdir()
    parent = dict(checkpoint_tree_sha256=report["checkpoint_tree_sha256"],
        native_measurement_closure=pin(MEASUREMENT / "measurement-closure.json"),
        native_behavior_result=measured["native_behavior_result"], dependency=dependency_pin,
        comparison_status="AV retained: F02 complete matched WSP10 ties code87.777778 with all26 code outcomes unchanged; AV N30 code72.936508 is a separate cohort.",
        hypothesis="Jointly adjust all60 layers' post-attention/post-FFW direct RMSNorm scales plus whole-layer scalar. These three structural controls are distinct; raw-unit gradient energy is descriptive, not ability attribution.",
        restrictions=["No transfer generation/gradient used", "No benchmark answer used as training target", "One fixed proposal; no post-outcome rescaling", "All24 constraints and existing full WSP cohorts retained", "Local predicted gain is not native generation or WSP gain"])
    save(OUT / "parent-state.json", parent)
    plan = dict(schema_version=1, kind="native_coupled_objectives_v1",
        checkpoint=dict(path=measured["checkpoint"], tree_sha256=report["checkpoint_tree_sha256"], attestation=measured["attestation"]),
        cases=measured["cases"], parent_state=pin(OUT / "parent-state.json"), native_tensors=names,
        collections={objective: str(MEASUREMENT / "work" / objective) for objective in ("correct-logprob", "margin")},
        candidates=[dict(id="fresh-native-contract-f03", constraints=constraints, cost_metric=dict(type="euclidean"),
            rounding=dict(relative_radius=.01, absolute_radius=1e-6, rounds=16))])
    save(OUT / "proposal-plan.json", plan)
    save(OUT / "preparation-receipt.json", dict(source=pin(Path(__file__).resolve()), plan=pin(OUT / "proposal-plan.json"),
        eligible_tensors=len(names), eligible_coordinates=645180, model_calls=0, solver_calls=0, weight_edits=0,
        limits=["Euclidean continuous relaxation, not globally optimal discrete BF16", "Continuous clip box only; rounded actual values may exceed1%", "16 is a maximum; early stop allowed", "Each history round records max shortfall; all24 residuals are available for chosen proposal only"] ))
    print(json.dumps(dict(plan=pin(OUT / "proposal-plan.json"), solver_calls=0, model_calls=0)))


if __name__ == "__main__":
    main()
