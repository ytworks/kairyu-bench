#!/usr/bin/env python3
"""Read-only manifest/result audit; writes numeric evidence, never runs WSP."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import load_manifest


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_run(path, manifest):
    run, report = read(path / "run.json"), read(path / "report.json")
    items = [json.loads(line) for line in (path / "items.jsonl").read_text().splitlines()]
    selected = {x.id: x for x in manifest.select(run["tier"], run["split"])}
    assert run["status"] == "completed" and run["manifest_sha256"] == manifest.sha256
    assert len(items) == len({x["id"] for x in items}) == run["items"] == len(selected)
    assert {x["id"] for x in items} == selected.keys()
    assert not run["unscored"] and not report["missing_areas"]
    assert all(x["score"] is not None and x["error"] is None for x in items)
    assert all((x["max_tokens"], x["samples"], x["source"]) ==
               (selected[x["id"]].max_tokens, selected[x["id"]].samples, selected[x["id"]].source)
               for x in items)
    sources = {}
    for name in sorted({x["source"] for x in items}):
        rows = [x for x in items if x["source"] == name]
        value = sum(x["score"] for x in rows) / len(rows)
        assert abs(value - report["sources"][name]["score"]) < 1e-10
        sources[name] = dict(items=len(rows), scored=len(rows), score=value,
                             area=SOURCE_BY_NAME[name].area)
    code = {k: v for k, v in sources.items() if v["area"] == "code"}
    code_mean = sum(x["score"] for x in code.values()) / len(code)
    assert abs(code_mean - report["areas"]["code"]) < 1e-12
    return dict(run_id=path.name, tier=run["tier"], split=run["split"], label=run["label"],
                status=run["status"], items=len(items), scored=len(items), unscored=0,
                canonical_selection_and_budgets_verified=True, code_source_mean=code_mean,
                overall=report["overall"], areas=report["areas"], sources=sources,
                timing=run["timing"], started_at=run["started_at"], finished_at=run["finished_at"],
                endpoint=run["endpoint"], code_items=[{k: x[k] for k in
                ("id", "source", "score", "sample_scores", "max_tokens", "samples")}
                for x in items if x["area"] == "code"],
                inputs={str(path / name): sha(path / name)
                        for name in ("run.json", "report.json", "items.jsonl")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--legacy-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite an earlier audit."
    manifest = load_manifest()
    selected = {}
    for tier in (10, 30, 60):
        selected[str(tier)] = {}
        for split in ("dev", "test"):
            rows = manifest.select(tier, split)
            selected[str(tier)][split] = dict(
                items=len(rows), sources=dict(sorted(Counter(x.source for x in rows).items())),
                areas=dict(sorted(Counter(SOURCE_BY_NAME[x.source].area or "diagnostic"
                                          for x in rows).items())),
                code_ids=[x.id for x in rows if SOURCE_BY_NAME[x.source].area == "code"])
    baseline = [audit_run(args.legacy_results / f"wsp-base-{tier}-{split}-1-20261104", manifest)
                for tier in (10, 30, 60) for split in ("dev", "test")]
    repeat_path = args.repository / "docs/experiments/evidence-2026-11-07-f02/av-repeat-audit.json"
    repeats = read(repeat_path)
    av_runs = [audit_run(args.project / "results" / x["run_id"], manifest) for x in repeats["runs"]]
    index_path = args.repository / "docs/experiments/evidence-2026-11-07-fresh-audit/runs.json"
    index = read(index_path)
    assert len(index) == 151
    historical_av = [dict(run_id=x["run_id"], tier=x["tier"], split=x["split"],
                         items=x["declared_items"], complete_canonical=x["complete_canonical_run"])
                     for x in index if x["label"] == repeats["checkpoint_tree"]]
    assert not any(x["tier"] in (30, 60) for x in historical_av)
    original10 = next(x for x in baseline if (x["tier"], x["split"]) == (10, "dev"))
    ratios = [x["timing"]["wall_seconds"] / original10["timing"]["wall_seconds"] for x in av_runs]
    estimates = {}
    for tier in (30, 60):
        old = next(x for x in baseline if (x["tier"], x["split"]) == (tier, "dev"))
        estimates[str(tier)] = dict(
            formula="historical_original_DP4_full_tier_wall * AV_DP1_dev10_wall / original_DP4_dev10_wall",
            minutes=[old["timing"]["wall_seconds"] * ratio / 60 for ratio in ratios],
            measured=False,
            limitations="Different models, capture, source/runner and concurrent workload; new source/agent mix is not calibrated.")
    source_paths = ["src/kairyu_bench/data/wsp/manifest.json", "src/kairyu_bench/wsp/manifest.py",
                    "src/kairyu_bench/wsp/cli.py", "src/kairyu_bench/wsp/report.py",
                    "src/kairyu_bench/wsp/run.py", "src/kairyu_bench/wsp/sources/code.py",
                    "src/kairyu_bench/wsp/scorers/scicode_score.py"]
    result = dict(
        schema_version=1, audit_kind="read_only_no_new_benchmark", created_at=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=manifest.sha256, selected=selected, original_full_runs=baseline, av_full_runs=av_runs,
        historical_av_runs=historical_av, historical_index_count=151,
        av_completed_canonical_dev30_count=0, av_completed_canonical_dev60_count=0,
        av_exact_current_condition_prior_repeat_count=0, single_replica_runtime_scenarios=estimates,
        cli_default_concurrency=dict(chat=256, logprob=32, agent=4),
        original_runtime_evidence_limit="Raw run records omit concurrency/topology/OCI hashes; DP4/MTP4 is contemporaneous documentation, not an exact runtime attestation.",
        scicode=dict(item_score="mean over samples of passed tested steps / all tested steps",
                     source_score="unweighted mean of item scores", code_area="unweighted mean of source scores",
                     any_step_truncated="entire sample scores zero",
                     author_steps_zero_based=[["13", 5], ["62", 0], ["76", 2]],
                     step_timeout_seconds=300, timeout_is_failure=True, environment_failure_raises=True),
        input_sha256={str(index_path): sha(index_path), str(repeat_path): sha(repeat_path)},
        source_sha256={name: sha(args.repository / name) for name in source_paths},
        helper_sha256=sha(Path(__file__)),
        preparatory_failures=["Direct CLI module help attempted /work creation and failed without a benchmark run.",
                              "First ad-hoc manifest area sort failed on diagnostic None; corrected by explicit diagnostic category."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), manifest=manifest.sha256,
                         original_dev=[dict(tier=x["tier"], items=x["items"], code=x["code_source_mean"],
                                            minutes=x["timing"]["wall_seconds"]/60) for x in baseline if x["split"]=="dev"],
                         av_runs=len(av_runs), estimates=estimates), ensure_ascii=False))


if __name__ == "__main__":
    main()
