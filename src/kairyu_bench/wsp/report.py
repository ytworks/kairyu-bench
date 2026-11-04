"""Scores per source, area and overall; paired comparison with a base run.

Area score = mean of its sources' scores; overall = mean of the area scores
(every area weighs the same). A comparison resamples items inside each
source (paired, stratified bootstrap with a fixed seed), so its intervals
are reproducible and reflect both item and sampling noise.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kairyu_bench.artifacts import read_object
from kairyu_bench.wsp.catalog import AREAS, SOURCE_BY_NAME

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261104
AREA_LABELS = {area.key: area.label for area in AREAS}


def read_items(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "items.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return sum(collected) / len(collected) if collected else None


def _samples(record: dict[str, Any]) -> list[dict[str, Any]]:
    outcome = record.get("outcome") or {}
    samples = outcome.get("samples")
    return samples if isinstance(samples, list) else []


def composite(source_scores: dict[str, float]) -> tuple[dict[str, float], float | None]:
    by_area: dict[str, list[float]] = defaultdict(list)
    for source, value in source_scores.items():
        area = SOURCE_BY_NAME[source].area
        if area is not None:
            by_area[area].append(value)
    areas = {area: sum(values) / len(values) for area, values in by_area.items()}
    return areas, _mean(areas.values())


def build_report(run_dir: Path) -> dict[str, Any]:
    run = read_object(run_dir / "run.json", "wsp run")
    records = read_items(run_dir)
    sources: dict[str, dict[str, Any]] = {}
    for name in sorted({record["source"] for record in records}):
        rows = [record for record in records if record["source"] == name]
        scored = [record for record in rows if record.get("score") is not None]
        samples = [sample for record in rows for sample in _samples(record)]
        chat = [sample for sample in samples if "finish_reason" in sample]
        # Multi-call samples (SciCode steps, BFCL turns) judge emptiness per call.
        replies = [call for sample in chat for call in sample.get("calls") or [sample]]
        sources[name] = {
            "area": SOURCE_BY_NAME[name].area,
            "items": len(rows),
            "scored": len(scored),
            "score": _mean(record["score"] for record in scored),
            "truncated_rate": _mean(float(s["finish_reason"] == "length") for s in chat),
            "empty_rate": _mean(float(not (s.get("content") or s.get("tool_calls"))) for s in replies),
            "completion_tokens_mean": _mean(float(s.get("completion_tokens", 0)) for s in chat),
            "completion_tokens": sum(int(s.get("completion_tokens", 0)) for s in chat),
            "prompt_tokens": sum(int(s.get("prompt_tokens", 0)) for s in chat),
        }
    capability = {
        name: entry["score"]
        for name, entry in sources.items()
        if entry["area"] is not None and entry["score"] is not None
    }
    areas, overall = composite(capability)
    return {
        "schema_version": 1,
        "run": {key: run.get(key) for key in ("tier", "split", "label", "model_id", "status", "manifest_sha256")},
        "timing": run.get("timing"),
        "unscored": run.get("unscored", []),
        "overall": overall,
        "areas": {area: areas.get(area) for area in AREA_LABELS},
        "sources": sources,
        "calibration_ece": _calibration_error(records),
    }


def _calibration_error(records: list[dict[str, Any]], bins: int = 10) -> float | None:
    """Expected calibration error of choice probabilities from logprob items."""

    pairs: list[tuple[float, float]] = []
    for record in records:
        detail = record.get("detail") or {}
        confidence = detail.get("confidence")
        correct = detail.get("confidence_correct")
        if isinstance(confidence, (int, float)) and isinstance(correct, (int, float)):
            pairs.append((float(confidence), float(correct)))
    if not pairs:
        return None
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [pair for pair in pairs if low <= pair[0] < high or (index == bins - 1 and pair[0] == 1.0)]
        if members:
            confidence = sum(pair[0] for pair in members) / len(members)
            accuracy = sum(pair[1] for pair in members) / len(members)
            total += len(members) / len(pairs) * abs(confidence - accuracy)
    return total


def _fmt(value: float | None, scale: float = 100.0, digits: int = 1) -> str:
    return "–" if value is None else f"{value * scale:.{digits}f}"


def report_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    timing = report.get("timing") or {}
    lines = [
        f"# WSP tier {run['tier']} ({run['split']}) — {run.get('label') or run['model_id']}",
        "",
        f"Overall: **{_fmt(report['overall'])}** (mean of the area scores, 0–100). "
        f"Status: {run['status']}; wall {timing.get('wall_seconds')} s "
        f"(target {timing.get('target_minutes')} min); unscored items: {len(report['unscored'])}.",
        "",
        "| Area | 能力 | Score |",
        "|---|---|---:|",
    ]
    lines += [f"| {key} | {AREA_LABELS[key]} | {_fmt(value)} |" for key, value in report["areas"].items()]
    lines += [
        "",
        "| Source | Area | Items | Score | Truncated % | Empty % | Mean output tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in report["sources"].items():
        score = entry["score"]
        shown = _fmt(score) if entry["area"] is not None else ("–" if score is None else f"{score:.3f} nats")
        lines.append(
            f"| {name} | {entry['area'] or 'diagnostic'} | {entry['scored']}/{entry['items']} | {shown} | "
            f"{_fmt(entry['truncated_rate'])} | {_fmt(entry['empty_rate'])} | "
            f"{_fmt(entry['completion_tokens_mean'], 1.0, 0)} |"
        )
    if report.get("calibration_ece") is not None:
        lines += ["", f"Calibration error (choice probabilities): {report['calibration_ece']:.3f}"]
    return "\n".join(lines) + "\n"


def _paired(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    by_id = {record["id"]: record for record in candidate}
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in base:
        other = by_id.get(record["id"])
        if (
            other is not None
            and SOURCE_BY_NAME[record["source"]].area is not None
            and record.get("score") is not None
            and other.get("score") is not None
        ):
            pairs[record["source"]].append((float(record["score"]), float(other["score"])))
    return pairs


def _interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    low = ordered[int(math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[int(math.ceil(0.975 * (len(ordered) - 1)))]
    return low, high


def _probe_divergence(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    """Next-token distribution change on the probe texts (top-k approximation).

    KL(base || edited) is summed over the base model's top-k tokens; a token
    missing from the edited top-k takes the edited top-k's lowest logprob, so
    the value is a lower bound.
    """

    by_id = {record["id"]: record for record in candidate}
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in base:
        if record["kind"] != "logprob" or SOURCE_BY_NAME[record["source"]].area is not None:
            continue
        other = by_id.get(record["id"])
        if other is None or not record.get("outcome") or not other.get("outcome"):
            continue
        left = record["outcome"]["positions"]
        right = other["outcome"]["positions"]
        if [entry["token"] for entry in left] != [entry["token"] for entry in right]:
            raise ValueError(f"{record['id']}: probe tokens differ between runs")
        bucket = totals[record["source"]]
        for p_entry, q_entry in zip(left, right, strict=True):
            p = {int(token): value for token, value in p_entry["top"]}
            q = {int(token): value for token, value in q_entry["top"]}
            floor = min(q.values())
            bucket["kl"] += sum(math.exp(lp) * (lp - q.get(token, floor)) for token, lp in p.items())
            bucket["top1"] += float(max(p, key=p.__getitem__) == max(q, key=q.__getitem__))
            bucket["nll_base"] -= p_entry["logprob"]
            bucket["nll_candidate"] -= q_entry["logprob"]
            bucket["positions"] += 1
    return {
        source: {
            "kl_per_token": values["kl"] / values["positions"],
            "top1_agreement": values["top1"] / values["positions"],
            "nll_base": values["nll_base"] / values["positions"],
            "nll_candidate": values["nll_candidate"] / values["positions"],
            "positions": int(values["positions"]),
        }
        for source, values in sorted(totals.items())
        if values["positions"]
    }


def compare(base_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    base_run = read_object(base_dir / "run.json", "wsp run")
    candidate_run = read_object(candidate_dir / "run.json", "wsp run")
    for key in ("manifest_sha256", "tier", "split"):
        if base_run.get(key) != candidate_run.get(key):
            raise ValueError(f"runs differ in {key}: {base_run.get(key)} vs {candidate_run.get(key)}")
    base, candidate = read_items(base_dir), read_items(candidate_dir)
    pairs = _paired(base, candidate)

    def summary(chosen: dict[str, list[tuple[float, float]]]) -> tuple[dict[str, float], dict[str, float], float]:
        base_areas, base_overall = composite({s: sum(b for b, _ in v) / len(v) for s, v in chosen.items() if v})
        cand_areas, cand_overall = composite({s: sum(c for _, c in v) / len(v) for s, v in chosen.items() if v})
        deltas = {area: cand_areas[area] - base_areas[area] for area in base_areas}
        sources = {s: sum(c - b for b, c in v) / len(v) for s, v in chosen.items() if v}
        return deltas, sources, (cand_overall or 0.0) - (base_overall or 0.0)

    area_delta, source_delta, overall_delta = summary(pairs)
    generator = random.Random(BOOTSTRAP_SEED)
    area_draws: dict[str, list[float]] = defaultdict(list)
    source_draws: dict[str, list[float]] = defaultdict(list)
    overall_draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resampled = {source: [values[generator.randrange(len(values))] for _ in values] for source, values in sorted(pairs.items())}
        deltas, sources, overall = summary(resampled)
        for key, value in deltas.items():
            area_draws[key].append(value)
        for key, value in sources.items():
            source_draws[key].append(value)
        overall_draws.append(overall)

    base_scores = {s: sum(b for b, _ in v) / len(v) for s, v in pairs.items() if v}
    cand_scores = {s: sum(c for _, c in v) / len(v) for s, v in pairs.items() if v}
    base_areas, base_overall = composite(base_scores)
    cand_areas, cand_overall = composite(cand_scores)
    flips: dict[str, dict[str, int]] = defaultdict(lambda: {"lost": 0, "gained": 0, "items": 0})
    for source, values in pairs.items():
        area = SOURCE_BY_NAME[source].area or ""
        for b, c in values:
            flips[area]["items"] += 1
            flips[area]["lost"] += int(b >= 0.5 > c)
            flips[area]["gained"] += int(c >= 0.5 > b)
    return {
        "schema_version": 1,
        "tier": base_run["tier"],
        "split": base_run["split"],
        "base": {"label": base_run.get("label"), "model_id": base_run.get("model_id"), "overall": base_overall},
        "candidate": {"label": candidate_run.get("label"), "model_id": candidate_run.get("model_id"), "overall": cand_overall},
        "overall": {"delta": overall_delta, "ci95": _interval(overall_draws)},
        "areas": {
            area: {
                "base": base_areas[area],
                "candidate": cand_areas[area],
                "delta": area_delta[area],
                "ci95": _interval(area_draws[area]),
                "flips": flips[area],
            }
            for area in AREA_LABELS
            if area in area_delta
        },
        "sources": {
            source: {
                "base": base_scores[source],
                "candidate": cand_scores[source],
                "delta": source_delta[source],
                "ci95": _interval(source_draws[source]),
                "items": len(pairs[source]),
            }
            for source in sorted(source_delta)
        },
        "probes": _probe_divergence(base, candidate),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }


def comparison_markdown(comparison: dict[str, Any]) -> str:
    def pp(value: float) -> str:
        return f"{value * 100:+.1f}"

    overall = comparison["overall"]
    lines = [
        f"# WSP tier {comparison['tier']} ({comparison['split']}): "
        f"{comparison['candidate']['label'] or comparison['candidate']['model_id']} vs "
        f"{comparison['base']['label'] or comparison['base']['model_id']}",
        "",
        f"Overall: {_fmt(comparison['base']['overall'])} → {_fmt(comparison['candidate']['overall'])} "
        f"({pp(overall['delta'])} pt, 95% CI {pp(overall['ci95'][0])} to {pp(overall['ci95'][1])})",
        "",
        "| Area | 能力 | Base | Edited | Δ pt | 95% CI | Lost | Gained | Items |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for area, entry in comparison["areas"].items():
        low, high = entry["ci95"]
        lines.append(
            f"| {area} | {AREA_LABELS[area]} | {_fmt(entry['base'])} | {_fmt(entry['candidate'])} | "
            f"{pp(entry['delta'])} | {pp(low)} to {pp(high)} | {entry['flips']['lost']} | "
            f"{entry['flips']['gained']} | {entry['flips']['items']} |"
        )
    if comparison["probes"]:
        lines += [
            "",
            "| Probe | KL/token (lower bound) | Top-1 agreement | NLL base → edited | Positions |",
            "|---|---:|---:|---|---:|",
        ]
        for source, entry in comparison["probes"].items():
            lines.append(
                f"| {source} | {entry['kl_per_token']:.4f} | {entry['top1_agreement'] * 100:.1f} % | "
                f"{entry['nll_base']:.3f} → {entry['nll_candidate']:.3f} | {entry['positions']} |"
            )
    return "\n".join(lines) + "\n"
