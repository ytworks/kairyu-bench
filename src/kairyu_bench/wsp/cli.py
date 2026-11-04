"""``kairyu-bench wsp``: prepare, build the calibration pool, run, freeze, compare."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from kairyu_bench.artifacts import write_json
from kairyu_bench.wsp.catalog import SOURCE_BY_NAME, SPLITS, TIERS
from kairyu_bench.wsp.manifest import WspManifestError, load_manifest, missing_coverage, parse_manifest


def _targets() -> dict[str, Any]:
    raw = resources.files("kairyu_bench").joinpath("data/wsp/targets.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert set(data["sources"]) == set(SOURCE_BY_NAME), "targets.json must name every source"
    return data


def _loader() -> Any:
    from kairyu_bench.wsp.sources import Loader

    def paths(variable: str) -> dict[str, Path]:
        return {name: Path(path) for name, path in json.loads(os.environ.get(variable, "{}")).items()}

    data = os.environ.get("KAIRYU_BENCH_WSP_DATA")
    return Loader(
        sources=paths("KAIRYU_BENCH_WSP_SOURCES"),
        pythons=paths("KAIRYU_BENCH_WSP_PYTHONS"),
        files=paths("KAIRYU_BENCH_WSP_FILES"),
        data_dir=Path(data) if data else None,
    )


def _load_rows(names: Sequence[str]) -> dict[str, dict[str, Any]]:
    from kairyu_bench.wsp.sources import adapters

    registry = adapters()
    loader = _loader()
    return {name: registry[name].load(loader) for name in names}


def command_prepare(args: argparse.Namespace) -> int:
    from kairyu_bench.wsp.sources.agents import prepare as prepare_agents
    from kairyu_bench.wsp.sources.long_context import generate_ruler

    digest = generate_ruler(_loader())
    print(f"ruler: generated data sha256 {digest}", flush=True)
    names = [name for name in SOURCE_BY_NAME if SOURCE_BY_NAME[name].kind != "agent"]
    for name, rows in _load_rows(names).items():
        print(f"{name}: {len(rows)} candidate items", flush=True)
    # Agent tasks of the given manifest (a calibration pool), else of the
    # frozen manifest once it exists.
    frozen = resources.files("kairyu_bench").joinpath("data/wsp/manifest.json")
    if args.manifest is not None or frozen.is_file():
        manifest = load_manifest(args.manifest, require_all_areas=args.manifest is None)
        prepare_agents([item for item in manifest.items if item.kind == "agent"])
    else:
        print("agent tasks: no manifest yet; run `wsp prepare --manifest <pool>` after `wsp pool`", flush=True)
    return 0


def command_pool(args: argparse.Namespace) -> int:
    """Calibration pool: ``factor`` x the 60-minute target per split, uncapped tiers."""

    from kairyu_bench.wsp.select import assign_splits, split_order
    from kairyu_bench.wsp.sources import adapters

    targets = _targets()
    registry = adapters()
    names = list(SOURCE_BY_NAME)
    rows = _load_rows(names)
    limit = int(targets["tiers"]["60"]["max_output_tokens"])

    def group(name: str, item_id: str) -> str:
        function = registry[name].group
        return function(rows[name][item_id]) if function else item_id

    split_of = assign_splits(
        (item_id, registry[name].stratum(row), group(name, item_id))
        for name in names
        for item_id, row in rows[name].items()
    )
    entries: list[dict[str, Any]] = []
    for name in names:
        spec = SOURCE_BY_NAME[name]
        orders = split_order(
            name,
            ((item_id, registry[name].stratum(row)) for item_id, row in rows[name].items()),
            split_of,
            equal=registry[name].equal_strata,
        )
        wanted = int(round(int(targets["sources"][name]["60"]) * args.factor))
        keep = min(wanted, *(len(order) for order in orders.values()))
        for split in SPLITS:
            for position, item_id in enumerate(orders[split][:keep]):
                entries.append(
                    {
                        "id": item_id,
                        "source": name,
                        "split": split,
                        "tier": 60,
                        "max_tokens": limit if spec.kind == "chat" else 1,
                        "samples": spec.samples,
                        "stratum": registry[name].stratum(rows[name][item_id]),
                        "order": position,
                    }
                )
    pool = {"schema_version": 1, "purpose": "calibration pool", "tiers": targets["tiers"], "items": entries}
    parsed = parse_manifest(pool, sha256="pool", require_all_areas=False)
    gaps = {key: areas for key, areas in missing_coverage(parsed.items).items() if key.startswith("60/")}
    if gaps:
        raise WspManifestError(f"calibration pool lacks areas: {gaps}")
    write_json(args.output, pool)
    print(f"pool: {len(entries)} items -> {args.output}")
    return 0


def _endpoint(supplied: str) -> Any:
    from kairyu_bench.target import Endpoint

    return Endpoint.parse(supplied)


def command_run(args: argparse.Namespace) -> int:
    from kairyu_bench.runner import endpoint_fingerprint
    from kairyu_bench.target import TargetClient
    from kairyu_bench.wsp.client import ChatClient, LogprobClient
    from kairyu_bench.wsp.run import RunSettings, execute

    # The frozen manifest covers every area in every tier; any manifest must
    # cover them in the tier and split that runs (calibration pools only
    # hold tier-60 items).
    manifest = load_manifest(args.manifest, require_all_areas=args.manifest is None)
    items = manifest.select(args.tier, args.split)
    gaps = missing_coverage(items).get(f"{args.tier}/{args.split}", [])
    if gaps and not args.incomplete_manifest:
        raise WspManifestError(f"tier {args.tier} {args.split} has no items for: {', '.join(gaps)}")
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - set(SOURCE_BY_NAME)
        if unknown:
            raise SystemExit(f"unknown sources: {sorted(unknown)}")
        items = [item for item in items if item.source in wanted]
    # Task images come from `wsp prepare`; removing them would move their
    # download into the next timed run. Only runs with agent tasks care.
    if any(item.kind == "agent" for item in items) and os.environ.get("KAIRYU_BENCH_CLEAN_TASK_IMAGES") == "1":
        raise SystemExit("wsp run keeps prepared task images: unset KAIRYU_BENCH_CLEAN_TASK_IMAGES=1")
    endpoint = _endpoint(args.endpoint)
    api_key = os.environ.get("KAIRYU_API_KEY")
    model = TargetClient(endpoint, api_key=api_key).discover_chat_model()
    logprob = (
        LogprobClient(tuple(_endpoint(url).base_url for url in args.logprob_endpoint), model=model)
        if args.logprob_endpoint
        else None
    )
    from datetime import datetime, timezone
    import secrets

    run_id = args.run_id or (
        f"wsp-{args.tier}-{args.split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"
    )
    settings = RunSettings(
        run_dir=args.results_dir / run_id,
        tier=args.tier,
        split=args.split,
        chat=ChatClient(endpoint.base_url, model, api_key=api_key, concurrency=args.chat_concurrency),
        logprob=logprob,
        loader=_loader(),
        chat_concurrency=args.chat_concurrency,
        logprob_concurrency=args.logprob_concurrency,
        agent_concurrency=args.agent_concurrency,
        label=args.label,
        endpoint_fingerprint=endpoint_fingerprint(endpoint),
    )
    meta = execute(manifest, items, settings)
    print(f"run: {run_id} status={meta['status']} wall={meta['timing']['wall_seconds']} s")
    print((settings.run_dir / "report.md").read_text(encoding="utf-8"), end="")
    return 0 if meta["status"] == "completed" else 1


def command_freeze(args: argparse.Namespace) -> int:
    from kairyu_bench.wsp.report import read_items
    from kairyu_bench.wsp.select import Measured, assign_tiers

    targets = _targets()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    tier_tokens = {int(key): int(value["max_output_tokens"]) for key, value in targets["tiers"].items()}
    tier_seconds = {int(key): 60 * int(value["minutes"]) for key, value in targets["tiers"].items()}
    measured: dict[str, Measured] = {}
    for run_dir in args.calibration:
        for record in read_items(run_dir):
            samples = (record.get("outcome") or {}).get("samples") or []
            chat = [sample for sample in samples if "finish_reason" in sample]
            if record.get("kind") == "chat":
                finished = bool(chat) and all(sample["finish_reason"] != "length" for sample in chat)
                tokens = max((int(sample["completion_tokens"]) for sample in chat), default=0)
                measured[record["id"]] = Measured(tokens=tokens, finished=finished and not record.get("error"))
            elif record.get("kind") == "agent" and samples and samples[0].get("seconds") is not None:
                measured[record["id"]] = Measured(tokens=int(samples[0]["seconds"]), finished=not record.get("error"))
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in pool["items"]:
        by_source.setdefault(entry["source"], {split: [] for split in SPLITS})[entry["split"]].append(entry)
    items: list[dict[str, Any]] = []
    for source, splits in sorted(by_source.items()):
        kind = SOURCE_BY_NAME[source].kind
        orders = {split: [e["id"] for e in sorted(entries, key=lambda e: e["order"])] for split, entries in splits.items()}
        wanted = {tier: int(targets["sources"][source][str(tier)]) for tier in TIERS}
        # Chat items by the base model's output tokens, agent tasks by their
        # measured seconds; log-likelihood items have no length to judge.
        picked = assign_tiers(
            source,
            orders,
            wanted,
            tier_tokens=tier_seconds if kind == "agent" else tier_tokens,
            margin=float(targets["margin"]),
            measured=measured if kind in {"chat", "agent"} else None,
        )
        lookup = {entry["id"]: entry for split in splits.values() for entry in split}
        for split in SPLITS:
            for item_id, tier in picked[split]:
                entry = lookup[item_id]
                outcome = measured.get(item_id)
                items.append(
                    {
                        "id": item_id,
                        "source": source,
                        "split": split,
                        "tier": tier,
                        "max_tokens": tier_tokens[tier] if kind == "chat" else 1,
                        "samples": entry["samples"],
                        "stratum": entry["stratum"],
                        "expected_tokens": outcome.tokens if outcome else 0,
                    }
                )
        counts = {tier: sum(1 for _, t in picked["dev"] if t <= tier) for tier in TIERS}
        print(f"{source}: per split {counts} (targets {wanted})")
    manifest = {
        "schema_version": 1,
        "reference": targets["reference"],
        "tiers": targets["tiers"],
        "margin": targets["margin"],
        "calibration_runs": [path.name for path in args.calibration],
        "items": items,
    }
    # Refuses (and writes nothing) unless every tier and split covers all
    # fourteen areas, e.g. when calibration left an area without items.
    parse_manifest(manifest, sha256="frozen")
    write_json(args.output, manifest)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    from kairyu_bench.wsp.report import compare, comparison_markdown

    comparison = compare(args.base, args.candidate, partial=args.partial)
    if args.output:
        write_json(args.output, comparison)
    print(comparison_markdown(comparison), end="")
    return 0


def command_report(args: argparse.Namespace) -> int:
    from kairyu_bench.wsp.report import build_report, report_markdown

    report = build_report(args.run_dir)
    write_json(args.run_dir / "report.json", report)
    (args.run_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
    print(report_markdown(report), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kairyu-bench wsp")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download every pinned dataset, source and task image")
    prepare.add_argument("--manifest", type=Path, default=None, help="prepare this manifest's agent tasks")
    pool = commands.add_parser("pool", help="write the calibration pool manifest")
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--factor", type=float, default=1.5)
    run = commands.add_parser("run", help="run one tier and split against an API")
    run.add_argument("endpoint")
    run.add_argument("--tier", type=int, choices=TIERS, required=True)
    run.add_argument("--split", choices=SPLITS, required=True)
    run.add_argument("--manifest", type=Path, default=None, help="default: the frozen manifest")
    run.add_argument(
        "--incomplete-manifest",
        action="store_true",
        help="accept a --manifest that lacks areas (development lists; no overall score)",
    )
    run.add_argument("--only", default=None, help="comma-separated sources (diagnostic runs)")
    run.add_argument("--logprob-endpoint", action="append", default=[], help="vLLM base URL with prompt_logprobs")
    run.add_argument("--label", default=None, help="name of the served weights, e.g. their tree hash")
    run.add_argument("--chat-concurrency", type=int, default=256)
    run.add_argument("--logprob-concurrency", type=int, default=32)
    run.add_argument(
        "--agent-concurrency",
        type=int,
        default=4,
        help="agent tasks running at once in total (each harness's own worker setting is also an upper bound)",
    )
    run.add_argument("--run-id", default=None)
    run.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("KAIRYU_BENCH_RESULTS_DIR", "/work/results")),
    )
    freeze = commands.add_parser("freeze", help="assign tiers from calibration runs")
    freeze.add_argument("--pool", type=Path, required=True)
    freeze.add_argument("--calibration", type=Path, action="append", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="paired comparison of two runs")
    compare.add_argument("base", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, default=None)
    compare.add_argument(
        "--partial",
        action="store_true",
        help="also compare runs with different or unscored items (no overall score unless all areas pair)",
    )
    report = commands.add_parser("report", help="rebuild a run's report")
    report.add_argument("run_dir", type=Path)
    return parser


COMMANDS = {
    "prepare": command_prepare,
    "pool": command_pool,
    "run": command_run,
    "freeze": command_freeze,
    "compare": command_compare,
    "report": command_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (WspManifestError, ValueError, OSError) as error:
        print(f"kairyu-bench wsp: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
