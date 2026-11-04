"""Agent tasks: Terminal-Bench 2.1 (Harbor, terminus-2) and SWE-bench Pro
(mini-swe-agent and the official evaluator), each run once per wsp run
through kairyu-bench's own adapter with an explicit task list."""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Adapter, Context, Loader, Outcome, Row, Scored, register

SELECTION_ENV = {
    "terminal-bench": "KAIRYU_BENCH_TERMINAL_BENCH_TASKS",
    "swe-bench-pro": "KAIRYU_BENCH_SWEBENCH_PRO_INSTANCES",
}
WORKERS_ENV = {
    "terminal-bench": "KAIRYU_BENCH_TERMINAL_BENCH_WORKERS",
    "swe-bench-pro": "KAIRYU_BENCH_SWEBENCH_PRO_WORKERS",
}


def harbor_cache() -> Path:
    """Harbor's task cache, kept in the benchmark cache (not the container's home)."""

    return Path(os.environ.get("KAIRYU_BENCH_CACHE_DIR", "/work/cache")) / "harbor"


def _task_images(source: str, names: list[str]) -> list[str]:
    """The prebuilt task images the selected tasks run in."""

    if source == "terminal-bench":
        import tomllib

        images: set[str] = set()
        for name in names:
            manifests = sorted((harbor_cache() / "tasks" / "packages" / "terminal-bench" / name).glob("*/task.toml"))
            if not manifests:
                raise RuntimeError(f"Terminal-Bench task {name} is not in Harbor's cache")
            for manifest in manifests:
                image = tomllib.loads(manifest.read_text(encoding="utf-8")).get("environment", {}).get("docker_image")
                if not image:
                    raise RuntimeError(f"Terminal-Bench task {name} has no prebuilt image")
                images.add(str(image))
        return sorted(images)
    from datasets import load_dataset

    from kairyu_bench.manifest import load_manifest as load_benchmarks

    dataset = load_benchmarks()[source]["dataset"]
    tags = {
        row["instance_id"]: row["dockerhub_tag"]
        for row in load_dataset(dataset["id"], split="test", revision=dataset["revision"])
    }
    return sorted(f"jefzda/sweap-images:{tags[name]}" for name in names)


def prepare(items: list[Item]) -> None:
    """Untimed setup for the selected agent tasks: each harness's pinned
    sources and environments (its setup-only mode), Harbor's task packages,
    and every task image pulled into the Docker daemon the runs use."""

    import subprocess
    import tempfile

    from kairyu_bench.manifest import load_manifest as load_benchmarks

    benchmarks = load_benchmarks()
    for source in ("terminal-bench", "swe-bench-pro"):
        names = sorted({item.id.split(":", 1)[1] for item in items if item.source == source})
        if not names:
            continue
        entry = benchmarks[source]
        with tempfile.TemporaryDirectory(prefix=f"wsp-prepare-{source}-") as directory:
            context = Path(directory) / "context.json"
            context.write_text(
                json.dumps(
                    {
                        "benchmark": source,
                        "source": entry["source"],
                        "generator": entry.get("generator"),
                        "dataset": entry["dataset"],
                        "limit": None,
                        "run_id": "wsp-prepare",
                    }
                ),
                encoding="utf-8",
            )
            environment = dict(
                os.environ,
                KAIRYU_BENCH_CONTEXT=str(context),
                KAIRYU_BENCH_RUN_DIR=directory,
                KAIRYU_BENCH_SETUP_ONLY="1",
                KAIRYU_BENCH_HARBOR_CACHE_DIR=str(harbor_cache()),
            )
            subprocess.run(
                ["sh", str(_app_root() / entry["adapter"])], cwd=_app_root(), env=environment, check=True
            )
        images = _task_images(source, names)
        for image in images:
            subprocess.run(["docker", "pull", "--quiet", image], check=True)
        print(f"{source}: {len(names)} tasks prepared, {len(images)} task images pulled", flush=True)


def _candidates() -> dict[str, Any]:
    raw = resources.files("kairyu_bench").joinpath("data/wsp/agent-candidates.json").read_text(encoding="utf-8")
    return json.loads(raw)


def _load(source: str):  # type: ignore[no-untyped-def]
    def load(loader: Loader) -> dict[str, Row]:
        rows: dict[str, Row] = {}
        for name in _candidates()[source]:
            repository = name.split("__")[0].removeprefix("instance_") if source == "swe-bench-pro" else "all"
            rows[f"{source}:{name}"] = {"name": name, "repository": repository}
        return rows

    return load


def _app_root() -> Path:
    configured = os.environ.get("KAIRYU_BENCH_APP_ROOT")
    if configured:
        return Path(configured)
    repository = Path(__file__).resolve().parents[4]
    return repository if (repository / "adapters").is_dir() else Path("/app")


def _terminal_bench_outcomes(run_dir: Path) -> dict[str, dict[str, Any]]:
    from datetime import datetime

    outcomes: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "raw" / "terminal-bench" / "jobs").glob("*/*/result.json")):
        trial = json.loads(path.read_text(encoding="utf-8"))
        name = str(trial.get("task_name", "")).split("/")[-1]
        rewards = ((trial.get("verifier_result") or {}).get("rewards") or {})

        def span(phase: dict[str, Any] | None) -> float | None:
            if not phase or not phase.get("started_at") or not phase.get("finished_at"):
                return None
            return (
                datetime.fromisoformat(str(phase["finished_at"]).replace("Z", "+00:00"))
                - datetime.fromisoformat(str(phase["started_at"]).replace("Z", "+00:00"))
            ).total_seconds()

        setup = [span(trial.get(phase)) for phase in ("environment_setup", "agent_setup")]
        outcomes[name] = {
            "reward": rewards.get("reward"),
            "exception": (trial.get("exception_info") or {}).get("exception_type"),
            "seconds": span(trial),
            # The official per-trial setup (environment start, agent setup);
            # it stays inside the timed window as in the official harness.
            "setup_seconds": sum(value for value in setup if value is not None),
            "trial": path.parent.name,
        }
    return outcomes


# mini-swe-agent's own ends (its InterruptAgentFlow exits) and a context
# overflow of the model's own trajectory. Any other exit status is an
# uncaught exception, e.g. a broken environment, which exits before or
# between model calls with an empty patch.
AGENT_ENDS = frozenset({"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError", "ContextWindowExceededError"})


def _swe_bench_pro_outcomes(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Official per-instance verdicts, read item by item.

    The harness writes the aggregate only when every item succeeded; each
    item's `evaluation/eval_results.json` (the official evaluator's
    `{instance_id: bool}`) is kept either way, so an item that failed leaves
    only its own instance unscored. An instance whose agent ended with an
    exception (no trajectory, or an exit status outside AGENT_ENDS) is
    unscored too: its verdict judges the empty patch, not the model.
    """

    outcomes: dict[str, dict[str, Any]] = {}
    for item in sorted((run_dir / "raw" / "swe-bench-pro" / "items").glob("*")):
        ids, verdict = item / "instance-ids.txt", item / "evaluation" / "eval_results.json"
        if not (ids.is_file() and verdict.is_file()):
            continue
        expected = ids.read_text(encoding="utf-8").strip()
        payload = json.loads(verdict.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {expected} or not isinstance(payload[expected], bool):
            continue
        trajectory = item / "generation" / expected / f"{expected}.traj.json"
        if not trajectory.is_file():
            continue
        if json.loads(trajectory.read_text(encoding="utf-8")).get("info", {}).get("exit_status") not in AGENT_ENDS:
            continue
        # The harness records no per-item time; an item's span is from its
        # selection file to its verdict file.
        outcomes[expected] = {
            "reward": float(payload[expected]),
            "seconds": verdict.stat().st_mtime - ids.stat().st_mtime,
        }
    return outcomes


def _worker_limit(source: str):  # type: ignore[no-untyped-def]
    # The kairyu-bench wrapper always passes these (default 4).
    return lambda: int(os.environ.get(WORKERS_ENV[source], "4"))


def _batch(source: str):  # type: ignore[no-untyped-def]
    def run(context: Context, items: list[Item], rows: dict[str, Row], workers: int) -> dict[str, Outcome]:
        from kairyu_bench.manifest import load_manifest as load_benchmarks
        from kairyu_bench.runner import RunConfig, run_benchmarks
        from kairyu_bench.target import Endpoint, TargetClient

        assert context.chat is not None and context.run_dir is not None
        names = [rows[item.id]["name"] for item in items]
        os.environ[SELECTION_ENV[source]] = ",".join(names)
        os.environ[WORKERS_ENV[source]] = str(workers)
        # Task packages and images come from `wsp prepare` (untimed).
        os.environ["KAIRYU_BENCH_HARBOR_CACHE_DIR"] = str(harbor_cache())
        endpoint = Endpoint.parse(context.chat.base_url)
        config = RunConfig(
            endpoint=endpoint,
            selected=(source,),
            limit=None,
            results_root=context.run_dir / "agents",
            run_id=source,
            app_root=_app_root(),
        )
        outcome = run_benchmarks(config, TargetClient(endpoint, api_key=context.chat.api_key), load_benchmarks())
        found = (
            _terminal_bench_outcomes(outcome.run_dir)
            if source == "terminal-bench"
            else _swe_bench_pro_outcomes(outcome.run_dir)
        )
        # A task without a verifier verdict is left out: the runner records it
        # as unscored (an infrastructure failure), never as 0.
        results: dict[str, Outcome] = {}
        for item in items:
            name = rows[item.id]["name"]
            trial = found.get(name)
            if trial is not None and trial.get("reward") is not None:
                results[item.id] = {"samples": [{"task": name, **trial}], "harness_run": str(outcome.run_dir)}
        return results

    return run


def _score(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    return [
        Scored(float(outcome["samples"][0]["reward"]), (float(outcome["samples"][0]["reward"]),), {})
        for _, _, outcome in entries
    ]


def _unused(context: Context, item: Item, row: Row) -> Outcome:
    raise AssertionError("agent items run in batches")


for _source in ("terminal-bench", "swe-bench-pro"):
    register(
        _source,
        Adapter(
            load=_load(_source),
            stratum=lambda row: str(row["repository"]),
            execute=_unused,
            score=_score,
            batch=_batch(_source),
            worker_limit=_worker_limit(_source),
        ),
    )
