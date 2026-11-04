"""Per-source adapters: load rows, place them in strata, run, score.

Each adapter module registers one :class:`Adapter` per source name. Official
checkers are imported lazily inside the scoring functions, so the runner and
its unit tests import without the scorer environment.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.client import ChatClient, ChatReply, LogprobClient
from kairyu_bench.wsp.manifest import Item

Row = dict[str, Any]
Outcome = dict[str, Any]


@dataclass(frozen=True)
class Loader:
    """Pinned datasets (HF cache), checked-out official sources, the scorer
    interpreters of separate environments, and pinned data files."""

    sources: Mapping[str, Path] = field(default_factory=dict)
    pythons: Mapping[str, Path] = field(default_factory=dict)
    files: Mapping[str, Path] = field(default_factory=dict)
    # Where `wsp prepare` writes generated data (RULER).
    data_dir: Path | None = None

    def dataset(
        self,
        name: str,
        *,
        split: str,
        revision: str | None,
        config: str | None = None,
        data_files: str | None = None,
    ) -> Any:
        from datasets import load_dataset

        if data_files is not None:
            return load_dataset(
                name,
                data_files=data_files,
                split=split,
                revision=revision,
                token=os.environ.get("HF_TOKEN") or None,
            )
        return load_dataset(
            name, config, split=split, revision=revision, token=os.environ.get("HF_TOKEN") or None
        )

    def json_files(self, name: str, *, revision: str, files: list[str]) -> Any:
        """JSON lines files of a dataset repository at a pinned revision, read
        without the repository's own loading script."""

        from datasets import load_dataset

        return load_dataset(
            "json",
            data_files=[f"hf://datasets/{name}@{revision}/{file}" for file in files],
            split="train",
            token=os.environ.get("HF_TOKEN") or None,
        )

    def source(self, name: str) -> Path:
        path = self.sources.get(name)
        if path is None or not path.is_dir():
            raise RuntimeError(f"official source {name} is not checked out (run wsp prepare)")
        return path

    def python(self, name: str) -> Path:
        path = self.pythons.get(name)
        if path is None or not path.exists():
            raise RuntimeError(f"scorer environment {name} is not prepared (run wsp prepare)")
        return path

    def file(self, name: str) -> Path:
        path = self.files.get(name)
        if path is None or not path.is_file():
            raise RuntimeError(f"pinned file {name} is missing (run wsp prepare)")
        return path


def run_scorer(
    python: Path, module: str, payload: Any, *, pythonpath: tuple[Path, ...] = (), env: Mapping[str, str] | None = None
) -> Any:
    """Run ``python -m module input output`` and return its JSON output.

    Code-executing scorers run in their own process (and environment), never
    in the threaded runner.
    """

    import json
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="wsp-score-") as directory:
        source, target = Path(directory) / "input.json", Path(directory) / "output.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        variables = dict(os.environ)
        variables.update(env or {})
        variables["PYTHONPATH"] = os.pathsep.join(str(path) for path in (root, *pythonpath))
        completed = subprocess.run(
            [str(python), "-m", module, str(source), str(target)],
            env=variables,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{module} failed ({completed.returncode}): {completed.stderr[-2000:]}")
        return json.loads(target.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Context:
    chat: ChatClient | None
    logprob: LogprobClient | None
    loader: Loader = field(default_factory=Loader)
    run_dir: Path | None = None


@dataclass(frozen=True)
class Scored:
    score: float  # mean over samples, 0..1
    samples: tuple[float, ...]
    detail: dict[str, Any]


@dataclass(frozen=True)
class Adapter:
    load: Callable[[Loader], dict[str, Row]]
    stratum: Callable[[Row], str]
    execute: Callable[[Context, Item, Row], Outcome]
    score: Callable[[list[tuple[Item, Row, Outcome]], Loader], list[Scored]]
    # The problem an item presents; one problem in several sources or forms
    # stays in one split. None: the item is its own problem.
    group: Callable[[Row], str] | None = None
    # Strata take equal shares of the selection instead of shares by size
    # (e.g. BFCL's four official groups).
    equal_strata: bool = False
    # Sources whose items run together in one official harness invocation
    # (agent tasks): returns an outcome per item id; `execute` is unused.
    batch: Callable[[Context, list[Item], dict[str, Row]], dict[str, Outcome]] | None = None


ADAPTERS: dict[str, Adapter] = {}
_IMPORTED: list[str] = []

MODULES = (
    "agents",
    "choice",
    "code",
    "instructions",
    "long_context",
    "logprob",
    "mathematics",
    "text",
    "tools",
    "vision",
)


def register(name: str, adapter: Adapter) -> None:
    if name in ADAPTERS:
        raise ValueError(f"duplicate adapter: {name}")
    ADAPTERS[name] = adapter


def adapters() -> dict[str, Adapter]:
    # Importing a module registers its sources; any may already be imported.
    if not _IMPORTED:
        for module in MODULES:
            importlib.import_module(f"kairyu_bench.wsp.sources.{module}")
        _IMPORTED.append("all")
    return ADAPTERS


def official_functions(
    path: Path, names: tuple[str, ...], *, module_globals: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Top-level functions of a pinned official file, without running the module.

    Some official scorers run their own experiment loop at import; only the
    named definitions are compiled from the file's syntax tree.
    """

    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    kept: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = set(names) - {node.name for node in kept if isinstance(node, ast.FunctionDef)}
    if missing:
        raise RuntimeError(f"{path}: missing official functions {sorted(missing)}")
    namespace: dict[str, Any] = dict(module_globals or {})
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return {name: namespace[name] for name in names}


def official_constants(path: Path, assignments: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Literal entries of dict assignments in a pinned official file (any depth).

    Non-literal entries (references to other objects) are skipped.
    """

    import ast

    found: dict[str, dict[str, Any]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in assignments:
                    entries: dict[str, Any] = {}
                    for key, value in zip(node.value.keys, node.value.values, strict=True):
                        try:
                            entries[ast.literal_eval(key)] = ast.literal_eval(value)  # type: ignore[arg-type]
                        except ValueError:
                            continue
                    found[target.id] = entries
    missing = set(assignments) - set(found)
    if missing:
        raise RuntimeError(f"{path}: missing official constants {sorted(missing)}")
    return found


def seed(item_id: str, sample: int) -> int:
    digest = hashlib.sha256(f"kairyu-bench-wsp-seed\0{item_id}\0{sample}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def reply_record(reply: ChatReply) -> dict[str, Any]:
    return asdict(reply)


def run_chat(
    context: Context, item: Item, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None
) -> Outcome:
    """``item.samples`` independent single-turn requests at the item's token cap."""

    if context.chat is None:
        raise RuntimeError("chat endpoint is required")
    samples = [
        reply_record(
            context.chat.chat(messages, max_tokens=item.max_tokens, seed=seed(item.id, sample), tools=tools)
        )
        for sample in range(item.samples)
    ]
    return {"samples": samples}


def score_each(
    function: Callable[[Row, dict[str, Any]], tuple[float, dict[str, Any]]],
) -> Callable[[list[tuple[Item, Row, Outcome]], Loader], list[Scored]]:
    """Batch scorer from a per-sample ``(row, sample) -> (score, detail)`` rule.

    A reply cut at the token cap scores 0 whatever it contains.
    """

    def score(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
        results: list[Scored] = []
        for _, row, outcome in entries:
            values: list[float] = []
            details: list[dict[str, Any]] = []
            for sample in outcome["samples"]:
                if sample.get("finish_reason") == "length":
                    values.append(0.0)
                    details.append({"truncated": True})
                    continue
                value, detail = function(row, sample)
                values.append(float(value))
                details.append(detail)
            results.append(Scored(sum(values) / len(values), tuple(values), {"samples": details}))
        return results

    return score
