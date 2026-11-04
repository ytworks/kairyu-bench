"""Instruction following: IFEval (google-research) and IFBench (allenai),
scored per response with their official checkers. Headline: prompt-level
loose accuracy for IFBench (its README) and prompt-level strict for IFEval."""

from __future__ import annotations

import sys
import threading
from typing import Any

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import Adapter, Loader, Outcome, Row, Scored, register, run_chat

_import_lock = threading.Lock()


def _kwargs(row: Row) -> list[dict[str, Any]]:
    # `is not None`, not truthiness: 0 is a real argument (IFBench n_start).
    return [{key: value for key, value in (entry or {}).items() if value is not None} for entry in row["kwargs"]]


def _load(source: str):  # type: ignore[no-untyped-def]
    spec = SOURCE_BY_NAME[source]

    def load(loader: Loader) -> dict[str, Row]:
        rows = loader.dataset(spec.dataset or "", split="train", revision=spec.revision)
        return {
            f"{source}:{row['key']}": {
                "key": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": list(row["instruction_id_list"]),
                "kwargs": _kwargs(dict(row)),
            }
            for row in rows
        }

    return load


def _evaluation_lib(source: str, loader_root: str) -> Any:
    with _import_lock:
        if loader_root not in sys.path:
            sys.path.insert(0, loader_root)
        if source == "ifeval":
            import langdetect  # type: ignore[import-untyped]
            from instruction_following_eval import evaluation_lib  # type: ignore[import-not-found]

            # langdetect is unseeded upstream; a fixed seed makes the
            # language checks deterministic.
            langdetect.DetectorFactory.seed = 0
            return evaluation_lib
        import evaluation_lib  # type: ignore[import-not-found]

        return evaluation_lib


def _scorer(source: str):  # type: ignore[no-untyped-def]
    def score(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
        library = _evaluation_lib(source, str(loader.source(source)))
        results: list[Scored] = []
        for _, row, outcome in entries:
            values: list[float] = []
            details: list[dict[str, Any]] = []
            for sample in outcome["samples"]:
                response = "" if sample.get("finish_reason") == "length" else (sample.get("content") or "")
                example = library.InputExample(
                    key=row["key"],
                    instruction_id_list=list(row["instruction_id_list"]),
                    prompt=row["prompt"],
                    kwargs=[dict(entry) for entry in row["kwargs"]],
                )
                strict = library.test_instruction_following_strict(example, {row["prompt"]: response})
                example.kwargs = [dict(entry) for entry in row["kwargs"]]
                loose = library.test_instruction_following_loose(example, {row["prompt"]: response})
                headline = strict if source == "ifeval" else loose
                values.append(float(headline.follow_all_instructions))
                details.append(
                    {
                        "strict": bool(strict.follow_all_instructions),
                        "loose": bool(loose.follow_all_instructions),
                        "strict_instructions": list(strict.follow_instruction_list),
                        "loose_instructions": list(loose.follow_instruction_list),
                    }
                )
            results.append(Scored(sum(values) / len(values), tuple(values), {"samples": details}))
        return results

    return score


for _source in ("ifeval", "ifbench"):
    register(
        _source,
        Adapter(
            load=_load(_source),
            stratum=lambda row: str(row["instruction_id_list"][0]).split(":")[0],
            execute=lambda context, item, row: run_chat(
                context, item, [{"role": "user", "content": row["prompt"]}]
            ),
            score=_scorer(_source),
        ),
    )
