"""Code generation: HumanEval+ / MBPP+ (EvalPlus 0.3.1), LiveCodeBench v6
(kairyu-bench's pinned source) and SciCode (pinned source, without
background). Generated code is executed only in scorer processes."""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from typing import Any

from kairyu_bench.wsp.catalog import SOURCE_BY_NAME
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.sources import (
    Adapter,
    Context,
    Loader,
    Outcome,
    Row,
    Scored,
    official_functions,
    register,
    reply_record,
    run_chat,
    run_scorer,
    seed,
)

# evalplus 0.3.1 codegen.py / provider/openai.py / gen/util/openai_request.py.
EVALPLUS_SYSTEM = "You are a helpful assistant good at coding."
EVALPLUS_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
)
# evalplus 0.3.1 always fails HumanEval/32's canonical solution (fixed later
# upstream), so the item could never be solved.
EVALPLUS_EXCLUDED = {"HumanEval/32"}


def _evalplus(source: str):  # type: ignore[no-untyped-def]
    dataset = "humaneval" if source == "humaneval-plus" else "mbpp"

    def load(loader: Loader) -> dict[str, Row]:
        tasks = json.loads(loader.file("evalplus-tasks").read_text(encoding="utf-8"))[dataset]
        loaded: dict[str, Row] = {}
        for task_id, task in tasks.items():
            if task_id in EVALPLUS_EXCLUDED:
                continue
            prompt = task["prompt"].strip() + "\n"
            loaded[f"{source}:{task_id}"] = {
                "task_id": task_id,
                "dataset": dataset,
                "entry_point": task["entry_point"],
                "messages": [
                    {"role": "system", "content": EVALPLUS_SYSTEM},
                    {"role": "user", "content": EVALPLUS_INSTRUCTION + f"\n```python\n{prompt.strip()}\n```"},
                ],
            }
        return loaded

    return load


def _score_evalplus(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    jobs = [
        {
            "key": f"{position}:{index}",
            "dataset": row["dataset"],
            "task_id": row["task_id"],
            "entry_point": row["entry_point"],
            "content": sample.get("content") or "",
        }
        for position, (_, row, outcome) in enumerate(entries)
        for index, sample in enumerate(outcome["samples"])
        if sample.get("finish_reason") != "length"
    ]
    verdicts = run_scorer(loader.python("evalplus"), "kairyu_bench.wsp.scorers.evalplus_score", jobs) if jobs else {}
    return _collect(entries, verdicts)


def _collect(entries: list[tuple[Item, Row, Outcome]], verdicts: dict[str, Any]) -> list[Scored]:
    results: list[Scored] = []
    for position, (_, _, outcome) in enumerate(entries):
        values, details = [], []
        for index, sample in enumerate(outcome["samples"]):
            verdict = verdicts.get(f"{position}:{index}")
            if verdict is None:
                values.append(0.0)
                details.append({"truncated": sample.get("finish_reason") == "length"})
            else:
                values.append(float(verdict["passed"]))
                details.append(verdict)
        results.append(Scored(sum(values) / len(values), tuple(values), {"samples": details}))
    return results


for _source in ("humaneval-plus", "mbpp-plus"):
    register(
        _source,
        Adapter(
            load=_evalplus(_source),
            stratum=lambda row: "all",
            execute=lambda context, item, row: run_chat(context, item, row["messages"]),
            score=_score_evalplus,
        ),
    )


# LiveCodeBench release_v6 at kairyu-bench's pins; problems from 2025 (the
# newest contests in v6), the official chat prompt, code extraction and
# codegen_metrics (6 s per test, as kairyu-bench's adapter).
LIVECODEBENCH_FROM = datetime(2025, 1, 1)


def _lcb_modules(loader: Loader) -> Any:
    import os
    import sys

    root = str(loader.source("livecodebench"))
    if root not in sys.path:
        sys.path.insert(0, root)
    # lcb_runner.prompts opens its few-shot files relative to the working
    # directory at import; the first import happens while loading, before
    # any worker thread exists.
    previous = os.getcwd()
    os.chdir(root)
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.prompts import format_prompt_generation
        from lcb_runner.utils.extraction_utils import extract_code
    finally:
        os.chdir(previous)

    return CodeGenerationProblem, LMStyle, format_prompt_generation, extract_code


# release_v6 is test.jsonl..test6.jsonl (code_generation_lite.py
# ALLOWED_FILES); its 2025 problems are in the last two files. The dataset
# script itself is not run (datasets 5 no longer executes dataset scripts).
LIVECODEBENCH_FILES = ["test5.jsonl", "test6.jsonl"]


def _livecodebench(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["livecodebench"]
    problem_class, style, prompt, _ = _lcb_modules(loader)
    rows = loader.json_files(spec.dataset or "", revision=spec.revision or "", files=LIVECODEBENCH_FILES)
    loaded: dict[str, Row] = {}
    for raw in rows:
        row = dict(raw)
        # The JSON reader parses dates; lcb_runner expects the file's strings.
        if isinstance(row["contest_date"], datetime):
            row["contest_date"] = row["contest_date"].isoformat()
        if datetime.fromisoformat(row["contest_date"]) < LIVECODEBENCH_FROM:
            continue
        problem = problem_class(**row)
        loaded[f"livecodebench:{problem.question_id}"] = {
            "messages": prompt(problem, style.OpenAIChat),
            "difficulty": problem.difficulty.value,
            "problem": problem,
        }
    return loaded


def _score_livecodebench(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    _, style, _, extract = _lcb_modules(loader)
    jobs = [
        {
            "key": f"{position}:{index}",
            "sample": row["problem"].get_evaluation_sample(),
            "code": extract(sample.get("content") or "", style.OpenAIChat),
        }
        for position, (_, row, outcome) in enumerate(entries)
        for index, sample in enumerate(outcome["samples"])
        if sample.get("finish_reason") != "length"
    ]
    verdicts = (
        run_scorer(
            loader.python("wsp"),
            "kairyu_bench.wsp.scorers.livecodebench_score",
            jobs,
            pythonpath=(loader.source("livecodebench"),),
        )
        if jobs
        else {}
    )
    return _collect(entries, verdicts)


register(
    "livecodebench",
    Adapter(
        load=_livecodebench,
        stratum=lambda row: str(row["difficulty"]),
        execute=lambda context, item, row: run_chat(context, item, row["messages"]),
        score=_score_livecodebench,
    ),
)


# SciCode e3158ea011d4235245a547460d3688d7ccbf9900 eval/inspect_ai/scicode.py:
# steps in order with the model's own earlier code, background_comment
# template (kairyu-bench's default, without background); three author-
# provided steps are not generated or tested. The item score is the share of
# a problem's tested steps that pass.
SCICODE_AUTHOR_STEPS = {("13", 5), ("62", 0), ("76", 2)}


def _scicode_official(loader: Loader) -> dict[str, Any]:
    root = loader.source("scicode")
    functions = official_functions(
        root / "src" / "scicode" / "gen" / "models.py", ("extract_python_script",), module_globals={"re": re}
    )
    functions.update(
        official_functions(
            root / "src" / "scicode" / "parse" / "parse.py",
            ("extract_function_name", "get_function_from_code"),
            module_globals={"re": re, "ast": ast},
        )
    )
    functions["template"] = (root / "eval" / "data" / "background_comment_template.txt").read_text(encoding="utf-8")
    return functions


def _scicode(loader: Loader) -> dict[str, Row]:
    spec = SOURCE_BY_NAME["scicode"]
    rows = loader.dataset(spec.dataset or "", split="test", revision=spec.revision)
    return {
        f"scicode:{row['problem_id']}": {"problem": dict(row), "steps": len(row["sub_steps"])}
        for row in rows
    }


def _execute_scicode(context: Context, item: Item, row: Row) -> Outcome:
    assert context.chat is not None
    loader = context.loader
    official = _scicode_official(loader)
    problem = row["problem"]
    problem_id, steps = problem["problem_id"], problem["sub_steps"]
    samples = []
    for sample_index in range(item.samples):
        previous: list[str | None] = [None] * len(steps)
        files: dict[str, str] = {}
        calls: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            if (problem_id, index) in SCICODE_AUTHOR_STEPS:
                author = loader.source("scicode") / "eval" / "data" / f"{problem_id}.{index + 1}.txt"
                previous[index] = official["get_function_from_code"](
                    author.read_text(encoding="utf-8"), official["extract_function_name"](step["function_header"])
                )
                continue
            lines: list[str] = []
            for earlier in range(index):
                lines += [steps[earlier]["step_description_prompt"], str(previous[earlier]), "------"]
            problem_steps = "\n\n".join(lines[:-1])
            next_step = "\n\n".join(
                [step["step_description_prompt"], f"{step['function_header']}\n\n{step['return_line']}"]
            )
            prompt = official["template"].format(
                problem_steps_str=problem_steps,
                next_step_str=next_step,
                dependencies=problem["required_dependencies"],
            )
            reply = context.chat.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=item.max_tokens,
                seed=seed(f"{item.id}:{index}", sample_index),
            )
            calls.append(reply_record(reply))
            code = official["extract_python_script"](reply.content) if reply.finish_reason != "length" else ""
            prefix = f"{problem['required_dependencies']}\n" + "\n".join(str(p) for p in previous[:index]) + "\n"
            files[step["step_number"]] = f"{prefix}\n{code}"
            previous[index] = code
        samples.append(
            {
                "calls": calls,
                "files": files,
                "completion_tokens": sum(call["completion_tokens"] for call in calls),
                "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
                "finish_reason": "length" if any(c["finish_reason"] == "length" for c in calls) else "stop",
                "content": "",
            }
        )
    return {"samples": samples}


def _score_scicode(entries: list[tuple[Item, Row, Outcome]], loader: Loader) -> list[Scored]:
    jobs = []
    for position, (_, row, outcome) in enumerate(entries):
        steps = row["problem"]["sub_steps"]
        for index, sample in enumerate(outcome["samples"]):
            # A sample with any step cut at the token cap scores 0 (below).
            if sample.get("finish_reason") == "length":
                continue
            for step_index, step in enumerate(steps):
                if (row["problem"]["problem_id"], step_index) in SCICODE_AUTHOR_STEPS:
                    continue
                jobs.append(
                    {
                        "key": f"{position}:{index}:{step['step_number']}",
                        "step": step["step_number"],
                        "code": sample["files"].get(step["step_number"], ""),
                        "tests": list(step["test_cases"]),
                    }
                )
    execution = (
        run_scorer(
            loader.python("wsp"),
            "kairyu_bench.wsp.scorers.scicode_score",
            {"python": str(loader.python("scicode")), "h5": str(loader.file("scicode-h5")), "jobs": jobs},
            pythonpath=(loader.source("scicode") / "src",),
        )
        if jobs
        else {"schema_version": 1, "verdicts": {}, "diagnostics": {}}
    )
    verdicts, diagnostics = execution["verdicts"], execution["diagnostics"]
    results: list[Scored] = []
    for position, (_, row, outcome) in enumerate(entries):
        values: list[float] = []
        details: list[dict[str, Any]] = []
        for index, sample in enumerate(outcome["samples"]):
            if sample.get("finish_reason") == "length":
                values.append(0.0)
                details.append({"truncated": True})
                continue
            keys = [key for key in verdicts if key.startswith(f"{position}:{index}:")]
            passed = sum(bool(verdicts[key]) for key in keys)
            values.append(passed / len(keys) if keys else 0.0)
            details.append({
                "steps_passed": passed, "steps_tested": len(keys), "problem_solved": passed == len(keys),
                "step_diagnostics": [
                    {"step": key.rsplit(":", 1)[1], "passed": bool(verdicts[key]), **diagnostics[key]}
                    for key in keys
                ],
            })
        results.append(Scored(sum(values) / len(values), tuple(values), {"samples": details}))
    return results


register(
    "scicode",
    Adapter(
        load=_scicode,
        stratum=lambda row: "steps<=3" if row["steps"] <= 3 else "steps>3",
        execute=_execute_scicode,
        score=_score_scicode,
    ),
)
