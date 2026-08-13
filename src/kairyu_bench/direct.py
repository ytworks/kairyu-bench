from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from kairyu_bench.results import BenchmarkResult
from kairyu_bench.selection import select_problem_ids
from kairyu_bench.target import Endpoint, TargetClient


DIRECT_BENCHMARKS = {
    "gpqa-diamond",
    "long-context-reasoning",
    "mrcr-v2",
}
MRCR_BIN_UPPER_BOUNDS = (8192, 16384, 32768, 65536, 131072)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def deterministic_gpqa_choices(
    item_id: str, correct: str, incorrect: list[str]
) -> tuple[list[str], str]:
    choices = [correct, *incorrect]

    def order(choice: str) -> str:
        material = f"kairyu-bench-gpqa-v1\0{item_id}\0{choice}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    choices.sort(key=order)
    return choices, "ABCD"[choices.index(correct)]


def extract_choice(response: str) -> str | None:
    stripped = response.strip().upper()
    if re.fullmatch(r"\(?[A-D]\)?[.!]?", stripped):
        return re.search(r"[A-D]", stripped).group(0)  # type: ignore[union-attr]
    matches = re.findall(
        r"(?:FINAL\s+)?ANSWER\s*(?:IS\s*)?[:=-]?\s*\(?([A-D])\)?\b",
        response.upper(),
    )
    return matches[-1] if matches else None


def gpqa_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    item_id = f"gpqa-diamond-{index:04d}"
    correct = str(row["Correct Answer"]).strip()
    incorrect = [
        str(row[f"Incorrect Answer {number}"]).strip() for number in range(1, 4)
    ]
    choices, expected = deterministic_gpqa_choices(item_id, correct, incorrect)
    prompt = (
        f"{str(row['Question']).strip()}\n\n"
        + "\n".join(f"{letter}) {choice}" for letter, choice in zip("ABCD", choices))
        + "\n\nAnswer with the single letter of the correct choice."
    )
    return {
        "id": item_id,
        "messages": [{"role": "user", "content": prompt}],
        "expected": expected,
    }


def longbench_item(row: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""Read the context and answer the multiple-choice question.

<context>
{row['context']}
</context>

Question: {row['question']}

Choices:
A) {row['choice_A']}
B) {row['choice_B']}
C) {row['choice_C']}
D) {row['choice_D']}

Answer with the single letter of the correct choice. End with 'Answer: <letter>'."""
    return {
        "id": str(row["_id"]),
        "messages": [{"role": "user", "content": prompt}],
        "expected": str(row["answer"]).strip().upper(),
    }


def token_bin(total_tokens: int) -> int | None:
    if total_tokens < 4096:
        return None
    for upper_bound in MRCR_BIN_UPPER_BOUNDS:
        if total_tokens <= upper_bound:
            return upper_bound
    return None


def mrcr_score(response: str, answer: str, prepend: str) -> float:
    if not response.startswith(prepend):
        return 0.0
    return SequenceMatcher(
        None,
        response.removeprefix(prepend),
        answer.removeprefix(prepend),
    ).ratio()


def _normalize_items(
    name: str,
    rows: Iterable[dict[str, Any]],
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    if name == "gpqa-diamond":
        items = [gpqa_item(row, index) for index, row in enumerate(rows)]
    elif name == "long-context-reasoning":
        items = [longbench_item(row) for row in rows]
    elif name == "mrcr-v2":
        try:
            import tiktoken
        except ImportError as error:
            raise RuntimeError("MRCR requires tiktoken for official token bins") from error
        encoder = tiktoken.get_encoding("o200k_base")
        items = []
        for index, row in enumerate(rows):
            if row.get("n_needles") != 8:
                continue
            messages = json.loads(row["prompt"])
            answer = str(row["answer"])
            total_tokens = sum(
                len(encoder.encode(str(message.get("content", ""))))
                for message in messages
            ) + len(encoder.encode(answer))
            bin_upper_bound = token_bin(total_tokens)
            if bin_upper_bound is None:
                continue
            items.append(
                {
                    "id": f"mrcr-v2-{index:04d}",
                    "messages": messages,
                    "answer": answer,
                    "prepend": str(row["random_string_to_prepend"]),
                    "token_bin": bin_upper_bound,
                }
            )
            if limit is not None and len(items) >= limit:
                break
    else:
        raise ValueError(f"unsupported direct benchmark: {name}")
    selected_ids = select_problem_ids([item["id"] for item in items], limit)
    selected = set(selected_ids)
    return [item for item in items if item["id"] in selected]


def _score(name: str, item: dict[str, Any], response: str) -> float:
    if name in {"gpqa-diamond", "long-context-reasoning"}:
        return 1.0 if extract_choice(response) == item["expected"] else 0.0
    return mrcr_score(response, item["answer"], item["prepend"])


def _result(
    context: dict[str, Any],
    *,
    status: str,
    items: list[dict[str, Any]],
    evaluated: int,
    scores: list[float],
    started_at: str,
    error: str | None,
) -> BenchmarkResult:
    name = context["benchmark"]
    primary = 100 * sum(scores) / len(scores) if scores else None
    metric_name = "sequence_match_ratio" if name == "mrcr-v2" else "accuracy"
    scoring = context["scoring"]
    return BenchmarkResult.from_dict(
        {
            "schema_version": 1,
            "run_id": context["run_id"],
            "benchmark": name,
            "status": status,
            "endpoint": {"fingerprint": context["endpoint_fingerprint"]},
            "model_id": context["model_id"],
            "source": {
                "repository": context["source"]["repository"],
                "revision": context["source"]["revision"],
                "dataset": context["dataset"]["id"],
                "dataset_revision": context["dataset"]["revision"],
            },
            "selection": {
                "requested_limit": context["limit"],
                "problem_ids": [item["id"] for item in items],
            },
            "counts": {"requested": len(items), "evaluated": evaluated},
            "score": {
                "primary": primary,
                "unit": scoring["unit"],
                "metrics": {metric_name: primary} if primary is not None else {},
            },
            "scoring": {
                "method": scoring["method"],
                "self_judged": bool(scoring.get("self_judged", False)),
                "self_simulated": bool(scoring.get("self_simulated", False)),
            },
            "artifacts": {
                "raw": [f"raw/{name}.jsonl"] if items else [],
                "logs": [f"logs/{name}.log"],
            },
            "timestamps": {"started_at": started_at, "finished_at": _utc_now()},
            "error": error,
        }
    )


def execute_direct(
    context: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    client: TargetClient,
) -> tuple[BenchmarkResult, list[dict[str, Any]]]:
    started_at = _utc_now()
    name = context["benchmark"]
    items = _normalize_items(name, rows, limit=context["limit"])
    records: list[dict[str, Any]] = []
    scores: list[float] = []
    errors: list[str] = []
    max_tokens = 8192 if name == "mrcr-v2" else 2048
    for item in items:
        try:
            response = client.chat(
                context["model_id"], item["messages"], max_tokens=max_tokens
            )
            score = _score(name, item, response)
            scores.append(score)
            records.append(
                {"id": item["id"], "status": "completed", "score": score, "response": response}
            )
        except Exception as exception:  # external endpoint failures become partial rows
            message = f"{item['id']}: {exception}"
            errors.append(message)
            records.append({"id": item["id"], "status": "failed", "error": str(exception)})
    if not items or not scores:
        status = "failed"
    elif len(scores) == len(items):
        status = "completed"
    else:
        status = "partial"
    error = "; ".join(errors) if errors else ("dataset selected no problems" if not items else None)
    return (
        _result(
            context,
            status=status,
            items=items,
            evaluated=len(scores),
            scores=scores,
            started_at=started_at,
            error=error,
        ),
        records,
    )


def _load_rows(context: dict[str, Any]) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("datasets package is not installed") from error
    name = context["benchmark"]
    dataset = context["dataset"]
    kwargs: dict[str, Any] = {
        "split": "train",
        "revision": dataset["revision"],
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    if name == "gpqa-diamond":
        return load_dataset(dataset["id"], "gpqa_diamond", **kwargs)
    return load_dataset(dataset["id"], **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=sorted(DIRECT_BENCHMARKS))
    args = parser.parse_args(argv)
    context_path = os.environ.get("KAIRYU_BENCH_CONTEXT")
    result_path = os.environ.get("KAIRYU_BENCH_RESULT_PATH")
    if not context_path or not result_path:
        parser.error("adapter environment is incomplete")
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    if context["benchmark"] != args.benchmark:
        parser.error("adapter benchmark differs from context")
    try:
        rows = _load_rows(context)
        client = TargetClient(
            Endpoint.parse(context["endpoint"]),
            api_key=os.environ.get("KAIRYU_API_KEY"),
            timeout=300,
        )
        result, records = execute_direct(context, rows, client)
    except Exception as exception:
        result = _result(
            context,
            status="unsupported",
            items=[],
            evaluated=0,
            scores=[],
            started_at=_utc_now(),
            error=str(exception),
        )
        records = []

    raw_path = Path(context["run_dir"]) / "raw" / f"{args.benchmark}.jsonl"
    if records:
        raw_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    result.write(Path(result_path))
    return 0 if result.status == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
