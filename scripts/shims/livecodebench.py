from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from kairyu_bench.adapter_context import read_context
from kairyu_bench.target import Endpoint, TargetClient


def main() -> int:
    context_path = os.environ.get("KAIRYU_BENCH_CONTEXT")
    raw_directory = os.environ.get("KAIRYU_BENCH_RAW_DIR")
    if not context_path or not raw_directory:
        raise RuntimeError("adapter environment is incomplete")
    context = read_context(Path(context_path))

    from datasets import load_dataset
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation import codegen_metrics, extract_instance_results
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts import format_prompt_generation
    from lcb_runner.utils.extraction_utils import extract_code

    dataset = context["dataset"]
    rows = load_dataset(
        dataset["id"],
        split="test",
        revision=dataset["revision"],
        version_tag="release_v6",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    problems = sorted(
        (CodeGenerationProblem(**row) for row in rows),
        key=lambda problem: problem.question_id,
    )
    limit = context.get("limit")
    if limit is not None:
        problems = problems[:limit]
    if not problems:
        raise RuntimeError("LiveCodeBench selected no problems")

    client = TargetClient(
        Endpoint.parse(context["endpoint"]),
        api_key=os.environ.get("KAIRYU_API_KEY"),
        timeout=300,
    )
    output_lists: list[list[str]] = []
    code_lists: list[list[str]] = []
    for problem in problems:
        messages = format_prompt_generation(problem, LMStyle.OpenAIChat)
        response = client.chat(context["model_id"], messages, max_tokens=2000)
        output_lists.append([response])
        code_lists.append([extract_code(response, LMStyle.OpenAIChat)])

    samples = [problem.get_evaluation_sample() for problem in problems]
    metrics = codegen_metrics(
        samples,
        code_lists,
        num_process_evaluate=1,
        timeout=6,
    )
    graded = extract_instance_results(metrics[1])
    metadata = metrics[2]
    records = [
        problem.insert_output_evaluation(outputs, codes, grades, metadata=meta)
        for problem, outputs, codes, grades, meta in zip(
            problems, output_lists, code_lists, graded, metadata
        )
    ]
    summary: dict[str, Any] = {"metrics": metrics, "records": records}
    output = Path(raw_directory) / "summary.json"
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
