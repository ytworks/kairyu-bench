from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from kairyu_bench.adapter_context import read_context
from kairyu_bench.target import Endpoint, TargetClient


def main() -> int:
    context_path = os.environ.get("KAIRYU_BENCH_CONTEXT")
    raw_directory = os.environ.get("KAIRYU_BENCH_RAW_DIR")
    if not context_path or not raw_directory:
        raise RuntimeError("adapter environment is incomplete")
    context = read_context(Path(context_path))

    import judge as official_judge
    from benchmark import BenchmarkResult, get_problem_set
    from datasets import load_dataset
    from judge import LightCPVerifierJudge, SupportedLanguage
    from util import extract_longest_cpp_code

    secondary = context.get("secondary_sources", [])
    testcase = next(
        source
        for source in secondary
        if source["id"] == "QAQAQAQAQ/LiveCodeBench-Pro-Testcase"
    )
    original_download = official_judge.hf_hub_download

    def pinned_download(*args, **kwargs):
        kwargs["revision"] = testcase["revision"]
        kwargs["token"] = os.environ.get("HF_TOKEN")
        return original_download(*args, **kwargs)

    official_judge.hf_hub_download = pinned_download
    dataset_lock = context["dataset"]
    dataset = load_dataset(
        dataset_lock["id"],
        revision=dataset_lock["revision"],
        token=os.environ.get("HF_TOKEN"),
    )
    problem_set = get_problem_set(dataset)
    problems = list(problem_set.values())
    limit = context.get("limit")
    if limit is not None:
        problems = problems[:limit]
    if not problems:
        raise RuntimeError("LiveCodeBench Pro selected no problems")

    client = TargetClient(
        Endpoint.parse(context["endpoint"]),
        api_key=os.environ.get("KAIRYU_API_KEY"),
        timeout=300,
    )
    system_prompt = (
        "You are a competitive programmer. You will be given a problem statement, "
        "please implement a solution in C++. Respect the execution time and memory "
        "limit. Wrap the code in ```cpp and ```."
    )
    with LightCPVerifierJudge(worker=1) as verifier:
        for problem in problems:
            response = client.chat(
                context["model_id"],
                [
                    {
                        "role": "user",
                        "content": system_prompt + "\n\n" + problem.problem_statement,
                    }
                ],
                max_tokens=8192,
            )
            problem.text_response = response
            problem.code = extract_longest_cpp_code(response)
            problem.response_meta = {"model_id": context["model_id"]}
            if problem.code:
                problem.submission_id = verifier.submit(
                    problem.problem_id, SupportedLanguage.CPP, problem.code
                )
            else:
                problem.judge_result = "No Code"
        for problem in problems:
            if problem.judge_result == "No Code":
                continue
            if problem.submission_id is None:
                raise RuntimeError(f"no verifier submission for {problem.problem_id}")
            while True:
                problem.judge_result = verifier.get_result(problem.submission_id)
                if problem.judge_result != "Judging":
                    break
                time.sleep(1)
            if problem.judge_result == "Judge Failed":
                raise RuntimeError(f"official verifier failed for {problem.problem_id}")

    results = [BenchmarkResult(**problem.model_dump()).model_dump() for problem in problems]
    output = Path(raw_directory) / "benchmark_result.json"
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
