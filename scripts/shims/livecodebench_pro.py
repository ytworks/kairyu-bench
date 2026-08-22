from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator

from kairyu_bench.adapter_context import read_context
from kairyu_bench.target import Endpoint, PreflightError, TargetClient


def configured_workers() -> int:
    raw_workers = os.environ.get("KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS", "4")
    try:
        workers = int(raw_workers)
    except ValueError as error:
        raise RuntimeError(
            "KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS must be an integer from 1 to 16"
        ) from error
    if workers < 1 or workers > 16:
        raise RuntimeError(
            "KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS must be an integer from 1 to 16"
        )
    return workers


def configured_retries() -> int:
    raw_retries = os.environ.get("KAIRYU_BENCH_LIVECODEBENCH_PRO_RETRIES", "3")
    try:
        retries = int(raw_retries)
    except ValueError as error:
        raise RuntimeError(
            "KAIRYU_BENCH_LIVECODEBENCH_PRO_RETRIES must be an integer from 0 to 10"
        ) from error
    if retries < 0 or retries > 10:
        raise RuntimeError(
            "KAIRYU_BENCH_LIVECODEBENCH_PRO_RETRIES must be an integer from 0 to 10"
        )
    return retries


def run_problem_pool(
    problems: list[Any],
    workers: int,
    solve_problem: Callable[[int, Any], tuple[int, Any]],
) -> Iterator[tuple[int, Any]]:
    """Run at most ``workers`` problems and refill each slot on completion."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(solve_problem, index, problem)
            for index, problem in enumerate(problems, start=1)
        ]
        for future in as_completed(futures):
            yield future.result()


def write_results(output: Path, completed: dict[int, Any], benchmark_result) -> None:
    results = [
        benchmark_result(**completed[index].model_dump()).model_dump()
        for index in sorted(completed)
    ]
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(results, indent=4, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def verifier_container_base_url(container) -> str:
    """Return the verifier address on the Docker network shared by both containers."""
    container.reload()
    networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
    for network in networks.values():
        address = network.get("IPAddress")
        if address:
            return f"http://{address}:8081"
    raise RuntimeError(
        "LightCPVerifier container has no reachable Docker network address"
    )


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

    verifier_image = os.environ.get("KAIRYU_LIGHTCPVERIFIER_IMAGE")
    if verifier_image:
        LightCPVerifierJudge.IMAGE_NAME = verifier_image

    verifier_host = os.environ.get("KAIRYU_LIGHTCPVERIFIER_HOST")

    class RoutedLightCPVerifierJudge(LightCPVerifierJudge):
        def _start_container(self) -> None:
            super()._start_container()
            if verifier_host == "container":
                self.base_url = verifier_container_base_url(self.container)
            elif verifier_host:
                port = self.container.ports["8081/tcp"][0]["HostPort"]
                self.base_url = f"http://{verifier_host}:{port}"

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

    endpoint = Endpoint.parse(context["endpoint"])
    api_key = os.environ.get("KAIRYU_API_KEY")
    workers = configured_workers()
    retries = configured_retries()
    official_prompt = """
        You are a competitive programmer. You will be given a problem statement, please implement solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted.
        """
    output = Path(raw_directory) / "benchmark_result.json"
    completed_problems: dict[int, Any] = {}
    with RoutedLightCPVerifierJudge(worker=8) as verifier:
        def solve_problem(index: int, problem: Any) -> tuple[int, Any]:
            print(
                f"Generating solution {index}/{len(problems)}: {problem.problem_id}",
                flush=True,
            )
            client = TargetClient(endpoint, api_key=api_key, timeout=1200)
            response = None
            for attempt in range(1, retries + 2):
                try:
                    response = client.chat(
                        context["model_id"],
                        [
                            {
                                "role": "user",
                                "content": official_prompt
                                + problem.problem_statement,
                            }
                        ],
                        max_tokens=None,
                        temperature=None,
                        stream=True,
                    )
                    break
                except (PreflightError, TimeoutError) as error:
                    if attempt > retries:
                        print(
                            f"Generation failed {index}/{len(problems)}: "
                            f"{problem.problem_id}: {error}",
                            flush=True,
                        )
                        break
                    print(
                        f"Retrying solution {index}/{len(problems)}: "
                        f"{problem.problem_id}: attempt {attempt + 1}/{retries + 1}: "
                        f"{error}",
                        flush=True,
                    )
                    time.sleep(min(5 * attempt, 30))

            if response is None:
                problem.text_response = ""
                problem.code = ""
                problem.response_meta = {"model_id": context["model_id"]}
                problem.submission_id = None
                problem.judge_result = "Judge Failed"
                print(
                    f"Judged solution {index}/{len(problems)}: "
                    f"{problem.problem_id}: {problem.judge_result}",
                    flush=True,
                )
                return index, problem
            problem.text_response = response
            problem.code = extract_longest_cpp_code(response)
            problem.response_meta = {"model_id": context["model_id"]}
            if problem.code:
                problem.submission_id = verifier.submit(
                    problem.problem_id, SupportedLanguage.CPP, problem.code
                )
            if problem.submission_id is None:
                problem.judge_result = "Judge Failed"
            else:
                while problem.judge_result == "Judging":
                    problem.judge_result = verifier.get_result(problem.submission_id)
                    if problem.judge_result == "Judging":
                        time.sleep(1)
            print(
                f"Judged solution {index}/{len(problems)}: "
                f"{problem.problem_id}: {problem.judge_result}",
                flush=True,
            )
            return index, problem

        print(
            f"LiveCodeBench Pro workers: {workers}; verifier workers: 8; "
            f"generation retries: {retries}",
            flush=True,
        )
        for index, problem in run_problem_pool(problems, workers, solve_problem):
            completed_problems[index] = problem
            write_results(output, completed_problems, BenchmarkResult)

    write_results(
        output,
        {index: problem for index, problem in enumerate(problems, start=1)},
        BenchmarkResult,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
