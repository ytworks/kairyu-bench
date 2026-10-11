from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kairyu_bench.manifest import load_manifest
from kairyu_bench.terminal_bench4 import (
    TerminalBench4Settings,
    build_job_config,
    build_trial_plan,
    compute_metrics,
    compute_pass_at_k,
    is_final_attempt,
    job_config_failures,
    load_lock,
    removable_images,
    removal_reference,
    resolve_terminal_bench4_conditions,
    select_tasks,
    trial_config_failures,
)

DIGEST = "sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732"


def conditions(agent: str = "claude-code", **env: str) -> dict[str, Any]:
    return resolve_terminal_bench4_conditions(
        load_manifest()["terminal-bench-4"], env, agent
    )


def context(agent: str = "claude-code", limit: int | None = None, **env: str) -> dict[str, Any]:
    entry = load_manifest()["terminal-bench-4"]
    return {
        "run_id": "run-1",
        "benchmark": "terminal-bench-4",
        "endpoint": "http://host.docker.internal:8003/v1",
        "model_id": "kairyu-auto",
        "agent": agent,
        "limit": limit,
        "source": entry["source"],
        "dataset": entry["dataset"],
        "scoring": entry["scoring"],
        "conditions": conditions(agent, **env),
    }


class SettingsTest(unittest.TestCase):
    def test_defaults_follow_the_other_official_harnesses(self) -> None:
        self.assertEqual(
            TerminalBench4Settings.from_env({}, "claude-code"),
            TerminalBench4Settings(workers=4, attempts=1, retries=3, reasoning_effort=None),
        )

    def test_rejects_invalid_values(self) -> None:
        prefix = "KAIRYU_BENCH_TERMINAL_BENCH_4_"
        cases = [
            ({prefix + "WORKERS": "0"}, "WORKERS must be an integer from 1 to 16"),
            ({prefix + "WORKERS": ""}, "WORKERS must be an integer from 1 to 16"),
            ({prefix + "ATTEMPTS": "4"}, "ATTEMPTS must be 1 or 5"),
            ({prefix + "ATTEMPTS": "6"}, "ATTEMPTS must be an integer from 1 to 5"),
            ({prefix + "RETRIES": "17"}, "RETRIES must be an integer from 0 to 16"),
            ({prefix + "REASONING_EFFORT": "none"}, "REASONING_EFFORT for claude-code"),
        ]
        for env, message in cases:
            with self.subTest(env=env):
                with self.assertRaisesRegex(ValueError, message):
                    TerminalBench4Settings.from_env(env, "claude-code")

    def test_reasoning_effort_is_validated_for_the_selected_agent(self) -> None:
        env = {"KAIRYU_BENCH_TERMINAL_BENCH_4_REASONING_EFFORT": "none"}

        self.assertEqual(
            TerminalBench4Settings.from_env(env, "codex").reasoning_effort, "none"
        )

    def test_conditions_record_no_timeout_or_resource_override(self) -> None:
        resolved = conditions(KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS="5")

        self.assertEqual(resolved["dataset_ref"], DIGEST)
        self.assertEqual(resolved["attempts_per_task"], 5)
        self.assertEqual(resolved["timeout_multiplier"], 1.0)
        self.assertIsNone(resolved["agent_setup_timeout_multiplier"])
        self.assertIsNone(resolved["resource_overrides"])
        self.assertEqual(resolved["agent_kwargs"], {"version": "2.1.231"})
        self.assertEqual(resolved["reasoning_effort"], "server_default")
        self.assertIn("AgentTimeoutError", resolved["retry_exclude_exceptions"])
        self.assertEqual(resolved["gpu_policy"], "run-all-gpu-unavailable-scored-zero-v1")


class SelectionAndConfigTest(unittest.TestCase):
    def test_lock_pins_66_tasks_three_gpu_tasks_and_two_offline_verifiers(self) -> None:
        lock = load_lock()

        self.assertEqual(len(lock["tasks"]), 66)
        self.assertEqual(
            lock["gpu_tasks"],
            [
                "terminal-bench/fp8-rmsnorm-gemm",
                "terminal-bench/jax-speedrun-gpu",
                "terminal-bench/math-eval-grader",
            ],
        )
        self.assertEqual(
            lock["offline_verifier_tasks"],
            ["terminal-bench/batched-eval-parity", "terminal-bench/lake-temp-glm"],
        )

    def test_limit_selects_a_sorted_prefix(self) -> None:
        lock = load_lock()

        self.assertEqual(select_tasks(lock, None), sorted(lock["tasks"]))
        self.assertEqual(select_tasks(lock, 2), sorted(lock["tasks"])[:2])
        for limit in (0, -1, True, "2"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                select_tasks(lock, limit)

    def test_claude_code_job_config_mirrors_the_official_run_shape(self) -> None:
        config = build_job_config(
            context(KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS="5"),
            Path("/raw"),
            Path("/raw/host-gateway.json"),
            Path("/cache/harbor-tasks"),
        )

        self.assertEqual(
            config,
            {
                "job_name": "run-1-terminal-bench-4",
                "jobs_dir": "/raw/jobs",
                "n_attempts": 5,
                "n_concurrent_trials": 4,
                "quiet": True,
                "retry": {
                    "max_retries": 3,
                    "exclude_exceptions": [
                        "AgentSafetyRefusalError",
                        "AgentTimeoutError",
                        "ApiUsageLimitError",
                        "RewardFileEmptyError",
                        "RewardFileNotFoundError",
                        "VerifierOutputParseError",
                        "VerifierTimeoutError",
                    ],
                },
                "environment": {
                    "type": "docker",
                    "delete": True,
                    "extra_docker_compose": ["/raw/host-gateway.json"],
                },
                "agents": [
                    {
                        "name": "claude-code",
                        "model_name": "kairyu-auto",
                        "kwargs": {"version": "2.1.231"},
                        "env": {
                            "CLAUDE_CODE_SIMPLE": "1",
                            "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
                            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
                            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                        },
                    }
                ],
                "datasets": [
                    {
                        "name": "terminal-bench/terminal-bench",
                        "ref": DIGEST,
                        "download_dir": "/cache/harbor-tasks",
                    }
                ],
            },
        )
        self.assertEqual(job_config_failures(config), [])

    def test_terminus_uses_the_openai_route_and_limit_selects_task_names(self) -> None:
        config = build_job_config(
            context(
                "terminus-2",
                limit=2,
                KAIRYU_BENCH_TERMINAL_BENCH_4_REASONING_EFFORT="high",
            ),
            Path("/raw"),
            Path("/raw/host-gateway.json"),
            Path("/cache/harbor-tasks"),
        )

        self.assertEqual(
            config["agents"],
            [
                {
                    "name": "terminus-2",
                    "model_name": "openai/kairyu-auto",
                    "kwargs": {
                        "reasoning_effort": "high",
                        "api_base": "http://host.docker.internal:8003/v1",
                    },
                }
            ],
        )
        self.assertEqual(
            config["datasets"][0]["task_names"], select_tasks(load_lock(), 2)
        )

    def test_static_checks_flag_every_forbidden_override(self) -> None:
        trial = {
            "timeout_multiplier": 2.0,
            "agent_setup_timeout_multiplier": 4,
            "agent": {"override_timeout_sec": 10},
            "verifier": {"max_timeout_sec": 10},
            "environment": {"override_cpus": 2},
        }

        self.assertEqual(
            trial_config_failures(trial),
            [
                "timeout_multiplier=2.0",
                "agent_setup_timeout_multiplier=4",
                "agent.override_timeout_sec=10",
                "verifier.max_timeout_sec=10",
                "environment.override_cpus=2",
            ],
        )
        self.assertEqual(trial_config_failures({"timeout_multiplier": 1.0}), [])
        self.assertEqual(
            job_config_failures({"agents": [{"override_setup_timeout_sec": 1}]}),
            ["agent.override_setup_timeout_sec=1"],
        )


class LeaderboardMetricsTest(unittest.TestCase):
    def test_accuracy_ci_and_pass_at_k_match_the_leaderboard(self) -> None:
        by_task = {"a": [1, 1, 0, 0, 0], "b": [1, 1, 1, 1, 1]}

        accuracy, stderr = compute_metrics(by_task)

        self.assertAlmostEqual(accuracy, 70.0)
        self.assertAlmostEqual(round(1.96 * stderr, 2), 24.0)
        self.assertEqual(
            compute_pass_at_k(by_task),
            {"pass_at_2": 0.85, "pass_at_3": 0.95, "pass_at_4": 1.0, "pass_at_5": 1.0},
        )

    def test_missing_rewards_are_failures_and_single_trials_have_no_variance(self) -> None:
        accuracy, stderr = compute_metrics({"a": [None], "b": [0.5]})

        self.assertEqual(accuracy, 50.0)
        self.assertEqual(stderr, 0.0)


class TrialPlanAndRetryTest(unittest.TestCase):
    def test_trial_plan_counts_attempts_in_harbor_order(self) -> None:
        configs = [
            {"task": {"name": "t/a"}, "trial_name": "a__1"},
            {"task": {"name": "t/b"}, "trial_name": "b__1"},
            {"task": {"name": "t/a"}, "trial_name": "a__2"},
        ]

        plan = build_trial_plan(configs, "job")

        self.assertEqual(
            [(slot["task"], slot["attempt_index"]) for slot in plan["slots"]],
            [("t/a", 0), ("t/b", 0), ("t/a", 1)],
        )
        self.assertEqual(plan["slots"][2]["result_path"], "jobs/job/a__2/result.json")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            build_trial_plan(configs + [configs[0]], "job")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            build_trial_plan([{"task": {"name": "t/a"}, "trial_name": "../x"}], "job")

    def test_final_attempt_mirrors_harbor_retry_queue(self) -> None:
        retry = {"max_retries": 3, "exclude_exceptions": ["AgentTimeoutError"]}

        self.assertTrue(is_final_attempt(None, 1, retry))
        self.assertTrue(is_final_attempt("AgentTimeoutError", 1, retry))
        self.assertFalse(is_final_attempt("RuntimeError", 1, retry))
        self.assertFalse(is_final_attempt("RuntimeError", 3, retry))
        self.assertTrue(is_final_attempt("RuntimeError", 4, retry))
        self.assertTrue(
            is_final_attempt("RuntimeError", 1, {"max_retries": 3, "include_exceptions": ["X"]})
        )

    def test_images_are_removed_only_after_every_user_finishes(self) -> None:
        lock = {
            "tasks": {
                "t/a": {"images": ["shared", "a"]},
                "t/b": {"images": ["shared", "b"]},
            }
        }

        self.assertEqual(removable_images(lock, ["t/a", "t/b"], {"t/a"}), ["a"])
        self.assertEqual(
            removable_images(lock, ["t/a", "t/b"], {"t/a", "t/b"}), ["a", "b", "shared"]
        )
        self.assertEqual(
            removal_reference("harborframework/terminal-bench:x-env@sha256:" + "a" * 64),
            "harborframework/terminal-bench@sha256:" + "a" * 64,
        )
        self.assertEqual(removal_reference("redis:7-alpine"), "redis:7-alpine")
        self.assertEqual(
            removal_reference("registry:5000/repo@sha256:" + "b" * 64),
            "registry:5000/repo@sha256:" + "b" * 64,
        )


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class RunFixture:
    """A raw directory with a selection, a trial plan and canonical results."""

    def __init__(self, root: Path, tasks: list[str], attempts: int, agent: str = "claude-code") -> None:
        self.root = root
        self.tasks = tasks
        self.attempts = attempts
        self.agent = agent
        self.lock = load_lock()
        write(
            root / "selected.json",
            {
                "problem_ids": tasks,
                "attempts": attempts,
                "agent": agent,
                "model_id": "kairyu-auto",
                "model_name": "kairyu-auto",
                "agent_kwargs": {"version": "2.1.231"},
                "job_name": "job",
            },
        )
        configs = [
            {"task": {"name": task}, "trial_name": f"{task.split('/')[-1]}__{attempt}"}
            for attempt in range(attempts)
            for task in tasks
        ]
        self.plan = build_trial_plan(configs, "job")
        write(root / "trial-plan.json", self.plan)
        write(root / "runtime.json", {"gpu_available": False})

    def slot(self, task: str, attempt: int) -> dict[str, Any]:
        return next(
            slot
            for slot in self.plan["slots"]
            if slot["task"] == task and slot["attempt_index"] == attempt
        )

    def trial(self, task: str, attempt: int, **overrides: Any) -> dict[str, Any]:
        slot = self.slot(task, attempt)
        payload: dict[str, Any] = {
            "id": f"{task}-{attempt}",
            "task_name": task,
            "trial_name": slot["trial_name"],
            "config": {
                "task": {"name": task, "ref": self.lock["tasks"][task]["digest"]},
                "trial_name": slot["trial_name"],
                "timeout_multiplier": 1.0,
                "agent": {
                    "name": self.agent,
                    "model_name": "kairyu-auto",
                    "kwargs": {"version": "2.1.231"},
                },
                "environment": {"type": "docker"},
                "verifier": {},
            },
            "agent_info": {"name": self.agent, "version": "2.1.231"},
            "agent_setup": {"started_at": "t"},
            "agent_execution": {"started_at": "t"},
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
            "finished_at": "2026-10-11T00:00:00Z",
        }
        payload.update(overrides)
        write(self.root / slot["result_path"], payload)
        return payload


def gpu_error() -> dict[str, Any]:
    return {
        "exception_type": "RuntimeError",
        "exception_message": (
            "Docker compose command failed for environment x. Stderr: Error response "
            'from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]'
        ),
    }


class AccountingTest(unittest.TestCase):
    def setUp(self) -> None:
        from kairyu_bench import terminal_bench4_accounting as accounting

        self.accounting = accounting

    def classify(self, payload: dict[str, Any], gpu_task: bool = False, gpu_available: bool = False):
        return self.accounting.classify_trial(
            payload, gpu_task=gpu_task, gpu_available=gpu_available
        )

    def test_classification_table(self) -> None:
        finished = {"finished_at": "t"}
        reward = lambda value: {"verifier_result": {"rewards": {"reward": value}}}  # noqa: E731
        error = lambda kind: {"exception_info": {"exception_type": kind}}  # noqa: E731
        Outcome = self.accounting.Outcome
        cases = [
            ({}, Outcome(None, "UnfinishedTrial")),
            ({**finished, **reward(1)}, Outcome(1, None)),
            ({**finished, **reward(0.25)}, Outcome(1, None)),
            ({**finished, **reward(0)}, Outcome(0, None)),
            ({**finished, **reward(0), **error("AgentTimeoutError")}, Outcome(0, "AgentTimeoutError")),
            ({**finished, **reward(1), **error("AgentTimeoutError")}, Outcome(1, "AgentTimeoutError")),
            ({**finished, **reward(0), **error("ApiConnectionClosedError")}, Outcome(None, "ApiConnectionClosedError")),
            ({**finished, **reward(1), **error("ApiConnectionClosedError")}, Outcome(1, "ApiConnectionClosedError")),
            ({**finished, **error("AgentTimeoutError")}, Outcome(0, "AgentTimeoutError")),
            ({**finished, **error("ContextWindowExceededError")}, Outcome(0, "ContextWindowExceededError")),
            ({**finished, **error("VerifierTimeoutError")}, Outcome(None, "VerifierTimeoutError")),
            ({**finished, **error("AgentSetupTimeoutError")}, Outcome(None, "AgentSetupTimeoutError")),
            (finished, Outcome(None, "MissingVerifierReward")),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.classify(payload), expected)

    def test_rejects_non_numeric_rewards(self) -> None:
        for value in (True, "1", float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.classify({"finished_at": "t", "verifier_result": {"rewards": {"reward": value}}})

    def test_gpu_unavailable_requires_every_condition(self) -> None:
        base = {"finished_at": "t", "exception_info": gpu_error()}

        self.assertEqual(self.classify(base, gpu_task=True), self.accounting.Outcome(0, "gpu-unavailable"))
        negatives = {
            "not a GPU task": (base, False, False),
            "GPU runtime present": (base, True, True),
            "other exception type": (
                {**base, "exception_info": {**gpu_error(), "exception_type": "ValueError"}},
                True,
                False,
            ),
            "other message": (
                {**base, "exception_info": {**gpu_error(), "exception_message": "port in use"}},
                True,
                False,
            ),
            "agent was set up": ({**base, "agent_setup": {"started_at": "t"}}, True, False),
            "agent ran": ({**base, "agent_execution": {"started_at": "t"}}, True, False),
        }
        for name, (payload, gpu_task, gpu_available) in negatives.items():
            with self.subTest(name=name):
                outcome = self.classify(payload, gpu_task, gpu_available)
                self.assertIsNone(outcome.success)

    def test_completed_five_attempt_summary_matches_the_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tasks = ["terminal-bench/fp8-rmsnorm-gemm", "terminal-bench/batched-eval-parity"]
            run = RunFixture(Path(directory), tasks, 5)
            for attempt in range(5):
                run.trial(
                    tasks[0],
                    attempt,
                    exception_info=gpu_error(),
                    agent_setup=None,
                    agent_execution=None,
                    verifier_result=None,
                )
                run.trial(
                    tasks[1],
                    attempt,
                    verifier_result={"rewards": {"reward": 1.0 if attempt < 2 else 0.0}},
                )

            summary = self.accounting.summarize(Path(directory))

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["scored_attempts"], 10)
        self.assertEqual(summary["passed_attempts"], 2)
        self.assertEqual(summary["gpu_unavailable_attempts"], 5)
        self.assertEqual(summary["evaluated_tasks"], 2)
        self.assertAlmostEqual(summary["accuracy_percent"], 20.0)
        self.assertAlmostEqual(summary["non_gpu_accuracy_percent"], 40.0)
        self.assertEqual(summary["pass_at_k"]["pass_at_2"], 0.35)
        self.assertEqual(summary["leaderboard_equivalent"]["accuracy_percent"], 20.0)
        self.assertIn("5 GPU trial slots scored 0 without a GPU", summary["comparability_notes"])
        self.assertFalse(summary["gpu_available"])

    def test_infrastructure_errors_and_missing_trials_make_the_run_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = "terminal-bench/batched-eval-parity"
            run = RunFixture(Path(directory), [task], 5)
            for attempt in range(3):
                run.trial(task, attempt)
            run.trial(
                task,
                3,
                verifier_result=None,
                exception_info={"exception_type": "EnvironmentStartTimeoutError"},
            )

            summary = self.accounting.summarize(Path(directory))

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["scored_attempts"], 3)
        self.assertEqual(summary["excluded_attempts"], 2)
        self.assertEqual(summary["accuracy_percent"], 100.0)
        self.assertEqual(summary["leaderboard_equivalent"]["accuracy_percent"], 60.0)
        self.assertEqual(
            [error["type"] for error in summary["errors"]],
            ["EnvironmentStartTimeoutError", "MissingTrial"],
        )

    def test_cleanup_errors_block_completion_and_no_scored_trials_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = "terminal-bench/batched-eval-parity"
            run = RunFixture(Path(directory), [task], 1)
            run.trial(task, 0)
            write(Path(directory) / "cleanup-errors" / "image.json", {"image": "x"})

            self.assertEqual(self.accounting.summarize(Path(directory))["status"], "partial")

        with tempfile.TemporaryDirectory() as directory:
            RunFixture(Path(directory), [task], 1)

            summary = self.accounting.summarize(Path(directory))

        self.assertEqual(summary["status"], "failed")
        self.assertIsNone(summary["accuracy_percent"])
        self.assertIsNone(summary["ci95_half_width_percent"])

    def test_identity_violations_raise_instead_of_scoring(self) -> None:
        task = "terminal-bench/batched-eval-parity"
        violations = {
            "digest": lambda p: p["config"]["task"].update(ref="sha256:" + "0" * 64),
            "agent": lambda p: p["agent_info"].update(name="codex"),
            "model": lambda p: p["config"]["agent"].update(model_name="other"),
            "version pin": lambda p: p["config"]["agent"]["kwargs"].update(version="1"),
            "timeout": lambda p: p["config"].update(agent_timeout_multiplier=2.0),
            "resources": lambda p: p["config"]["environment"].update(override_memory_mb=1),
        }
        for name, mutate in violations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                run = RunFixture(Path(directory), [task], 1)
                payload = run.trial(task, 0)
                mutate(payload)
                write(Path(directory) / run.slot(task, 0)["result_path"], payload)

                with self.assertRaises(ValueError):
                    self.accounting.summarize(Path(directory))

    def test_results_outside_the_plan_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = "terminal-bench/batched-eval-parity"
            run = RunFixture(Path(directory), [task], 1)
            run.trial(task, 0)
            write(Path(directory) / "jobs" / "job" / "stray__x" / "result.json", {})

            with self.assertRaisesRegex(ValueError, "outside the trial plan"):
                self.accounting.summarize(Path(directory))

    def test_missing_plan_reports_every_slot_unscored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = "terminal-bench/batched-eval-parity"
            RunFixture(Path(directory), [task], 5)
            (Path(directory) / "trial-plan.json").unlink()

            summary = self.accounting.summarize(Path(directory))

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["excluded_attempts"], 5)
        self.assertEqual(summary["errors"], [{"type": "MissingTrialPlan"}])



class NormalizerTest(unittest.TestCase):
    def normalize(self, raw: Path, **overrides: Any):
        from kairyu_bench.official import normalize_official

        entry = load_manifest()["terminal-bench-4"]
        value = {
            **context(KAIRYU_BENCH_TERMINAL_BENCH_4_ATTEMPTS="5"),
            "schema_version": 1,
            "endpoint_fingerprint": "sha256:test",
            "run_dir": str(raw.parent),
            "result_path": str(raw.parent / "normalized.json"),
            "scoring": entry["scoring"],
            **overrides,
        }
        return normalize_official(value, raw)

    def test_normalized_result_uses_trial_accuracy_and_task_counts(self) -> None:
        task = "terminal-bench/batched-eval-parity"
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            run = RunFixture(raw, [task], 5)
            for attempt in range(5):
                run.trial(task, attempt, verifier_result={"rewards": {"reward": float(attempt < 3)}})

            result = self.normalize(raw).data

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["agent"], "claude-code")
        self.assertEqual(result["score"]["primary"], 60.0)
        self.assertEqual(result["counts"], {"requested": 1, "evaluated": 1})
        self.assertEqual(result["selection"]["problem_ids"], [task])
        self.assertEqual(result["score"]["metrics"]["scored_attempts"], 5)

    def test_normalizer_rejects_another_model_or_attempt_count(self) -> None:
        from kairyu_bench.official import OfficialNormalizationError

        task = "terminal-bench/batched-eval-parity"
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            run = RunFixture(raw, [task], 5)
            for attempt in range(5):
                run.trial(task, attempt)

            with self.assertRaisesRegex(OfficialNormalizationError, "observed model"):
                self.normalize(raw, model_id="other")
            with self.assertRaisesRegex(OfficialNormalizationError, "attempts differ"):
                self.normalize(raw, conditions=conditions())

    def test_partial_runs_explain_the_unscored_slots(self) -> None:
        task = "terminal-bench/batched-eval-parity"
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            run = RunFixture(raw, [task], 5)
            run.trial(task, 0)

            result = self.normalize(raw).data

        self.assertEqual(result["status"], "partial")
        self.assertIn("4 unscored trial slots", result["error"])

if __name__ == "__main__":
    unittest.main()
