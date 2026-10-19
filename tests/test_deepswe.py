from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from kairyu_bench.manifest import load_manifest, select_benchmarks
from kairyu_bench.official import normalize_official


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class DeepSWETest(unittest.TestCase):
    def module(self):
        self.assertIn("deepswe", load_manifest(), "DeepSWE must be a public adapter")
        return importlib.import_module("kairyu_bench.deepswe")

    def test_public_selection_and_settings_keep_existing_invocation(self):
        module = self.module()
        self.assertEqual(
            select_benchmarks("deepswe,gpqa-diamond"), ["gpqa-diamond", "deepswe"]
        )
        settings = module.DeepSWESettings.from_env({})
        self.assertEqual(
            (settings.workers, settings.attempts, settings.retries), (4, 1, 3)
        )
        self.assertIsNone(settings.reasoning_effort)
        self.assertIsNone(settings.api_timeout_seconds)
        for key, value in (
            ("WORKERS", "0"),
            ("WORKERS", "17"),
            ("WORKERS", "1.5"),
            ("WORKERS", ""),
            ("ATTEMPTS", "2"),
            ("RETRIES", "-1"),
            ("REASONING_EFFORT", "high --agent oracle"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                module.DeepSWESettings.from_env({f"KAIRYU_BENCH_DEEPSWE_{key}": value})

    def test_api_timeout_is_optional_bounded_and_recorded_when_configured(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        self.assertNotIn("api_timeout_seconds", module.resolve_deepswe_conditions(entry, {}))
        for seconds in (1, 3600, 10800):
            with self.subTest(seconds=seconds):
                env = {"KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS": str(seconds)}
                self.assertEqual(module.DeepSWESettings.from_env(env).api_timeout_seconds, seconds)
                conditions = module.resolve_deepswe_conditions(entry, env)
                self.assertEqual(conditions["api_timeout_seconds"], seconds)
                self.assertEqual(conditions["timeout_multiplier"], 1)
        for value in ("", "0", "10801", "-1", "1.5", "1; echo unsafe"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "API_TIMEOUT_SECONDS"):
                module.DeepSWESettings.from_env({"KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS": value})

    def test_named_tasks_select_a_recorded_subset_and_reject_unknown_ids(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        env = {"KAIRYU_BENCH_DEEPSWE_TASKS": "task-z, task-a"}
        self.assertEqual(module.DeepSWESettings.from_env(env).tasks, ("task-z", "task-a"))
        self.assertEqual(
            module.resolve_deepswe_conditions(entry, env)["selected_tasks"],
            ["task-z", "task-a"],
        )
        self.assertNotIn("selected_tasks", module.resolve_deepswe_conditions(entry, {}))
        for value in ("task-a,task-a", "task-a,", "../task-a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.DeepSWESettings.from_env({"KAIRYU_BENCH_DEEPSWE_TASKS": value})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("task-z", "task-a", "task-m"):
                task = root / name
                task.mkdir()
                (task / "task.toml").write_text(f'[task]\nname = "datacurve/{name}"\n')
                (task / "instruction.md").write_text("Synthetic task\n")
            self.assertEqual(
                module.select_deepswe_tasks(root, None, ["task-z", "task-a"]),
                ["task-a", "task-z"],
            )
            with self.assertRaisesRegex(ValueError, "not in the dataset"):
                module.select_deepswe_tasks(root, None, ["task-b"])

    def test_mini_swe_agent_conditions_keep_their_existing_keys(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        conditions = module.resolve_deepswe_conditions(entry, {})
        self.assertEqual(
            set(conditions),
            {
                "benchmark_version", "agent_revision", "agent_version",
                "adapter_sha256", "agent_lock_sha256", "model_class", "transport",
                "reasoning_effort", "attempts_per_task", "workers", "max_retries",
                "agent_setup_timeout_multiplier", "timeout_multiplier",
                "error_policy", "cost_policy",
            },
        )
        self.assertEqual(
            module.resolve_deepswe_conditions(
                entry, {"KAIRYU_BENCH_DEEPSWE_AGENT": "mini-swe-agent"}
            ),
            conditions,
        )
        for key in ("CONTEXT_WINDOW", "MAX_TOKENS"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "only to deepseek"):
                module.DeepSWESettings.from_env({f"KAIRYU_BENCH_DEEPSWE_{key}": "200000"})
        with self.assertRaisesRegex(ValueError, "KAIRYU_BENCH_DEEPSWE_AGENT"):
            module.DeepSWESettings.from_env({"KAIRYU_BENCH_DEEPSWE_AGENT": "dsh"})

    def test_deepseek_harness_requires_a_context_window_that_keeps_compaction(self):
        module = self.module()
        dsh = {"KAIRYU_BENCH_DEEPSWE_AGENT": "deepseek-harness"}
        with self.assertRaisesRegex(ValueError, "CONTEXT_WINDOW is required"):
            module.DeepSWESettings.from_env(dsh)
        settings = module.DeepSWESettings.from_env(
            {**dsh, "KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "262144"}
        )
        self.assertEqual(
            (settings.agent, settings.context_window, settings.max_tokens),
            ("deepseek-harness", 262144, 49152),
        )
        # dsh disables automatic compaction unless retain < threshold.
        self.assertTrue(module.dsh_compaction_valid(127172, 49152))
        self.assertFalse(module.dsh_compaction_valid(127171, 49152))
        for env, message in (
            ({"KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "127171"}, "too small"),
            ({"KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "1e6"}, "CONTEXT_WINDOW"),
            (
                {"KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "262144", "KAIRYU_BENCH_DEEPSWE_MAX_TOKENS": "0"},
                "MAX_TOKENS",
            ),
            (
                {"KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "262144", "KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT": "xhigh"},
                "for deepseek-harness",
            ),
            (
                {"KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "262144", "KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS": "3600"},
                "only to mini-swe-agent",
            ),
        ):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, message):
                module.DeepSWESettings.from_env({**dsh, **env})
        with self.assertRaisesRegex(ValueError, "for mini-swe-agent"):
            module.DeepSWESettings.from_env({"KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT": "off"})

    def test_deepseek_harness_conditions_record_the_locked_minimal_agent(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        env = {
            "KAIRYU_BENCH_DEEPSWE_AGENT": "deepseek-harness",
            "KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "262144",
            "KAIRYU_BENCH_DEEPSWE_ATTEMPTS": "4",
        }
        conditions = module.resolve_deepswe_conditions(entry, env)
        self.assertEqual(conditions["agent"], "deepseek-harness")
        self.assertEqual(conditions["agent_version"], "0.2.1rc2")
        self.assertEqual(
            conditions["agent_revision"], "687f282e8625c6b8d11ff58211a90524be2e4317"
        )
        self.assertEqual(conditions["agent_profile"], "sdk-minimal")
        self.assertEqual(conditions["agent_provider"], "deepseek-official")
        self.assertEqual(conditions["transport"], "anthropic_messages")
        self.assertEqual(conditions["reasoning_effort"], "agent_default")
        self.assertEqual(
            conditions["disabled_profile_rows"],
            ["session-log-deepseek", "plugin-package-inventory-deepseek"],
        )
        self.assertEqual(
            (conditions["context_window"], conditions["max_tokens"]), (262144, 49152)
        )
        self.assertEqual(conditions["attempts_per_task"], 4)
        self.assertEqual(conditions["timeout_multiplier"], 1)
        self.assertNotIn("model_class", conditions)
        self.assertEqual(
            module.resolve_deepswe_conditions(
                entry, {**env, "KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT": "max"}
            )["reasoning_effort"],
            "max",
        )
        lock = json.loads(module.DSH_LOCK.read_text())
        self.assertTrue(all("--hash=sha256:" in line for line in lock["requirements"]))
        self.assertIn(
            "deepseek-harness-runtime-bin==0.2.1rc2 "
            "--hash=sha256:0a0a4eb85a22cd9339972e6a5ced77a27ec791353a5b220e3e10b8cdbbdeca39",
            lock["requirements"][1],
        )

    def test_limit_selects_sorted_prefix_and_ignores_non_tasks(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("task-z", "task-a", "task-m"):
                task = root / name
                task.mkdir()
                (task / "task.toml").write_text(f'[task]\nname = "datacurve/{name}"\n')
                (task / "instruction.md").write_text("Synthetic task\n")
            (root / "README.md").write_text("Not a task\n")
            self.assertEqual(module.select_deepswe_tasks(root, 2), ["task-a", "task-m"])
            self.assertEqual(
                module.select_deepswe_tasks(root, None), ["task-a", "task-m", "task-z"]
            )

    def fixture(self, root: Path, attempts: int = 4):
        module = self.module()
        write_json(
            root / "selected.json",
            {
                "problem_ids": ["task-a", "task-b"],
                "model_id": "org/model",
                "attempts": attempts,
                "agent_version": "2.4.6",
            },
        )
        module.write_trial_plan(root, ["task-a", "task-b"], attempts)
        for repeat in range(attempts):
            for task in ("task-a", "task-b"):
                self.trial(
                    root,
                    task,
                    repeat,
                    int((task, repeat) in {("task-a", 0), ("task-b", 1)}),
                )

    def trial(
        self,
        root: Path,
        task: str,
        repeat: int,
        reward: object,
        exception: str | None = None,
    ):
        payload = {
            "id": f"{task}-{repeat}",
            "task_name": "datacurve/" + task,
            "trial_name": f"{task}__random",
            "agent_info": {
                "name": "mini-swe-agent",
                "version": "2.4.6",
                "model_info": {
                    "name": "org/model",
                    "provider": "openai",
                },
            },
            "verifier_result": {
                "rewards": {"reward": reward, "test_pass_fraction": 0.9}
            },
            "exception_info": {"exception_type": exception} if exception else None,
            "finished_at": "2026-09-10T00:00:00Z",
        }
        path = root / "trials" / f"repeat-{repeat}" / task / "result.json"
        write_json(path, payload)
        return path

    def test_repeats_use_attempt_pass_rate_not_best_of_four(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            summary = module.summarize_deepswe(root)
        self.assertEqual(summary["problem_ids"], ["task-a", "task-b"])
        self.assertEqual(summary["evaluated_tasks"], 2)
        self.assertEqual(summary["scored_attempts"], 8)
        self.assertEqual(summary["passed_attempts"], 2)
        self.assertEqual(summary["pass_at_1_percent"], 25.0)
        self.assertEqual(summary["pass_at_4_percent"], 100.0)
        self.assertEqual(summary["repeat_scores"], [50.0, 50.0, 0.0, 0.0])
        self.assertEqual(summary["status"], "completed")

    def test_missing_trial_keeps_selected_denominator_and_partial_status(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "trials/repeat-3/task-b/result.json").unlink()
            summary = module.summarize_deepswe(root)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["requested_attempts"], 8)
        self.assertEqual(summary["scored_attempts"], 7)
        self.assertEqual(summary["evaluated_tasks"], 1)
        self.assertIsNone(summary["pass_at_4_percent"])
        self.assertIsNone(summary["ci_half_percent"])

    def test_model_timeout_scores_zero_but_verifier_timeout_is_excluded(self):
        module = self.module()
        for exception, scored, status in (
            ("AgentTimeoutError", 2, "completed"),
            ("ContextWindowExceededError", 2, "completed"),
            ("VerifierTimeoutError", 1, "partial"),
            ("RateLimitError", 1, "partial"),
            ("UnclassifiedError", 1, "partial"),
        ):
            with (
                self.subTest(exception=exception),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.fixture(root, attempts=1)
                self.trial(root, "task-a", 0, None, exception)
                summary = module.summarize_deepswe(root)
                self.assertEqual(summary["scored_attempts"], scored)
                self.assertEqual(summary["status"], status)
                self.assertEqual(summary["passed_attempts"], 0)

    def test_corrupt_reward_or_identity_is_not_silently_scored(self):
        module = self.module()
        for mutation in ("fraction", "nan", "bool", "string", "agent", "model", "task"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.fixture(root, attempts=1)
                path = root / "trials/repeat-0/task-a/result.json"
                payload = json.loads(path.read_text())
                if mutation in {"fraction", "nan", "bool", "string"}:
                    payload["verifier_result"]["rewards"]["reward"] = {
                        "fraction": 0.5,
                        "nan": float("nan"),
                        "bool": True,
                        "string": "1",
                    }[mutation]
                elif mutation == "agent":
                    payload["agent_info"]["name"] = "oracle"
                elif mutation == "model":
                    payload["agent_info"]["model_info"]["name"] = "other-model"
                else:
                    payload["task_name"] = "task-not-selected"
                write_json(path, payload)
                with self.assertRaises(ValueError):
                    module.summarize_deepswe(root)

    def dsh_fixture(self, root: Path, statuses: dict[str, object]) -> None:
        write_json(
            root / "selected.json",
            {
                "problem_ids": sorted(statuses),
                "model_id": "org/model",
                "attempts": 1,
                "agent": "deepseek-harness",
                "agent_version": "0.2.1rc2",
            },
        )
        self.module().write_trial_plan(root, sorted(statuses), 1)
        for task, status in statuses.items():
            path = self.trial(root, task, 0, 1)
            payload = json.loads(path.read_text())
            payload["agent_info"] = {
                "name": "deepseek-harness",
                "version": "0.2.1rc2",
                "model_info": {"name": "org/model", "provider": "deepseek-official"},
            }
            if status is not None:
                payload["agent_result"] = {"metadata": {"dsh_exit_status": status}}
            write_json(path, payload)

    def test_deepseek_harness_exit_statuses_follow_the_error_policy(self):
        module = self.module()
        statuses = {
            "task-a": "completed",
            "task-b": "max-tokens",
            "task-c": "ContextWindowExceededError",
            "task-d": "provider:SERVER",
            "task-e": "RuntimeError:TransportClosedError",
            "task-f": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.dsh_fixture(root, statuses)
            summary = module.summarize_deepswe(root)
        self.assertEqual(summary["agent"], "deepseek-harness")
        self.assertEqual(summary["scored_attempts"], 3)
        self.assertEqual(summary["passed_attempts"], 2)
        self.assertEqual(
            {error["task_id"]: error["type"] for error in summary["errors"]},
            {
                "task-d": "provider:SERVER",
                "task-e": "RuntimeError:TransportClosedError",
                "task-f": "MissingAgentExitStatus",
            },
        )
        self.assertEqual(summary["status"], "partial")

    def test_deepseek_harness_selection_rejects_other_agent_identities(self):
        module = self.module()
        for field, value in (
            ("name", "mini-swe-agent"),
            ("version", "2.4.6"),
            ("provider", "openai"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.dsh_fixture(root, {"task-a": "completed"})
                path = root / "trials/repeat-0/task-a/result.json"
                payload = json.loads(path.read_text())
                if field == "provider":
                    payload["agent_info"]["model_info"]["provider"] = value
                else:
                    payload["agent_info"][field] = value
                write_json(path, payload)
                with self.assertRaises(ValueError):
                    module.summarize_deepswe(root)

    def test_selection_without_agent_is_a_mini_swe_agent_run(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, attempts=1)
            summary = module.summarize_deepswe(root)
        self.assertEqual(summary["agent"], "mini-swe-agent")
        self.assertEqual(summary["status"], "completed")

    def test_no_scored_trials_has_no_score(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, attempts=1)
            for task in ("task-a", "task-b"):
                self.trial(root, task, 0, None, "ProviderError")
            summary = module.summarize_deepswe(root)
        self.assertEqual(summary["status"], "failed")
        self.assertIsNone(summary["pass_at_1_percent"])

    def test_official_normalizer_preserves_selected_tasks_when_all_trials_fail(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, attempts=1)
            for task in ("task-a", "task-b"):
                self.trial(root, task, 0, None, "ProviderError")
            context = {
                "benchmark": "deepswe",
                "run_id": "test",
                "run_dir": str(root),
                "model_id": "org/model",
                "agent": "mini-swe-agent",
                "limit": 2,
                "endpoint_fingerprint": "sha256:test",
                "source": entry["source"],
                "dataset": entry["dataset"],
                "scoring": entry["scoring"],
                "conditions": module.resolve_deepswe_conditions(entry, {}),
            }
            result = normalize_official(context, root).data
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["selection"]["problem_ids"], ["task-a", "task-b"])
        self.assertEqual(result["counts"], {"requested": 2, "evaluated": 0})
        self.assertIsNone(result["score"]["primary"])

    def test_official_normalizer_reports_the_deepseek_harness_agent(self):
        module = self.module()
        entry = load_manifest()["deepswe"]
        env = {
            "KAIRYU_BENCH_DEEPSWE_AGENT": "deepseek-harness",
            "KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW": "262144",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.dsh_fixture(root, {"task-a": "completed", "task-b": "completed"})
            context = {
                "benchmark": "deepswe",
                "run_id": "test",
                "run_dir": str(root),
                "model_id": "org/model",
                "agent": "deepseek-harness",
                "limit": 2,
                "endpoint_fingerprint": "sha256:test",
                "source": entry["source"],
                "dataset": entry["dataset"],
                "scoring": entry["scoring"],
                "conditions": module.resolve_deepswe_conditions(entry, env),
            }
            result = normalize_official(context, root).data
            context["agent"] = "mini-swe-agent"
            with self.assertRaisesRegex(ValueError, "expected 'mini-swe-agent'"):
                normalize_official(context, root)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["agent"], "deepseek-harness")
        self.assertEqual(result["score"]["primary"], 100.0)


if __name__ == "__main__":
    unittest.main()
