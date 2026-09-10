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


if __name__ == "__main__":
    unittest.main()
