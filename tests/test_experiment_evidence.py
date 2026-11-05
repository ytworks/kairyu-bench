from __future__ import annotations

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kairyu_bench.deepswe import summarize_deepswe, write_trial_plan
from kairyu_bench.wsp.report import compare

SCRIPTS = Path(__file__).resolve().parents[1] / "docs" / "experiments" / "scripts"


def module(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), SCRIPTS / f"{name}.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


WSP = module("wsp-run-evidence")
DEEPSWE = module("deepswe-recorded-usage")
SECRET = "must-not-appear-prompt-reasoning-api-key"
UUID_A = "00000000-0000-0000-0000-000000000001"
UUID_B = "00000000-0000-0000-0000-000000000002"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def wsp_run(root, name, scores, *, tier=10, label="base", split="dev"):
    path = root / name
    path.mkdir()
    save(
        path / "run.json",
        {
            "status": "completed",
            "tier": tier,
            "split": split,
            "manifest_sha256": "manifest",
            "label": label,
            "model_id": "model",
            "api_key": SECRET,
        },
    )
    rows = []
    for key, score in scores.items():
        rows.append(
            {
                "id": f"scicode:{key}",
                "source": "scicode",
                "area": "code",
                "kind": "chat",
                "score": score,
                "prompt": SECRET,
                "outcome": {
                    "samples": [
                        {
                            "calls": [
                                {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 3,
                                    "seconds": 0.2,
                                    "content": SECRET,
                                    "reasoning_content": SECRET,
                                },
                                {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 4,
                                    "seconds": 0.3,
                                },
                            ],
                            "completion_tokens": 7,
                            "seconds": 0.5,
                            "content": SECRET,
                        }
                    ]
                },
            }
        )
    (path / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def trial(identity=UUID_A, reward=1):
    return {
        "id": identity,
        "task_name": "datacurve/task-a",
        "task_checksum": "checksum",
        "finished_at": "2026-01-01T00:00:10Z",
        "agent_info": {
            "name": "mini-swe-agent",
            "version": "test",
            "model_info": {"name": "model", "provider": "openai"},
        },
        "agent_result": {
            "n_agent_steps": 2,
            "n_input_tokens": 10,
            "n_output_tokens": 4,
            "metadata": {"mini_exit_status": "Submitted"},
            "reasoning_content": SECRET,
        },
        "agent_execution": {
            "started_at": "2026-01-01T00:00:01Z",
            "finished_at": "2026-01-01T00:00:04Z",
        },
        "verifier_result": {"rewards": {"reward": reward}},
    }


def deepswe_run(root, payload=None):
    path = root / "run"
    raw = path / "raw" / "deepswe"
    save(
        path / "run.json",
        {
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:10Z",
            "api_key": SECRET,
        },
    )
    save(
        raw / "selected.json",
        {
            "problem_ids": ["task-a"],
            "attempts": 1,
            "model_id": "model",
            "agent_version": "test",
        },
    )
    write_trial_plan(raw, ["task-a"], 1)
    if payload is not None:
        save(raw / "trials/repeat-0/task-a/result.json", payload)
    return path, raw


class WSPRunEvidenceTest(unittest.TestCase):
    def test_official_comparison_fractional_changes_and_repeat_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = wsp_run(root, "base", {"a": 0.25, "b": 1})
            c = wsp_run(root, "candidate", {"a": 0.4, "b": 0}, label="edited")
            br = wsp_run(root, "base-repeat", {"a": 0.3, "b": 1})
            cr = wsp_run(root, "candidate-repeat", {"a": 0.45, "b": 1}, label="edited")
            report = WSP.build_evidence(
                b, c, baseline_repeats=(br,), candidate_repeats=(cr,), partial=True
            )
            self.assertEqual(report["official_comparison"], compare(b, c, partial=True))
            gain, loss = report["capability_items"]
            self.assertEqual(gain["change"], "gain")
            self.assertFalse(gain["crossed_0_5"])
            self.assertTrue(gain["persistent_gain"])
            self.assertFalse(loss["persistent_loss"])
            self.assertTrue(loss["intermittent_loss"])
            self.assertNotIn(SECRET, json.dumps(report))

    def test_historical_tier_is_range_only_and_does_not_change_official_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = wsp_run(root, "base", {"a": 0})
            c = wsp_run(root, "candidate", {"a": 1}, label="edited")
            history = wsp_run(root, "historical", {"a": 1, "extra": 0}, tier=60)
            report = WSP.build_evidence(
                b, c, historical_dev_refs=(history,), partial=True
            )
            self.assertEqual(report["official_comparison"], compare(b, c, partial=True))
            row = report["capability_items"][0]
            self.assertTrue(row["candidate_above_all_observed_baselines"])
            self.assertFalse(row["above_all_including_historical"])
            self.assertEqual(row["historical_dev_observations"][0]["tier"], 60)

    def test_repeat_identity_cohort_and_holdout_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = wsp_run(root, "base", {"a": 0})
            c = wsp_run(root, "candidate", {"a": 1}, label="edited")
            for name, kwargs, scores in [
                ("wrong-weight", {"label": "another"}, {"a": 0}),
                ("wrong-tier", {"tier": 30}, {"a": 0}),
                ("wrong-cohort", {}, {"different": 0}),
                ("holdout", {"split": "test"}, {"a": 0}),
            ]:
                reference = wsp_run(root, name, scores, **kwargs)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    WSP.build_evidence(
                        b, c, baseline_repeats=(reference,), partial=True
                    )
            with self.assertRaises(ValueError):
                WSP.build_evidence(b, c, baseline_repeats=(b,), partial=True)

    def test_unknown_usage_and_scores_are_null_not_zero(self):
        usage = WSP.usage({"samples": [{"reward": 1, "seconds": 12, "trial": "task"}]})
        self.assertIsNone(usage["output_tokens"]["total"])
        self.assertIsNone(usage["model_calls"]["total"])
        self.assertEqual(usage["sample_elapsed_seconds"]["total"], 12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = wsp_run(root, "base", {"a": None})
            c = wsp_run(root, "candidate", {"a": 1})
            row = WSP.build_evidence(b, c, partial=True)["capability_items"][0]
            self.assertIsNone(row["delta"])
            self.assertIsNone(row["candidate_above_all_observed_baselines"])

    def test_nested_calls_are_not_double_counted_as_aggregate_sample(self):
        record = WSP.usage(
            {
                "samples": [
                    {
                        "completion_tokens": 7,
                        "seconds": 0.5,
                        "calls": [
                            {
                                "completion_tokens": 3,
                                "prompt_tokens": 10,
                                "seconds": 0.2,
                            },
                            {
                                "completion_tokens": 4,
                                "prompt_tokens": 20,
                                "seconds": 0.3,
                            },
                        ],
                    }
                ]
            }
        )
        self.assertEqual(record["model_calls"]["total"], 2)
        self.assertEqual(record["output_tokens"]["total"], 7)
        self.assertEqual(record["input_tokens"]["total"], 30)
        self.assertAlmostEqual(record["model_seconds_sum"]["total"], 0.5)

    def test_fresh_output_protects_input_existing_hardlink_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = wsp_run(root, "input", {"a": 1})
            with self.assertRaises(ValueError):
                WSP.write_fresh(source / "new.json", {}, [source])
            output = root / "evidence.json"
            WSP.write_fresh(output, {"ok": True}, [source])
            with self.assertRaises(FileExistsError):
                WSP.write_fresh(output, {}, [source])
            hardlink = root / "hardlink.json"
            hardlink.hardlink_to(source / "run.json")
            with self.assertRaises(FileExistsError):
                WSP.write_fresh(hardlink, {}, [source])
            link = root / "broken.json"
            link.symlink_to(root / "absent")
            with self.assertRaises(FileExistsError):
                WSP.write_fresh(link, {}, [source])
            self.assertEqual(json.loads(output.read_text()), {"ok": True})


class DeepSWERecordedUsageTest(unittest.TestCase):
    def test_same_uuid_copy_once_and_new_archived_uuid_adds_only_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = deepswe_run(Path(directory), trial())
            save(raw / f"retries/repeat-0/task-a/{UUID_A}/result.json", trial())
            save(
                raw / f"retries/repeat-0/task-a/{UUID_B}/result.json", trial(UUID_B, 0)
            )
            # jobs copies are irrelevant to the canonical+retry union.
            save(raw / "jobs/copy/result.json", trial())
            report = DEEPSWE.collect(path)
            self.assertEqual(report["official_summary"], summarize_deepswe(raw))
            self.assertEqual(report["official_summary"]["passed_attempts"], 1)
            self.assertEqual(report["recorded_trial_count"], 2)
            self.assertEqual(report["archived_only_trial_count"], 1)
            self.assertEqual(report["usage"]["output_tokens"]["total"], 8)
            self.assertEqual(report["agent_seconds"]["total"], 6)
            self.assertEqual(report["wall_seconds"], 10)
            self.assertNotIn(SECRET, json.dumps(report))

    def test_same_uuid_usage_time_or_checksum_conflict_rejected(self):
        for field in ("usage", "time", "checksum"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path, raw = deepswe_run(Path(directory), trial())
                altered = copy.deepcopy(trial())
                if field == "usage":
                    altered["agent_result"]["n_output_tokens"] = 99
                elif field == "time":
                    altered["agent_execution"]["finished_at"] = "2026-01-01T00:00:05Z"
                else:
                    altered["task_checksum"] = "other"
                save(raw / f"retries/repeat-0/task-a/{UUID_A}/result.json", altered)
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    DEEPSWE.collect(path)

    def test_missing_canonical_and_usage_keep_null_totals_and_partial_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = deepswe_run(Path(directory))
            archived = trial(UUID_B)
            del archived["agent_result"]["n_output_tokens"]
            save(raw / f"retries/repeat-0/task-a/{UUID_B}/result.json", archived)
            report = DEEPSWE.collect(path)
            self.assertEqual(report["official_summary"]["scored_attempts"], 0)
            self.assertEqual(report["official_summary"]["excluded_attempts"], 1)
            self.assertIsNone(report["official_summary"]["pass_at_1_percent"])
            self.assertIsNone(report["usage"]["output_tokens"]["total"])
            self.assertEqual(report["usage"]["output_tokens"]["expected_trials"], 2)
            self.assertEqual(report["usage"]["output_tokens"]["missing_trials"], 2)

    def test_archive_slot_uuid_and_escape_rejected(self):
        for location in ("wrong-task", "wrong-uuid", "escape", "malformed"):
            with (
                self.subTest(location=location),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                path, raw = deepswe_run(root, trial())
                if location == "wrong-task":
                    target = raw / f"retries/repeat-0/task-b/{UUID_B}/result.json"
                elif location == "wrong-uuid":
                    target = raw / f"retries/repeat-0/task-a/{UUID_A}/result.json"
                elif location == "malformed":
                    target = raw / "retries/result.json"
                else:
                    outside = root / "outside.json"
                    save(outside, trial(UUID_B))
                    target = raw / f"retries/repeat-0/task-a/{UUID_B}/result.json"
                    target.parent.mkdir(parents=True)
                    target.symlink_to(outside)
                if location != "escape":
                    save(target, trial(UUID_B))
                with self.assertRaises(ValueError):
                    DEEPSWE.collect(path)

    def test_no_fixed_113_requirement_and_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = deepswe_run(root, trial())
            report = DEEPSWE.collect(path)
            self.assertEqual(report["official_summary"]["requested_attempts"], 1)
            output = root / "usage.json"
            DEEPSWE.write_fresh(output, report, path)
            with self.assertRaises(FileExistsError):
                DEEPSWE.write_fresh(output, report, path)
            with self.assertRaises(ValueError):
                DEEPSWE.write_fresh(path / "usage.json", report, path)


class DeepSWEAgentApiMetadataTest(unittest.TestCase):
    @staticmethod
    def trajectory(result, calls, *, suffix=b',"messages":["body must not be read"]}'):
        path = result.parent / "agent" / "mini-swe-agent.trajectory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {"model_stats": {"api_calls": calls}, "ignored": SECRET + ' {"}\\'}
        prefix = b'{"info":' + json.dumps(info).encode()
        path.write_bytes(prefix + suffix)
        return path, prefix

    def test_opt_in_counts_api_counter_once_per_uuid_without_changing_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            run, raw = deepswe_run(Path(directory), trial())
            canonical = raw / "trials/repeat-0/task-a/result.json"
            self.trajectory(canonical, 5)
            archive = raw / f"retries/repeat-0/task-a/{UUID_B}/result.json"
            save(archive, trial(UUID_B, 0))
            self.trajectory(archive, 4)
            job = raw / "jobs/repeat-0/task-a__copy/result.json"
            save(job, trial())
            self.trajectory(job, 5)
            with patch.object(DEEPSWE, "agent_api_prefix", side_effect=AssertionError("opt in only")):
                default = DEEPSWE.collect(run)
            self.assertNotIn("agent_api_calls", default["usage"])
            report = DEEPSWE.collect(run, agent_api_metadata=True)
            self.assertEqual(report["official_summary"], default["official_summary"])
            self.assertEqual(report["usage"]["agent_steps"]["total"], 4)
            self.assertEqual(report["usage"]["agent_api_calls"]["total"], 9)
            self.assertEqual(report["usage"]["agent_api_calls"]["measured_trials"], 2)
            self.assertEqual(report["recorded_trial_count"], 2)
            self.assertNotIn(SECRET, json.dumps(report))

    def test_prefix_stops_exactly_before_messages_and_handles_escaped_braces(self):
        with tempfile.TemporaryDirectory() as directory:
            run, raw = deepswe_run(Path(directory), trial())
            trajectory, prefix = self.trajectory(
                raw / "trials/repeat-0/task-a/result.json", 0, suffix=b'\xffINVALID BODY'
            )
            original = Path.open
            read_count = []

            class PrefixOnly(io.BytesIO):
                def read(self, size=-1):
                    self_test.assertEqual(size, 1)
                    self_test.assertLess(self.tell(), len(prefix))
                    value = super().read(size)
                    read_count.append(len(value))
                    return value

            self_test = self

            def opened(path, *args, **kwargs):
                if path == trajectory:
                    return PrefixOnly(prefix)
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", opened):
                report = DEEPSWE.collect(run, agent_api_metadata=True)
            self.assertEqual(sum(read_count), len(prefix))
            self.assertEqual(report["usage"]["agent_api_calls"]["total"], 0)

    def test_missing_invalid_unsupported_and_bounded_prefixes_remain_unknown(self):
        examples = [
            b'{"messages":INVALID BODY', b'{"info":', b'{"info":[]}',
            b'{"info":{"model_stats":{"api_calls":true}}}',
            b'{"info":{"model_stats":{"api_calls":1.0}}}',
            b'{"info":{"model_stats":{"api_calls":-1}}}',
            b'{"info":{"model_stats":{}}}', b'{"info":{"huge":"' + b'x' * 100,
        ]
        for content in [None, *examples]:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                run, raw = deepswe_run(Path(directory), trial())
                if content is not None:
                    target, _ = self.trajectory(raw / "trials/repeat-0/task-a/result.json", 9)
                    target.write_bytes(content)
                with patch.object(DEEPSWE, "MAX_INFO_PREFIX_BYTES", 80):
                    report = DEEPSWE.collect(run, agent_api_metadata=True)
                result = report["usage"]["agent_api_calls"]
                self.assertIsNone(result["total"])
                self.assertEqual(result["sum_recorded"], 0)
                self.assertEqual(result["missing_trials"], 1)
                self.assertEqual(report["usage"]["agent_steps"]["total"], 2)

    def test_job_uuid_counter_checksum_task_and_repeat_conflicts_are_rejected(self):
        for kind in ("counter", "checksum", "missing_checksum", "task", "repeat"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                payload = trial()
                if kind == "missing_checksum":
                    payload["task_checksum"] = None
                run, raw = deepswe_run(Path(directory), payload)
                self.trajectory(raw / "trials/repeat-0/task-a/result.json", 5)
                job = raw / f"jobs/repeat-{1 if kind == 'repeat' else 0}/task-a__job/result.json"
                payload = copy.deepcopy(payload)
                if kind == "checksum":
                    payload["task_checksum"] = "other"
                if kind == "task":
                    payload["task_name"] = "datacurve/other"
                save(job, payload)
                self.trajectory(job, 6 if kind == "counter" else 5)
                with self.assertRaises(ValueError):
                    DEEPSWE.collect(run, agent_api_metadata=True)

    def test_unknown_job_uuid_is_not_added_and_missing_canonical_slot_stays_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            run, raw = deepswe_run(Path(directory))
            archive = raw / f"retries/repeat-0/task-a/{UUID_B}/result.json"
            save(archive, trial(UUID_B))
            self.trajectory(archive, 7)
            job = raw / "jobs/repeat-0/task-a__unfinished/result.json"
            save(job, trial())
            self.trajectory(job, 99)
            report = DEEPSWE.collect(run, agent_api_metadata=True)
            result = report["usage"]["agent_api_calls"]
            self.assertIsNone(result["total"])
            self.assertEqual(result["sum_recorded"], 7)
            self.assertEqual(result["expected_trials"], 2)
            self.assertEqual(result["measured_trials"], 1)
            self.assertEqual(report["official_summary"]["scored_attempts"], 0)

    def test_trajectory_escape_and_changed_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, raw = deepswe_run(root, trial())
            target, _ = self.trajectory(raw / "trials/repeat-0/task-a/result.json", 3)
            outside = root / "outside.json"
            outside.write_bytes(target.read_bytes())
            target.unlink()
            target.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                DEEPSWE.collect(run, agent_api_metadata=True)
            target.unlink()
            target.write_bytes(outside.read_bytes())
            real_state = DEEPSWE.file_state
            calls = 0

            def changed(path):
                nonlocal calls
                value = real_state(path)
                if path == target:
                    calls += 1
                    if calls > 1:
                        return (*value[:-1], value[-1] + 1)
                return value

            with patch.object(DEEPSWE, "file_state", changed):
                with self.assertRaisesRegex(ValueError, "changed"):
                    DEEPSWE.collect(run, agent_api_metadata=True)


if __name__ == "__main__":
    unittest.main()
