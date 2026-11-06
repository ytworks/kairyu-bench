from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kairyu_bench.wsp.catalog import AREA_KEYS, SOURCES
from kairyu_bench.wsp.manifest import Item
from kairyu_bench.wsp.report import compare
from kairyu_bench.wsp.run import allocate_workers
from kairyu_bench.wsp.select import Measured, assign_splits, assign_tiers, split_order
from kairyu_bench.wsp.sources import Context, Loader, adapters
from kairyu_bench.wsp.sources.agents import _swe_bench_pro_outcomes

# One capability source per area, for runs that cover all fourteen.
AREA_SOURCE = {area: next(source.name for source in SOURCES if source.area == area) for area in AREA_KEYS}


def _item(source: str, samples: int = 1) -> Item:
    return Item(f"{source}:x", source, "dev", 10, 16384, samples, "", 0)


def _reply(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {"content": content, "finish_reason": finish_reason, "completion_tokens": 10, "prompt_tokens": 10}


class SplitTest(unittest.TestCase):
    def test_one_problem_in_several_sources_never_spans_dev_and_test(self) -> None:
        candidates = []
        for number in range(40):
            # MMLU-Pro and its Japanese translation share question ids.
            candidates.append((f"mmlu-pro:{number}", "law" if number % 2 else "math", f"mmlu:{number}"))
            candidates.append((f"mmlu-prox-ja:{number}", "law", f"mmlu:{number}"))
        candidates += [(f"gpqa-diamond:{n:04d}", "Physics", f"gpqa-diamond:{n:04d}") for n in range(10)]

        split_of = assign_splits(candidates)

        for number in range(40):
            self.assertEqual(split_of[f"mmlu-pro:{number}"], split_of[f"mmlu-prox-ja:{number}"])
        english = [item for item, _, _ in candidates if item.startswith("mmlu-pro:")]
        orders = split_order("mmlu-pro", [(item, "s") for item in english], split_of)
        self.assertEqual(len(orders["dev"]), len(orders["test"]))
        self.assertEqual(sum(split_of[f"gpqa-diamond:{n:04d}"] == "dev" for n in range(10)), 5)


class SampleConcurrencyTest(unittest.TestCase):
    def test_samples_are_sent_at_once_within_the_request_bound(self) -> None:
        import threading
        from unittest import mock

        from kairyu_bench.wsp.client import ChatClient
        from kairyu_bench.wsp.sources import run_chat

        for bound, expected in ((4, 4), (2, 2)):
            with self.subTest(bound=bound):
                state = {"in_flight": 0, "peak": 0}
                changed = threading.Condition()
                release = threading.Event()

                def fake_post(url: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
                    with changed:
                        state["in_flight"] += 1
                        state["peak"] = max(state["peak"], state["in_flight"])
                        changed.notify_all()
                    self.assertTrue(release.wait(30), "requests were never released")  # hang bound
                    with changed:
                        state["in_flight"] -= 1
                    return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}}

                client = ChatClient("http://api.test/v1", "m", concurrency=bound)
                item = Item("aime-2026:1", "aime-2026", "dev", 10, 16, 4, "", 0)
                replies: list[Any] = []
                with mock.patch("kairyu_bench.wsp.client._post", fake_post):
                    worker = threading.Thread(
                        target=lambda: replies.append(run_chat(Context(chat=client, logprob=None), item, []))
                    )
                    worker.start()
                    with changed:
                        self.assertTrue(changed.wait_for(lambda: state["in_flight"] == expected, 30))
                    release.set()
                    worker.join(30)

                self.assertEqual(state["peak"], expected)
                self.assertEqual(len(replies[0]["samples"]), 4)


class TierTest(unittest.TestCase):
    def test_tiers_fill_smallest_first_within_cumulative_targets(self) -> None:
        # Review case: two items that fit only the 30-minute tier come before
        # one that fits 10 minutes; targets are cumulative {10: 1, 30: 2, 60: 3}.
        limits = {10: 100, 30: 300, 60: 600}
        measured = {}
        orders: dict[str, list[str]] = {}
        for split in ("dev", "test"):
            ids = [f"mmlu-pro:{split}-{n}" for n in range(4)]
            for item_id, tokens in zip(ids, (200, 200, 50, 500), strict=True):
                measured[item_id] = Measured(tokens=tokens, finished=True)
            orders[split] = ids

        chosen = assign_tiers(
            "mmlu-pro", orders, {10: 1, 30: 2, 60: 3}, tier_tokens=limits, margin=1.0, measured=measured
        )

        for split in ("dev", "test"):
            tiers = dict(chosen[split])
            self.assertEqual(tiers[f"mmlu-pro:{split}-2"], 10)  # the short item is not lost
            self.assertEqual([sum(1 for t in tiers.values() if t <= tier) for tier in (10, 30, 60)], [1, 2, 3])


class ComparisonTest(unittest.TestCase):
    def _run(self, root: Path, name: str, records: list[dict[str, Any]], status: str = "completed") -> Path:
        run_dir = root / name
        run_dir.mkdir()
        meta = {"tier": 10, "split": "dev", "manifest_sha256": "m", "label": name, "model_id": "g", "status": status}
        (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
        with (run_dir / "items.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({"kind": "chat", "outcome": {"samples": []}, **record}) + "\n")
        return run_dir

    def _scores(self, scores: dict[str, float]) -> list[dict[str, Any]]:
        return [{"id": item_id, "source": item_id.split(":")[0], "score": score} for item_id, score in scores.items()]

    def test_paired_deltas_flips_and_intervals_are_reproducible(self) -> None:
        base_scores = {f"{AREA_SOURCE[area]}:{n}": 1.0 for area in AREA_KEYS for n in range(10)}
        edited_scores = dict(base_scores)
        knowledge, instructions = AREA_SOURCE["knowledge"], AREA_SOURCE["instructions"]
        base_scores.update({f"{instructions}:{n}": 0.0 for n in range(10)})
        edited_scores.update({f"{knowledge}:{n}": 0.0 for n in range(4)})  # four answers lost
        edited_scores.update({f"{instructions}:{n}": float(n < 2) for n in range(10)})  # two gained
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._run(root, "base", self._scores(base_scores))
            edited = self._run(root, "edited", self._scores(edited_scores))

            first = compare(base, edited)
            second = compare(base, edited)

        self.assertAlmostEqual(first["areas"]["knowledge"]["delta"], -0.4)
        self.assertAlmostEqual(first["areas"]["instructions"]["delta"], 0.2)
        self.assertAlmostEqual(first["overall"]["delta"], (-0.4 + 0.2) / len(AREA_KEYS))
        self.assertEqual(first["areas"]["knowledge"]["flips"], {"lost": 4, "gained": 0, "items": 10})
        self.assertEqual(first["areas"]["instructions"]["flips"], {"lost": 0, "gained": 2, "items": 10})
        low, high = first["overall"]["ci95"]
        self.assertLessEqual(low, first["overall"]["delta"])
        self.assertGreaterEqual(high, first["overall"]["delta"])
        self.assertEqual(first, second)

    def test_runs_with_different_items_are_refused_unless_partial(self) -> None:
        # Review case: base has MMLU-Pro=1 and IFEval=0, the candidate only MMLU-Pro=1.
        base_scores = {"mmlu-pro:1": 1.0, "ifeval:1": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._run(root, "base", self._scores(base_scores))
            edited = self._run(root, "edited", self._scores({"mmlu-pro:1": 1.0}))

            with self.assertRaisesRegex(ValueError, "item sets differ"):
                compare(base, edited)
            partial = compare(base, edited, partial=True)

        self.assertIsNone(partial["overall"]["delta"])
        self.assertIsNone(partial["base"]["overall"])
        self.assertEqual(partial["unpaired_items"], {"instructions": 1})

    def test_kl_is_a_lower_bound_including_the_remaining_mass(self) -> None:
        # Review case: the same top 20 tokens at 0.045 (base) and 0.04
        # (edited); the other 80 share 0.1 and 0.2 evenly. True KL 0.036690.
        def position(each: float) -> dict[str, Any]:
            top = [[token, math.log(each)] for token in range(20)]
            return {"token": 0, "logprob": math.log(each), "top": top}

        def probe(each: float) -> list[dict[str, Any]]:
            return [{"id": "probe-code:x", "source": "probe-code", "kind": "logprob", "score": 1.0,
                     "outcome": {"positions": [position(each)]}}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._run(root, "base", probe(0.045))
            edited = self._run(root, "edited", probe(0.04))
            forward = compare(base, edited)["probes"]["probe-code"]["kl_per_token"]
            backward = compare(edited, base)["probes"]["probe-code"]["kl_per_token"]

        true_forward = 0.9 * math.log(0.045 / 0.04) + 0.1 * math.log(0.1 / 0.2)
        true_backward = 0.8 * math.log(0.04 / 0.045) + 0.2 * math.log(0.2 / 0.1)
        self.assertLessEqual(forward, true_forward + 1e-9)
        self.assertAlmostEqual(forward, true_forward, places=6)
        self.assertGreaterEqual(backward, 0.0)
        self.assertLessEqual(backward, true_backward + 1e-9)


class AgentTest(unittest.TestCase):
    def test_total_agent_concurrency_and_harness_settings_bound_the_workers(self) -> None:
        # 30-minute tier shape: Terminal-Bench 6 tasks, SWE-bench Pro 4.
        counts = {"terminal-bench": 6, "swe-bench-pro": 4}

        self.assertEqual(
            allocate_workers(counts, 4, {"terminal-bench": 4, "swe-bench-pro": 4}),
            [{"swe-bench-pro": 2, "terminal-bench": 2}],
        )
        # A harness set to one worker keeps one, whatever its task count.
        self.assertEqual(
            allocate_workers(counts, 4, {"terminal-bench": 1, "swe-bench-pro": 4}),
            [{"swe-bench-pro": 2, "terminal-bench": 1}],
        )
        # Fewer slots than sources: one source after the other.
        self.assertEqual(
            allocate_workers(counts, 1, {"terminal-bench": 4, "swe-bench-pro": 4}),
            [{"swe-bench-pro": 1}, {"terminal-bench": 1}],
        )

    def test_task_image_cleanup_is_refused_only_when_agent_tasks_run(self) -> None:
        from unittest import mock

        from kairyu_bench.target import PreflightError
        from kairyu_bench.wsp.cli import main

        items = [
            {"id": f"{source}:{name}-{split}", "source": source, "split": split, "tier": 10, "max_tokens": 16384}
            for source, name in (("mmlu-pro", "1"), ("terminal-bench", "git-leak-recovery"))
            for split in ("dev", "test")
        ]
        tiers = {str(t): {"minutes": t, "max_output_tokens": 16384} for t in (10, 30, 60)}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ", {"KAIRYU_BENCH_CLEAN_TASK_IMAGES": "1"}
        ):
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "tiers": tiers, "items": items}))
            arguments = [
                "run", "http://127.0.0.1:9/v1", "--tier", "10", "--split", "dev",
                "--manifest", str(manifest), "--incomplete-manifest", "--results-dir", directory,
            ]

            with self.assertRaisesRegex(SystemExit, "KAIRYU_BENCH_CLEAN_TASK_IMAGES"):
                main(arguments)
            # Without agent tasks the run proceeds to the endpoint (none here).
            with self.assertRaises(PreflightError):
                main([*arguments, "--only", "mmlu-pro"])

    def test_swe_bench_pro_keeps_item_verdicts_without_the_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            items = Path(directory) / "raw" / "swe-bench-pro" / "items"
            for index, (instance, verdict) in enumerate((("task-a", True), ("task-b", None)), 1):
                item = items / f"{index:04d}"
                (item / "evaluation").mkdir(parents=True)
                (item / "instance-ids.txt").write_text(instance + "\n", encoding="utf-8")
                if verdict is not None:
                    (item / "evaluation" / "eval_results.json").write_text(json.dumps({instance: verdict}))

            outcomes = _swe_bench_pro_outcomes(Path(directory))

        self.assertEqual({name: entry["reward"] for name, entry in outcomes.items()}, {"task-a": 1.0})


class RunSelectionTest(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        items = [
            {"id": f"{source}:{name}-{split}", "source": source, "split": split,
             "tier": tier, "max_tokens": 16384}
            for source, name, tier in (
                ("mmlu-pro", "keep", 10), ("mmlu-pro", "omit-a", 10),
                ("mmlu-pro", "omit-b", 10), ("mmlu-pro", "later", 30),
                ("ifeval", "outside", 10),
            )
            for split in ("dev", "test")
        ]
        path = root / "manifest.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "tiers": {str(t): {"minutes": t, "max_output_tokens": 16384} for t in (10, 30, 60)},
            "items": items,
        }))
        return path

    def _arguments(self, root: Path, manifest: Path) -> list[str]:
        return [
            "run", "http://api.test/v1", "--tier", "10", "--split", "dev",
            "--manifest", str(manifest), "--incomplete-manifest", "--only", "mmlu-pro",
            "--results-dir", str(root), "--run-id", "selected",
        ]

    def test_excluded_items_are_not_executed_and_keep_manifest_identity(self) -> None:
        import hashlib
        import io
        from contextlib import redirect_stdout
        from dataclasses import replace
        from unittest import mock

        from kairyu_bench.wsp.cli import main

        executed: list[str] = []

        def run_one(context: Context, item: Item, row: dict[str, Any]) -> dict[str, Any]:
            executed.append(item.id)
            return {"samples": [_reply("Answer: A")]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            original = manifest.read_bytes()
            rows = {entry["id"]: {"expected": "A"} for entry in json.loads(original)["items"]}
            adapter = replace(adapters()["mmlu-pro"], load=lambda loader: rows, execute=run_one)
            with (
                mock.patch("kairyu_bench.target.TargetClient.discover_chat_model", return_value="fixture"),
                mock.patch("kairyu_bench.wsp.run.adapters", return_value={"mmlu-pro": adapter}),
                mock.patch("kairyu_bench.wsp.client._post", side_effect=AssertionError("unexpected API call")),
                redirect_stdout(io.StringIO()),
            ):
                result = main([
                    *self._arguments(root, manifest),
                    "--exclude-item", "mmlu-pro:omit-b-dev",
                    "--exclude-item", "mmlu-pro:omit-a-dev",
                ])
            meta = json.loads((root / "selected/run.json").read_text())
            records = [json.loads(line) for line in (root / "selected/items.jsonl").read_text().splitlines()]
            self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual(result, 0)
        self.assertEqual(executed, ["mmlu-pro:keep-dev"])
        self.assertEqual([(record["id"], record["score"]) for record in records], [("mmlu-pro:keep-dev", 1.0)])
        self.assertEqual(meta["excluded_item_ids"], ["mmlu-pro:omit-a-dev", "mmlu-pro:omit-b-dev"])
        self.assertEqual(meta["manifest_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual((meta["items"], meta["status"]), (1, "completed"))

    def test_outside_selection_and_all_item_exclusions_fail_before_discovery(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest import mock

        from kairyu_bench.wsp.cli import main

        cases = [
            ["mmlu-pro:missing"], ["mmlu-pro:keep-test"], ["mmlu-pro:later-dev"],
            ["ifeval:outside-dev"],
            ["mmlu-pro:keep-dev", "mmlu-pro:omit-a-dev", "mmlu-pro:omit-b-dev"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            for excluded in cases:
                with (
                    self.subTest(excluded=excluded),
                    mock.patch("kairyu_bench.target.TargetClient.discover_chat_model") as discovery,
                    redirect_stderr(io.StringIO()),
                ):
                    flags = [flag for item_id in excluded for flag in ("--exclude-item", item_id)]
                    self.assertEqual(main([*self._arguments(root, manifest), *flags]), 2)
                    discovery.assert_not_called()
                    self.assertFalse((root / "selected").exists())


class ScoringTest(unittest.TestCase):
    """Official extraction rules: a right reply scores 1, a wrong one 0, and a
    reply cut at the token cap 0 whatever it says."""

    CASES: list[tuple[str, dict[str, Any], str, str]] = [
        ("gpqa-diamond", {"expected": "C"}, "Reasoning...\nAnswer: C", "Answer: B"),
        ("mmlu-pro", {"expected": "D"}, "So the answer is (D).", "The answer is (A)."),
        ("mmlu-prox-ja", {"expected": "b"}, "よって、答えは (B) です", "答えは (C) です"),
        ("jamc-qa", {"expected": "A"}, "考察...\n答え: (A)", "答え: (D)"),
        ("gsm8k-platinum", {"target": "18"}, "Work.\nAnswer: 18", "Answer: 17"),
        ("mgsm", {"target": "18", "language": "ja"}, "計算...\n答え: 18", "答え: 20"),
        ("triviaqa", {"aliases": ["eritrea"]}, "The answer is Eritrea.", "The answer is Ethiopia."),
        ("musr", {"choices": ["a", "b"], "gold": 2}, "Thinking.\nANSWER: 2", "ANSWER: 1"),
        ("mmmu-pro", {"options": ["x", "y", "z"], "answer": "B"}, "Steps.\nAnswer: B", "Answer: C"),
        ("ruler", {"outputs": ["1234567"]}, "The number is 1234567.", "The number is 7654321."),
        ("mrcr-v2", {"answer": "pfx hello world", "prepend": "pfx"}, "pfx hello world", "hello world"),
        ("longbench-v2", {"expected": "A"}, "Answer: A", "Answer: B"),
    ]

    def test_official_rules_score_right_wrong_and_truncated_replies(self) -> None:
        registry = adapters()
        for source, row, right, wrong in self.CASES:
            with self.subTest(source=source):
                outcomes = [{"samples": [_reply(right)]}, {"samples": [_reply(wrong)]}, {"samples": [_reply(right, "length")]}]
                scored = registry[source].score([(_item(source), row, outcome) for outcome in outcomes], Loader())
                self.assertEqual([entry.score for entry in scored], [1.0, 0.0, 0.0])

    def test_scicode_sample_cut_at_the_cap_scores_zero(self) -> None:
        # One step reached the token cap; the step tests are not run.
        row = {"problem": {"problem_id": "1", "sub_steps": [{"step_number": "1.1", "test_cases": []}]}}
        outcome = {"samples": [{"finish_reason": "length", "files": {"1.1": "def f(): return 1"}}]}

        scored = adapters()["scicode"].score([(_item("scicode"), row, outcome)], Loader())

        self.assertEqual((scored[0].score, scored[0].detail), (0.0, {"samples": [{"truncated": True}]}))

    def test_scicode_step_diagnostics_preserve_success_failure_timeout_and_redact_streams(self) -> None:
        import hashlib
        import subprocess
        from unittest import mock

        from kairyu_bench.wsp.scorers import scicode_score as scorer

        marker = "PRIVATE_STREAM_SENTINEL"
        cases = [
            (subprocess.CompletedProcess(["python"], 0, marker, ""), True, 0, False, None),
            (subprocess.CalledProcessError(1, ["python"], output=marker,
                                          stderr="ValueError: " + marker), False, 1, False, "ValueError"),
            (subprocess.TimeoutExpired(["python"], 300, output=marker.encode(), stderr=None),
             False, None, True, None),
        ]
        job = {"key": "0:0:1.1", "step": "1.1", "code": "value = 1", "tests": []}
        with tempfile.TemporaryDirectory() as directory:
            for effect, expected, returncode, timeout, exception in cases:
                patch_args = {"side_effect": effect} if isinstance(effect, Exception) else {"return_value": effect}
                with (
                    self.subTest(expected=expected, timeout=timeout),
                    mock.patch.object(scorer.subprocess, "run", **patch_args) as run,
                    mock.patch.object(scorer.time, "perf_counter", side_effect=[5.0, 5.25]),
                ):
                    passed, detail = scorer._run("python", "data.h5", job, Path(directory))
                self.assertEqual((passed, detail["returncode"], detail["timed_out"], detail["exception_class"]),
                                 (expected, returncode, timeout, exception))
                self.assertEqual(detail["elapsed_seconds"], .25)
                self.assertEqual(detail["stdout"]["bytes"], len(marker.encode()))
                self.assertEqual(detail["stdout"]["sha256"], hashlib.sha256(marker.encode()).hexdigest())
                self.assertNotIn(marker, json.dumps(detail))
                self.assertNotIn("code", detail)
                self.assertNotIn("tests", detail)
                self.assertEqual(run.call_args.kwargs,
                                 {"check": True, "capture_output": True, "text": True,
                                  "timeout": 300, "cwd": Path(directory)})
                if timeout:
                    self.assertIsNone(detail["stderr"]["bytes"])
                    self.assertIsNone(detail["stderr"]["sha256"])
        self.assertEqual(scorer._exception_class("AssertionError"), "AssertionError")
        self.assertEqual(scorer._exception_class("numpy.linalg.LinAlgError: " + marker), "LinAlgError")
        self.assertIsNone(scorer._exception_class("CustomSecretError: " + marker))

    def test_scicode_sanitized_step_metadata_does_not_change_fractional_score(self) -> None:
        from unittest import mock

        row = {"problem": {"problem_id": "1", "sub_steps": [
            {"step_number": f"1.{i}", "test_cases": []} for i in range(1, 4)
        ]}}
        outcome = {"samples": [{"finish_reason": "stop", "files": {
            f"1.{i}": "value = 1" for i in range(1, 4)
        }}]}
        verdicts = {"0:0:1.1": True, "0:0:1.2": False, "0:0:1.3": False}
        diagnostics = {
            key: {"returncode": 0 if passed else (None if key.endswith("3") else 1),
                  "timed_out": key.endswith("3"), "elapsed_seconds": .1,
                  "exception_class": "AssertionError" if key.endswith("2") else None,
                  "stdout": {"bytes": 0, "sha256": "hash", "representation": "utf8_of_captured_text"},
                  "stderr": {"bytes": 0, "sha256": "hash", "representation": "utf8_of_captured_text"}}
            for key, passed in verdicts.items()
        }
        loader = mock.Mock()
        loader.python.return_value = Path("python")
        loader.source.return_value = Path("source")
        loader.file.return_value = Path("data.h5")
        with mock.patch("kairyu_bench.wsp.sources.code.run_scorer", return_value={
            "schema_version": 1, "verdicts": verdicts, "diagnostics": diagnostics,
        }):
            scored = adapters()["scicode"].score([(_item("scicode"), row, outcome)], loader)
        self.assertEqual((scored[0].score, scored[0].samples), (1 / 3, (1 / 3,)))
        detail = scored[0].detail["samples"][0]
        self.assertEqual((detail["steps_passed"], detail["steps_tested"], detail["problem_solved"]), (1, 3, False))
        self.assertEqual([x["step"] for x in detail["step_diagnostics"]], ["1.1", "1.2", "1.3"])
        self.assertEqual([x["passed"] for x in detail["step_diagnostics"]], [True, False, False])
        self.assertEqual([x["timed_out"] for x in detail["step_diagnostics"]], [False, False, True])

    def test_choice_log_likelihood_divides_by_choice_characters_as_lm_eval(self) -> None:
        # Review case: ARC choices "a" and "bbb" with log-likelihoods -1 and
        # -2.2. lm-eval's acc_norm divides by 1 and 3 and picks "bbb".
        targets = {" a": -1.0, " bbb": -2.2}

        class Logprobs:
            def tokenize(self, text: str, *, add_special_tokens: bool) -> list[int]:
                return [ord(character) for character in text]

            def prompt_logprobs(self, tokens: list[int], *, top_k: int) -> list[Any]:
                continuation = next(text for text in targets if [ord(c) for c in text] == tokens[-len(text):])
                share = targets[continuation] / len(continuation)
                return [None] + [{token: (share, 1)} for token in tokens[1:]]

        from kairyu_bench.wsp.sources import logprob

        logprob._bos.clear()
        logprob._bos[0] = [2]
        row = {"question": "q", "choices": {"text": ["a", "bbb"], "label": ["A", "B"]}, "answerKey": "B"}
        item = _item("arc-challenge")
        adapter = adapters()["arc-challenge"]
        outcome = adapter.execute(Context(chat=None, logprob=Logprobs()), item, row)  # type: ignore[arg-type]
        scored = adapter.score([(item, row, outcome)], Loader())

        self.assertEqual(outcome["lengths"], [1, 3])
        self.assertEqual(scored[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
