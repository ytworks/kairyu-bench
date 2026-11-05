from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reviewer = load("claude-method-review")
reserve = load("wsp-reserve-audit")


def events(model="claude-opus-5-5", tool=False):
    return "\n".join(json.dumps(value) for value in [
        {"type": "system", "cwd": "/private/machine/path", "environment": {"secret": "NOT_PERSISTED"}},
        {"type": "assistant", "message": {"model": model, "content": [
            {"type": "thinking", "thinking": "HIDDEN_BLOCK_MUST_NOT_PERSIST"},
            *([{"type": "tool_use", "name": "Read"}] if tool else []),
        ]}},
        {"type": "result", "is_error": False, "result": json.dumps({
            "reviews": [{"label": label, "priority": "conditional"} for label in "ABCDEFGH"],
            "evidence_limits": ["Method-only."],
        }), "modelUsage": {model: {"inputTokens": 3, "outputTokens": 12, "canonicalModel": model}}},
    ])


def args(root, send=False):
    prompt = root / "prompt.txt"
    prompt.write_text("Review anonymous public joint BF16 mechanisms A through H; no measured candidate data.")
    return argparse.Namespace(
        prompt=prompt, output=root / "review", labels="ABCDEFGH", model="claude-opus-5-5",
        send=send, approved_prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
        timeout_seconds=30, max_budget_usd=2.0, claude_binary=Path("/trusted/claude"),
    )


class ClaudeMethodReviewTests(unittest.TestCase):
    def test_preview_never_calls_cli_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            options = args(Path(directory))
            with patch.object(reviewer.subprocess, "run") as execute:
                result = reviewer.review(options)
            execute.assert_not_called()
            self.assertEqual(result["mode"], "preview_only")
            self.assertFalse(options.output.exists())

    def test_prompt_lint_and_hash_prevent_send(self):
        for text in ["read /home/person/private", "https://example.org/private", "Bearer secret-value"]:
            with self.assertRaises(ValueError):
                reviewer.validate_prompt(text.encode())
        with tempfile.TemporaryDirectory() as directory:
            options = args(Path(directory), send=True)
            options.approved_prompt_sha256 = "0" * 64
            with patch.object(reviewer.subprocess, "run") as execute, self.assertRaises(ValueError):
                reviewer.review(options)
            execute.assert_not_called()
            self.assertFalse(options.output.exists())

    def test_verified_identity_and_no_hidden_or_environment_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            options = args(Path(directory), send=True)
            process = subprocess.CompletedProcess([], 0, events(), "authentication secret never persisted")
            with patch.object(reviewer.subprocess, "run", return_value=process) as execute:
                result = reviewer.review(options)
            self.assertTrue(result["verified_requested_model"])
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index("--tools") + 1], "")
            self.assertIn("--safe-mode", command)
            self.assertIn("--restricted", command)
            self.assertIn("--strict-mcp-config", command)
            self.assertNotIn("--fallback-model", command)
            receipt = (options.output / "receipt.json").read_text()
            for forbidden in ["HIDDEN_BLOCK", "NOT_PERSISTED", "/private/machine", "authentication secret"]:
                self.assertNotIn(forbidden, receipt)
            with self.assertRaises(FileExistsError):
                reviewer.review(options)

    def test_wrong_model_tool_call_and_alias_usage_fail_closed(self):
        self.assertFalse(reviewer.parse_stream(events("another-model"), "claude-opus-5-5", "ABCDEFGH")["verified_requested_model"])
        self.assertFalse(reviewer.parse_stream(events(tool=True), "claude-opus-5-5", "ABCDEFGH")["verified_requested_model"])
        stream = events().replace('"canonicalModel": "claude-opus-5-5"', '"canonicalModel": "another-model"')
        self.assertFalse(reviewer.parse_stream(stream, "claude-opus-5-5", "ABCDEFGH")["verified_requested_model"])
        self.assertFalse(reviewer.parse_stream(events(), "claude-opus-5-5", "AB")["verified_requested_model"])

    def test_timeout_is_recorded_without_raw_output_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            options = args(Path(directory), send=True)
            error = subprocess.TimeoutExpired("private-command", 30, output="HIDDEN_BLOCK")
            with patch.object(reviewer.subprocess, "run", side_effect=error) as execute:
                result = reviewer.review(options)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(result["error_class"], "TimeoutExpired")
            self.assertFalse(result["verified_requested_model"])
            self.assertNotIn("HIDDEN_BLOCK", (options.output / "receipt.json").read_text())


def item(source, suffix, tier, split="test"):
    return {"id": source + ":" + suffix, "source": source, "split": split, "tier": tier,
            "max_tokens": 100, "samples": 1, "stratum": "fixture", "expected_tokens": 0}


class ReserveMetadataTests(unittest.TestCase):
    def test_translation_presentation_exclusion_and_unresolved_quarantine(self):
        data = {"schema_version": 1, "items": [
            item("mmlu-pro", "7", 10), item("mmlu-prox-ja", "7", 30),
            item("mgsm", "en:001", 10), item("mgsm", "ja:001", 30),
            item("mmmu-pro", "standard:9", 10), item("mmmu-pro", "vision:9", 30),
            item("gsm8k-platinum", "0002", 30), item("mgsm", "de:002", 30),
            item("humaneval-plus", "HumanEval/9", 30),
            item("mmlu-pro", "99", 30), item("mmlu-prox-ja", "99", 30),
        ]}
        result = reserve.audit(data, "a" * 64)
        self.assertEqual(result["raw_difference_items"], 8)
        self.assertEqual(result["known_group_overlap_exclusions"], 3)
        self.assertEqual(result["unresolved_cross_source_quarantine"], 2)
        self.assertEqual(result["proposed_reserve_items"], 3)
        self.assertEqual(result["distinct_underlying_groups"], 2)
        self.assertEqual(result["within_reserve_extra_presentations"], 1)

    def test_seen_dev_groups_and_content_fields_rejected(self):
        data = {"schema_version": 1, "items": [
            item("mmlu-pro", "20", 60, "dev"), item("mmlu-prox-ja", "20", 30),
            item("mbpp-plus", "Mbpp/10", 30),
        ]}
        self.assertEqual(reserve.audit(data, "b" * 64)["known_group_overlap_exclusions"], 1)
        data["items"][0]["question"] = "Task text is forbidden in this audit."
        with self.assertRaisesRegex(ValueError, "metadata only"):
            reserve.audit(data, "b" * 64)

    def test_duplicate_ids_rejected(self):
        row = item("mbpp-plus", "Mbpp/10", 30)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reserve.audit({"schema_version": 1, "items": [row, dict(row)]}, "c" * 64)


if __name__ == "__main__":
    unittest.main()
