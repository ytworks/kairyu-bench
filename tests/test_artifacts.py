from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kairyu_bench.artifacts import (
    env_integer,
    read_object,
    redact_artifacts,
    sha256_file,
    write_json,
)


class ArtifactsTest(unittest.TestCase):
    def test_write_json_is_atomic_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "value.json"

            write_json(path, {"a": 1})

            self.assertEqual(read_object(path, "Test"), {"a": 1})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_read_object_names_the_benchmark_and_rejects_non_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.json"
            path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Terminal-Bench artifact"):
                read_object(path, "Terminal-Bench")
            with self.assertRaisesRegex(ValueError, "cannot read Terminal-Bench"):
                read_object(Path(directory) / "missing.json", "Terminal-Bench")

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_bytes(b"abc")

            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_redact_artifacts_replaces_each_configured_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "a" / "log.txt").write_text(
                "key=sk-test token=hf-test anthropic=ak-test", encoding="utf-8"
            )
            env = {
                "OPENAI_API_KEY": "sk-test",
                "HF_TOKEN": "hf-test",
                "ANTHROPIC_API_KEY": "ak-test",
            }

            redact_artifacts(
                root, ("OPENAI_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY"), env
            )

            self.assertEqual(
                (root / "a" / "log.txt").read_text(encoding="utf-8"),
                "key=[REDACTED] token=[REDACTED] anthropic=[REDACTED]",
            )

    def test_redact_artifacts_ignores_placeholder_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "log.txt").write_text("not-required", encoding="utf-8")

            redact_artifacts(root, ("OPENAI_API_KEY",), {"OPENAI_API_KEY": "not-required"})

            self.assertEqual(
                (root / "log.txt").read_text(encoding="utf-8"), "not-required"
            )

    def test_env_integer_validates_range_and_digits(self) -> None:
        self.assertEqual(env_integer({}, "X", 4, 1, 16), 4)
        self.assertEqual(env_integer({"X": "16"}, "X", 4, 1, 16), 16)
        for value in ("", "0", "17", "-1", "1.5", "four"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "X must be an integer from 1 to 16"):
                    env_integer({"X": value}, "X", 4, 1, 16)


if __name__ == "__main__":
    unittest.main()
