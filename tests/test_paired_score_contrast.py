from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kairyu_bench.wsp import report

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/experiments/paired-score-contrast.py"
)
spec = importlib.util.spec_from_file_location("paired_score_contrast", SCRIPT)
assert spec and spec.loader
TOOL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TOOL)


def export(arms, rows):
    return {
        "schema_version": 1,
        "kind": "paired-score-export",
        "data_kind": "synthetic",
        "source_artifact_sha256": "a" * 64,
        "arms": arms,
        "rows": [
            {"id": name, "source": source, "scores": dict(zip(arms, scores))}
            for name, source, scores in rows
        ],
    }


class PairedScoreContrastTests(unittest.TestCase):
    def test_joint_rows_preserve_covariance_and_source_weights(self):
        data = export(
            ["O", "L", "P", "M"],
            [
                ("a", "small", (0, 0.25, 2, 2.25)),
                ("b", "large", (1, 1.25, 4, 4.25)),
                ("c", "large", (0, 0.25, 8, 8.25)),
                ("d", "large", (1, 1.25, 0, 0.25)),
            ],
        )
        result = TOOL.contrast(data, {"O": 1, "L": -1, "P": -1, "M": 1}, resamples=400)
        self.assertAlmostEqual(result["estimate"], 0, places=14)
        for endpoint in result["ci95"]:
            self.assertAlmostEqual(endpoint, 0, places=14)
        # Source with one row has the same weight as source with three rows.
        weighted = export(
            ["O", "L"],
            [
                ("a", "small", (0, 1)),
                ("b", "large", (0, 0)),
                ("c", "large", (0, 0)),
                ("d", "large", (0, 0)),
            ],
        )
        result = TOOL.contrast(weighted, {"O": -1, "L": 1}, resamples=100)
        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["ci95"], [0.5, 0.5])

    def test_pair_matches_official_compare_with_same_source_row_order(self):
        # Nonalphabetical row/source order exercises the official draw-order contract.
        data = export(
            ["O", "L"],
            [
                ("s:z", "scicode", (1 / 3, 2 / 3)),
                ("h:z", "humaneval-plus", (0, 1)),
                ("s:a", "scicode", (0.75, 0.5)),
                ("h:a", "humaneval-plus", (1, 0)),
                ("s:m", "scicode", (0, 1)),
                ("m:z", "mbpp-plus", (1, 1)),
                ("l:z", "livecodebench", (0, 1)),
                ("m:a", "mbpp-plus", (1, 0)),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in data["arms"]:
                d = root / arm
                d.mkdir()
                (d / "run.json").write_text(
                    json.dumps(
                        {
                            "manifest_sha256": "same",
                            "tier": 30,
                            "split": "dev",
                            "status": "completed",
                            "unscored": [],
                        }
                    )
                )
                (d / "items.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {
                                "id": row["id"],
                                "source": row["source"],
                                "score": row["scores"][arm],
                                "kind": "chat",
                            }
                        )
                        + "\n"
                        for row in data["rows"]
                    )
                )
            official = report.compare(root / "O", root / "L", partial=True)["areas"][
                "code"
            ]
            result = TOOL.contrast(
                data,
                {"O": -1, "L": 1},
                seed=report.BOOTSTRAP_SEED,
                resamples=report.BOOTSTRAP_RESAMPLES,
            )
        self.assertAlmostEqual(result["estimate"], official["delta"], places=14)
        for actual, expected in zip(result["ci95"], official["ci95"]):
            self.assertAlmostEqual(actual, expected, places=14)
        self.assertEqual(result, TOOL.contrast(data, {"O": -1, "L": 1}))

    def test_missing_duplicate_nonfinite_and_body_fields_are_refused(self):
        good = export(["O", "L"], [("a", "source", (0.25, 0.75))])
        variants = []
        for value in (None, True, float("nan"), float("inf")):
            bad = copy.deepcopy(good)
            bad["rows"][0]["scores"]["L"] = value
            variants.append(bad)
        bad = copy.deepcopy(good)
        del bad["rows"][0]["scores"]["L"]
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["rows"][0]["scores"]["extra"] = 0
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["rows"].append(copy.deepcopy(bad["rows"][0]))
        bad["rows"][-1]["source"] = "other"
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["rows"][0]["response"] = "do-not-read"
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["arms"] = ["O", "O"]
        variants.append(bad)
        for bad in variants:
            with (
                self.subTest(bad=repr(bad)),
                self.assertRaises((ValueError, OverflowError)),
            ):
                TOOL.contrast(bad, {"O": -1, "L": 1}, resamples=10)
        for coefficients in ({"O": -1}, {"O": 0, "L": 0}, {"O": -1, "L": float("inf")}):
            with self.assertRaises(ValueError):
                TOOL.contrast(good, coefficients, resamples=10)

    def test_captured_input_pin_continuity_and_exclusive_output(self):
        data = export(["O", "L"], [("a", "source", (0.25, 0.75))])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "scores.json", root / "contrast.json"
            raw = json.dumps(data).encode()
            source.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            with self.assertRaises(ValueError):
                TOOL.execute(
                    source, "0" * 64, output, {"O": -1, "L": 1}, seed=1, resamples=10
                )
            self.assertFalse(output.exists())
            real = TOOL.contrast

            def changed(*args, **kwargs):
                result = real(*args, **kwargs)
                source.write_bytes(raw + b" ")
                return result

            with (
                patch.object(TOOL, "contrast", side_effect=changed),
                self.assertRaises(ValueError),
            ):
                TOOL.execute(
                    source, digest, output, {"O": -1, "L": 1}, seed=1, resamples=10
                )
            self.assertFalse(output.exists())
            source.write_bytes(raw)
            result = TOOL.execute(
                source, digest, output, {"O": -1, "L": 1}, seed=1, resamples=10
            )
            before = output.read_bytes()
            self.assertEqual(result["estimate"], 0.5)
            self.assertEqual(result["input"]["sha256"], digest)
            with self.assertRaises(ValueError):
                TOOL.execute(
                    source, digest, output, {"O": -1, "L": 1}, seed=1, resamples=10
                )
            self.assertEqual(output.read_bytes(), before)
            duplicate = raw.replace(b'"O": 0.25', b'"O": 0.25, "O": 0.5')
            source.write_bytes(duplicate)
            with self.assertRaises(ValueError):
                TOOL.execute(
                    source,
                    hashlib.sha256(duplicate).hexdigest(),
                    root / "bad.json",
                    {"O": -1, "L": 1},
                    seed=1,
                    resamples=10,
                )
            self.assertFalse((root / "bad.json").exists())


if __name__ == "__main__":
    unittest.main()
