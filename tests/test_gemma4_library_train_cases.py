from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


TRAIN = module("library_train_fixture", "library_train_cases.py")
COMMON = module("library_train_reader_fixture", "recipe_common.py")
HAS_LIBRARIES = all(importlib.util.find_spec(name) for name in ("scipy", "sympy"))


class LibraryTrainingTest(unittest.TestCase):
    @unittest.skipUnless(
        HAS_LIBRARIES, "optional SciPy/SymPy CPU reference environment"
    )
    def test_executed_library_contracts_preserve_existing_training_and_output(self):
        old = {
            "code": [
                [{"id": "old-code-0", "prompt": "p", "correct": "c", "foil": "f"}],
                [{"id": "old-code-1", "prompt": "p", "correct": "c", "foil": "f"}],
            ],
            "science_numerics": [
                [{"id": "old-science-0", "prompt": "p", "correct": "c", "foil": "f"}],
                [{"id": "old-science-1", "prompt": "p", "correct": "c", "foil": "f"}],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            base.write_text(json.dumps(old))
            before = base.read_bytes()
            output = root / "new"
            report = TRAIN.write_bundle(
                output,
                base=base,
                base_sha256=hashlib.sha256(before).hexdigest(),
                seed=58421,
                trials=4,
            )
            self.assertEqual(base.read_bytes(), before)
            combined = json.loads((output / "train-cases.json").read_text())
            for purpose, folds in old.items():
                for fold, rows in enumerate(folds):
                    self.assertEqual(combined[purpose][fold][: len(rows)], rows)
            readback = COMMON.read_cases(output / "train-cases.json")
            self.assertEqual(sum(len(f) for fs in readback.values() for f in fs), 20)
            new = json.loads((output / "train-extension.json").read_text())
            public = [r for folds in new.values() for fold in folds for r in fold]
            self.assertEqual(len(public), 16)
            self.assertEqual(len({r["family"] for r in public}), 8)
            self.assertTrue(
                all(r["gradient_allowed"] and r["split"] == "train" for r in public)
            )
            self.assertTrue(
                all(r["correct"]["failed"] == 0 for r in report["validation"]["cases"])
            )
            self.assertTrue(
                all(r["foil"]["failed"] > 0 for r in report["validation"]["cases"])
            )
            self.assertFalse(report["validation"]["token_count_measured"])
            snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                TRAIN.write_bundle(output)
            self.assertEqual(
                {p.name: p.read_bytes() for p in output.iterdir()}, snapshot
            )


if __name__ == "__main__":
    unittest.main()
