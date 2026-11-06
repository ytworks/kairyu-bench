from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/experiments/gemma4-purpose"
sys.path.insert(0, str(DIRECTORY))
import library_transfer_cases as LIBRARY  # noqa: E402
import science_sandbox as SANDBOX  # noqa: E402


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("scipy", "sympy")),
    "optional SciPy/SymPy authored execution environment",
)
class LibraryTransferTest(unittest.TestCase):
    def test_bundle_independent_references_foils_and_worker_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            metadata = LIBRARY.write_bundle(bundle, seed=481205, trials=4)
            self.assertTrue(
                all(
                    row["correct"]["failed"] == 0
                    for row in metadata["validation"]["cases"]
                )
            )
            self.assertTrue(
                all(
                    row["foil"]["failed"] > 0 for row in metadata["validation"]["cases"]
                )
            )
            self.assertTrue(
                all(row["passed"] == 4 for row in metadata["validation"]["metamorphic"])
            )
            source = bundle / "transfer-cases.json"
            data, receipt = SANDBOX.capture(source, 8 * 1024 * 1024)
            rows = json.loads(data)["cases"]
            responses = root / "responses.json"
            responses.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "split": "transfer",
                        "cases_sha256": receipt["sha256"],
                        "generation": {"reasoning_saved": False},
                        "responses": [
                            {"id": row["id"], "public_answer": row["correct"]}
                            for row in rows
                        ],
                    }
                )
            )
            preview = SANDBOX.main(
                [
                    "--cases",
                    str(source),
                    "--responses",
                    str(responses),
                    "--catalog",
                    "library-transfer-v1",
                    "--output",
                    str(root / "preview"),
                    "--docker-socket",
                    "/not-opened.sock",
                    "--daemon-id",
                    "not-contacted",
                    "--image",
                    "sha256:" + "a" * 64,
                ],
                docker_factory=lambda *a, **k: self.fail("preview contacted Docker"),
            )
            self.assertEqual(preview["responses_received"], 8)
            with self.assertRaisesRegex(ValueError, "transfer cases"):
                LIBRARY.shared.training_payload(LIBRARY.catalog())
            for case in LIBRARY.catalog():
                # The worker subprocess executes only these trusted authored references.
                # Arbitrary generated programs still require science_sandbox.Docker.
                for foil in (False, True):
                    code = case.code(foil=foil)
                    # Exercise the library-only worker policy, independent of the
                    # reference's local guard around SciPy nextafter(0, inf).
                    if case.name.endswith("event-integral"):
                        code = code.replace(
                            'with np.errstate(under="ignore"):', "if True:"
                        )
                    payload, expected = SANDBOX.prepare(
                        case,
                        code,
                        481205,
                        4,
                        sample_fn=LIBRARY.sample,
                    )
                    self.assertNotIn("reference", payload)
                    child = subprocess.run(
                        [sys.executable, "-B", "-c", SANDBOX.WORKER],
                        input=json.dumps(payload),
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=True,
                    )
                    result = SANDBOX.compare(json.loads(child.stdout), expected)
                    self.assertEqual(result["passed"], not foil, case.name)
            self.assertEqual(json.loads((bundle / "train-cases.json").read_text()), {})


if __name__ == "__main__":
    unittest.main()
