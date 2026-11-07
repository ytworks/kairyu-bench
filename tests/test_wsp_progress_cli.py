"""Guard the progress CLI's actual help and cross-run failure boundaries."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "docs/experiments/scripts/fresh-av-dev30-progress-20261107.py"
spec = importlib.util.spec_from_file_location("wsp_progress_cli", SCRIPT)
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


class ProgressCliTests(unittest.TestCase):
    def test_help_and_missing_required_roots_have_no_io_or_monitor_calls(self):
        for argv, status in ([str(SCRIPT), "--help"], 0), ([str(SCRIPT)], 2):
            with self.subTest(argv=argv), mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(progress, "collect") as collect, \
                    mock.patch.object(progress, "validate_roots") as validate, \
                    mock.patch.object(Path, "open", side_effect=AssertionError("file I/O")), \
                    mock.patch.object(progress.urllib.request, "urlopen") as urlopen, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    progress.main()
                self.assertEqual(error.exception.code, status)
                collect.assert_not_called()
                validate.assert_not_called()
                urlopen.assert_not_called()

    def fixture(self, root):
        analysis = root / "analysis"
        run = root / "results/run1"
        (analysis / "http-trace").mkdir(parents=True)
        run.mkdir(parents=True)
        config = analysis / "http-trace/capture.json"
        config.write_text("{}")
        plan = {"cwd": str(run.parent), "run_id": run.name,
                "command": ["bench", "wsp", "run", "endpoint", "--run-id", run.name,
                            "--results-dir", str(run.parent), "--observe-config", str(config),
                            "--tier", "30", "--split", "dev", "--label", "same-label"],
                "http_trace": {"config": {"path": str(config),
                                         "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}}}
        plan_path = analysis / "any-approved-name.json"
        plan_path.write_text(json.dumps(plan))
        (analysis / "owner-start.json").write_text(json.dumps({
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()}))
        metadata = {"kind": "wsp", "tier": 30, "split": "dev", "label": "same-label",
                    "observation": {"run_id": run.name, "output_dir": str(config.parent)}}
        (run / "run.json").write_text(json.dumps(metadata))
        return analysis, run

    def test_valid_binding_and_wrong_analysis_or_run_rejected_before_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis, run = self.fixture(root)
            progress.validate_roots(analysis, run)
            wrong_run = root / "results/wrong"
            wrong_run.mkdir()
            (wrong_run / "run.json").write_bytes((run / "run.json").read_bytes())
            wrong_analysis = root / "wrong-analysis"
            wrong_analysis.mkdir()
            (wrong_analysis / "owner-start.json").write_bytes((analysis / "owner-start.json").read_bytes())
            for a, r in ((analysis, wrong_run), (wrong_analysis, run)):
                with self.subTest(a=a, r=r), mock.patch.object(
                    sys, "argv", [str(SCRIPT), "--analysis-root", str(a), "--run-root", str(r)]
                ), mock.patch.object(progress, "collect") as collect, mock.patch.object(
                    progress.urllib.request, "urlopen"
                ) as urlopen:
                    with self.assertRaises(ValueError):
                        progress.main()
                    collect.assert_not_called()
                    urlopen.assert_not_called()
                    self.assertFalse((a / "progress.jsonl").exists())

    def test_run_metadata_or_owner_plan_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis, run = self.fixture(Path(directory))
            metadata = json.loads((run / "run.json").read_text())
            metadata["label"] = "other-checkpoint"
            (run / "run.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "do not match"):
                progress.validate_roots(analysis, run)
            (analysis / "any-approved-name.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "no matching plan"):
                progress.validate_roots(analysis, run)


if __name__ == "__main__":
    unittest.main()
