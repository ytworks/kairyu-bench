"""Safety boundaries of the one-off F02 archive representation recipe; CPU only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "docs/experiments/scripts/fresh-f02-payload-release-20261107.py"


def load():
    spec = importlib.util.spec_from_file_location("f02_release_recipe", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class F02ReleaseGuards(unittest.TestCase):
    def test_unapproved_apply_rejects_before_directory_lock_or_file_operations(self):
        module = load()
        for approval in ([], ["--approved-plan-sha256", module.PLAN_SHA,
                              "--approved-operation-sha256", "0" * 64]):
            with self.subTest(approval=approval), mock.patch.object(module.os, "getuid", return_value=1003), \
                    mock.patch.object(module, "sha", side_effect=lambda p: module.PLAN_SHA if p == module.PLAN else "a" * 64), \
                    mock.patch("sys.argv", [str(SOURCE), "--output", str(module.ROOT / "analysis/never-created"),
                                           "--apply", *approval]), \
                    mock.patch.object(module.os, "open") as opened, \
                    mock.patch.object(module.os, "unlink") as unlinked, \
                    mock.patch.object(Path, "mkdir") as mkdir:
                with self.assertRaisesRegex(ValueError, "approval required"):
                    module.main()
                opened.assert_not_called()
                unlinked.assert_not_called()
                mkdir.assert_not_called()

    def test_durable_evidence_never_overwrites_existing_journal(self):
        module = load()
        with tempfile.TemporaryDirectory(dir=module.os.environ.get("TMPDIR")) as temporary:
            path = Path(temporary)
            directory_fd = module.os.open(path, module.os.O_RDONLY | module.os.O_DIRECTORY)
            try:
                with mock.patch.object(module.os, "fsync", wraps=module.os.fsync) as synced:
                    digest = module.durable_json(directory_fd, "intent.json", {"released": []})
                self.assertEqual(synced.call_count, 2)
                self.assertEqual(synced.call_args.args, (directory_fd,))
                before = (path / "intent.json").read_bytes()
                self.assertEqual(module.sha(path / "intent.json"), digest)
                self.assertEqual(json.loads(before), {"released": []})
                with self.assertRaises(FileExistsError):
                    module.durable_json(directory_fd, "intent.json", {"released": ["unexpected"]})
                self.assertEqual((path / "intent.json").read_bytes(), before)
            finally:
                module.os.close(directory_fd)



class F03ReleaseBindings(unittest.TestCase):
    @staticmethod
    def load_wrapper():
        source = SOURCE.with_name("fresh-f03-payload-release-20261107.py")
        spec = importlib.util.spec_from_file_location("f03_release_wrapper", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_unbound_or_duplicate_serving_cids_are_rejected_without_io(self):
        module = self.load_wrapper()
        for plan in ({"status": "unbound"}, {"status": "ready_for_exact_preview",
                     "f04_serving": {"service_cids": {f"gemma-{i}": "a" * 64 for i in range(4)}}}):
            with self.subTest(plan=plan), mock.patch.object(module.urllib.request, "urlopen") as requested:
                with self.assertRaises(ValueError):
                    module.validate_serving_layout(plan)
                requested.assert_not_called()

    def test_wrong_wrapper_pin_rejects_before_delegate_or_lock(self):
        module = self.load_wrapper()
        with mock.patch("sys.argv", ["wrapper", "--plan", str(module.PLAN_ROOT / "plan-ready.json"),
                                    "--plan-sha256", "a" * 64, "--expected-wrapper-sha256", "0" * 64,
                                    "--output", str(module.ROOT / "analysis/never-created"), "--apply"]), \
                mock.patch.object(module.importlib.util, "spec_from_file_location") as loaded, \
                mock.patch.object(module.os, "open") as opened, \
                mock.patch.object(module.os, "unlink") as unlinked:
            with self.assertRaisesRegex(ValueError, "wrapper/helper source pins"):
                module.main()
            loaded.assert_not_called()
            opened.assert_not_called()
            unlinked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
