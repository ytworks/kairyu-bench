# CPU validation r13

One complete pass after the explicit progress-CLI fix passed: **527 PASS + 59 skipped = 586 unittest tests**, zero failures/errors. Unittest reported 47.127s (47.590810s wall); the whole CPU check pass took 48.041790s. Full Ruff, mypy on 74 source files, generated comparison data, and all 31 tracked repository shell syntax checks passed. No check was repeated.

All 590 pinned source/config files matched before and after. This includes the live monitor, existing training/merge/owner sources, and 264 immutable record-source snapshots present at start, covering archive and binding one-offs. The immutable snapshots were hashed for preservation, not executed or relinted as live source; the existing records-only Ruff exclusion was not changed. New concurrent Markdown and packaged records were outside the source continuity inventory. Source hashes, all command arguments, and complete stdout/stderr are retained.

The existing CPU venv, nice15, one-thread numerical libraries, and a short NVMe TMPDIR were used. The shell check was the explicitly requested tracked-source scope, not a recursive traversal of ignored runtime caches. Optional unavailable-runtime skips remain counted, with reasons in the summary. This pass does not certify optional GPU/serving/runtime paths. The driver performed no model request, GPU work, Docker worker, or service operation.

[Summary, full raw/losslessly compressed logs, before/after pins, drivers, and SHA256 index](records/2026-11-07-cpu-validation-r13/) preserve the actual result. Prior r12 and focused CLI records remain separate and unchanged.
