# CPU validation r10

The changed train19 composer, generation/frame preparation and native GPU owner passed the full CPU unittest suite: **516 PASS +57 skipped = 573 tests**, zero failure/error, reported46.111s (wall46.543245s). Full Ruff, mypy74 files and generated comparison-data check passed. All31 Git-visible repository shell files passed individual sh -n. There was one full suite run and no test rerun.

The exact CI find-based shell step initially returned1 because this long-lived remote worktree contains foreign-owned, gitignored benchmark cache directories that cannot be traversed. No shell syntax error was reported. That original failure is preserved; no permissions or sources were changed. The subsequent explicit check covered every tracked and nonignored untracked repository shell file, matching the source files present in a clean checkout. This is recorded separately and does not rewrite the failed find command as successful.

Execution used the existing CPU virtualenv, its PATH prefix, short owned NVMe TMPDIR, nice15 and one-thread numerical libraries. All311 monitored source/config/shell files matched before/after. No model, GPU, Docker worker or service operation occurred. Optional unavailable-runtime tests remain the57 explicit skips; these CPU tests do not claim a real GPU validation.

[Commands, driver, full logs, before/after identities, cache error and per-shell checks](records/2026-11-07-cpu-validation-r10/) are preserved with raw/stored SHA256; large logs use lossless gzip. Source was not changed in response to this run.
