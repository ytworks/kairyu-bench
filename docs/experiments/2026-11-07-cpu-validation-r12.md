# CPU validation r12

After the readonly private-copy dense-merge fix, the frozen sources (including the new training-log audit) passed one complete CPU check pass: **524 PASS + 59 skipped = 583 unittest tests**, zero failure/error, reported 49.742s (wall 50.204576s). Full Ruff, mypy on 74 source files, generated comparison-data verification and all31 tracked repository shell syntax checks passed. Total time was 50.653950s; no full-suite or individual-check repetition occurred.

All322 monitored Python/config/tracked-shell sources matched before/after, including the explicit trainer `8bfa6f3f5568264417d52e1cf297ebf0f988458c0c075f329a189a0233a7a846`, dense merger `7660758dd18712813cfdfcb493a621fd5e9a0d92b20eb0b541a9a4f54f1cd515` and owner `3e7e9c58b68a679d87623e1507e6508adcc731c3da36d51dd390f9bffcdb2b07`. The new training-log audit SHA `988a0de1ee1c00d60ee6ce96e44d71d8d86a0aa140d1fc47dbab1c27d74a629e` was additionally pinned. Concurrent Markdown/record additions were excluded from source continuity. r11 remains a distinct earlier result on the pre-fix merge source; it was not overwritten. Execution used the existing CPU venv, nice15, one-thread numerical libraries and the short owned NVMe TMPDIR.

The shell step followed the parent's explicit tracked-source scope: every tracked `.sh` was checked with `sh -n`. This covers the repository shell sources; it does not claim that a recursive `find` over unrelated ignored runtime caches was rerun. The r10 cache-permission failure remains intact in its earlier record.

Optional unavailable-runtime skips remain explicit with counts/reasons in summary and full log. This CPU pass does not certify a real GPU/serving run. No model, GPU, Docker worker or service operation was performed by this validation driver; any separately owned materialization or service was not touched.

[Exact drivers, complete logs, command receipts, before/after source identities and raw/compressed SHA256](records/2026-11-07-cpu-validation-r12/) are retained. Logs above64KiB use lossless gzip; no ignored raw .log needs force-add.

The already completed PR33 body v7 update was previously copied byte-for-byte during r11 and remains unchanged in [its records](records/2026-11-07-pr33-body-update-f05-v7/): body SHA `58fe5525d2895ccd71a71efa5416774c4e63f8ceb51c6147eb0b82601d22157a`, existing PATCH receipt `returned_equal=true`. This copy did not issue another external API update.
