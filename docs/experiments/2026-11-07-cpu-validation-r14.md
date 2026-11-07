# CPU validation r14

The required CPU checks pass after one format-only Ruff correction. The full unit
suite ran once: **592 tests = 532 passed + 60 skipped**, no failures; 45.932s
reported / 46.412s wall. The initial check sequence took46.85985s.

Initial Ruff returned five E702 errors in the reused validation driver's
semicolon-separated statements. This is retained as an actual failed check:
original completion.json still has success=false. Root formatted that driver,
proved its AST unchanged, and reran **only Ruff**, which passed in0.03450s.
The independent packaging check recomputed the before/after AST equality.
corrected-summary.json is a separate aggregate; it does not rewrite the failure.

Mypy passed all74 source files, the model-comparison data check passed, and all31
tracked shell files passed syntax checks. Unit tests, mypy, data and shell checks
were not repeated. The622 source identities (including288 immutable record
sources) matched before/after the initial run, and all explicit experiment source
pins remained unchanged. Only the CI driver's formatting changed afterward.

The low-priority run used the existing CPU environment and short NVMe temporary
directory. GPU/model/service operations0. Skips remain skips, not validations of
real-model training or CUDA behavior.

[All logs, executed/formatted driver sources, before/after source inventories,
format diff, independent AST proof, original failure and corrected summary](records/2026-11-07-cpu-validation-r14/)
retain original-byte hashes; larger logs/inventories use lossless gzip.
The packaging source and SHA256SUMS.json make those representations explicit.

This run includes the [body/KL opt-in and separate parent/serving owner contract](2026-11-07-body-kl-training-implementation.md),
the [all204 mask proof, including failed v1 and successful v2](2026-11-07-mbpp204-body-frame-proof.md),
and the [preview-only plan builder](scripts/fresh-body-kl-plan-20261107.py).
It establishes CPU implementation checks, not a new model result or an
improvement over the completed official benchmark.
