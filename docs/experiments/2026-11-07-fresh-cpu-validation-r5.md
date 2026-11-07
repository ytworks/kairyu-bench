# CPU validation after the complete-run audit

The complete CPU suite passed: **525 collected = 469 passed + 56 optional-dependency
skips**, no failures, 46.347 seconds reported by unittest (46.855 seconds including
process startup). The previous suite had 522 tests; three concrete evidence-boundary
tests were added, none removed. They preserve actual zero versus unknown/error,
retain missing/duplicate/unexpected records, and serialize nonfinite evidence
without replacing it by zero. The initial narrow test run exposed a JSON newline
escape defect in the audit writer; that failed preparation and the corrected
three-test run remain in the independent-audit method record.

Full mypy, data consistency, whole-tree Ruff and all 31 shell syntax checks passed.
The audit source and its test SHA remained identical before and after the suite.
[Receipt, exact commands, timings, source pins and log hashes](artifacts/fresh-cpu-validation-r5-20261107.json)
are retained. The checks used the existing NVMe CPU environment, its bin on PATH,
repository src on PYTHONPATH, and the previously validated short NVMe TMPDIR.
Unittest, mypy and data checks ran concurrently with separate logs. No model call,
GPU numerical validation or service operation is implied by these CPU checks.

Run recipe from the remote worktree (logs below are under a new validation directory):

```sh
export PATH=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin:$PATH
export PYTHONPATH=$PWD/src
export TMPDIR=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/tmp-ci-r3
python -m unittest discover -s tests -v
mypy --ignore-missing-imports src
python scripts/update_model_comparison.py --check
ruff check .
```

Every path returned by `rg --files --hidden -g '*.sh' -g '!.git'` was checked
individually using `sh -n`; the receipt retains all 31 exact paths and return codes.
The last pushed evidence commit fe51456 also passed all four GitHub checks
(CPU, DeepSWE runtime, Terminal-Bench runtime, Docker). Newer additions require
their own CI results.
