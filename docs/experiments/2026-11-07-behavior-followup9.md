# Nine independent follow-up contracts after F03

Status: **all nine authored correct programs pass all144 fixtures; every foil fails a fixed counterexample**. One authorized CPU sandbox run completed after F04 launch. No model generation, additional baseline, gradient or candidate calculation has occurred. The existing behavior12 is retained. These nine new problems use independent public contracts; no benchmark question text, reference program or hidden test input is copied.

| Family | Train (two independent singleton folds) | Held-out transfer (one) | Deliberately wrong foil |
|---|---|---|---|
| G: every-stage constraint | G1 initial/intermediate balance boundaries; G2 reachable integer interval at every stage | G3 every visited automaton state must be permitted | Check only final balance; let an empty reachability interval heal; forget earlier forbidden state |
| H: explicit discretization | H1 adjacent interval secants; H2 centered interior-point slopes | H3 sum absolute adjacent interval slopes | Substitute sample-point gradient; use one-sided slope; aggregate sample gradients |
| I: frequency and sentinel | I1 unique mode, −99 for ties/empty; I2 unique second DISTINCT frequency, −88 otherwise | I3 unique mode with inclusive minimum frequency, −77 otherwise | Resolve tied modes/runners-up numerically; change inclusive frequency threshold to strict |

[Case/fixture source](../../scripts/experiments/gemma4-purpose/behavior_followup_cases.py) reuses the existing `Case` schema. [Train6](records/2026-11-07-behavior-followup9/train-cases.json) is the same raw purpose/fold mapping as existing recipes; [transfer3](records/2026-11-07-behavior-followup9/transfer-cases.json) has `gradient_allowed=false` and remains ungenerated. No selection or direction will use transfer outputs. All nine definitions, correct answers, foils and fixture plans are fixed before model output.

The [one-off validation driver](records/2026-11-07-behavior-followup9/validate-authored-once.py.txt) calls existing `behavior_offline.score` and `science_sandbox.Docker.run` without changing either. Each correct/foil gets eight fixed boundary fixtures plus eight seeded fixtures (18 code executions; 288 trial outcomes). Independent expected values use sequential Python accumulation, enumerated integer reachable sets, direct secant arithmetic and `Counter`. Authored correct/foil programs are not executed on the host.

The planned sandbox is the existing immutable c4ab502… image, explicit host daemon3e518d2b… on NVMe, network none, no host mounts/GPU, uid1003:1003, read-only rootfs, all capabilities dropped, CPU1/512MiB/15s. Only CPU scheduling shares are lowered to128 during the running F04 benchmark. Start/end epochs, WSP state and protected running container IDs will record overlap. Execution waited for F04's formal launch to preserve the release/deployment container-set guard.

The [preview](records/2026-11-07-behavior-followup9/preview.json) pins all sources, full split inputs and each code/fixture identity; it made zero Docker/model calls. Actual tokenizer length is not measured (target≤1024, no truncation allowed). Initial Ruff found two intended constant re-exports marked unused; explicit aliases fixed that source-only issue. Ruff and diff-check now pass. Runtime failures and every foil/correct trial will be retained, with unknown distinguished from numerical failure. The frozen preview predates execution and remains unchanged.

## One completed authored validation

[Full raw worker/trial report](records/2026-11-07-behavior-followup9/validation-report.json.gz) and [compact receipt](records/2026-11-07-behavior-followup9/validation-summary.json): 18 Docker calls, 288 trial outcomes, correct144/144, every foil rejected by a fixed numeric counterexample, unknown0, retries0, empty stderr. Wall time10.988155s. All pre-existing running containers remain present. F04 overlap epoch1794051738.7987063–1794051749.7868617; WSP was running at both ends (progress still in loading at85.9s). This records overlap and does not infer zero timing impact.

| Case | Correct pass/16 | Foil pass/16 | Fixed foil witness rejected |
|---|---:|---:|---|
| behavior-G1 (train) | 16 | 13 | yes |
| behavior-G2 (train) | 16 | 14 | yes |
| behavior-G3 (transfer) | 16 | 6 | yes |
| behavior-H1 (train) | 16 | 1 | yes |
| behavior-H2 (train) | 16 | 2 | yes |
| behavior-H3 (transfer) | 16 | 1 | yes |
| behavior-I1 (train) | 16 | 14 | yes |
| behavior-I2 (train) | 16 | 11 | yes |
| behavior-I3 (transfer) | 16 | 10 | yes |

These are fixture checks of newly authored programs, not AV/F04 model capability scores. Transfer3 was used only for authored validation; no model generation/gradient/optimization used it. Source, full-input, per-code payload, fixture and raw-result hashes remain in preview/fullreport/SHA256SUMS. The only pre-execution source adjustment was the recorded unused-import alias fix; no case/foil/fixture changed after preview or in response to execution.
