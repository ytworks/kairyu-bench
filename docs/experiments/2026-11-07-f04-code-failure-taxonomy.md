# F04: observable failure taxonomy before F06 results

Existing full dev30 records contain 13 failed standalone coding items (HumanEval+ 1, LiveCodeBench 5, MBPP+ 7) and 18 failed SciCode steps out of 39. This audit covers every one of those 31 failures; the other 21 SciCode steps are retained in the JSON. It reads saved public code, verdicts, finish fields and stream metadata. It makes **0 model calls, 0 scoring calls and 0 generated-code executions**, and creates no training data from benchmark material.

The conservative primary classification is **1 definite entry-point/format failure, 13 functional-assertion hints, 17 unknown causes**; explicit length/timeout/syntax failures are 0 in the retained fields. All 31 finish with `stop` and their extracted/integrated code parses. This does **not** prove zero per-test timeouts: EvalPlus folds internal exceptions/time limits into `fail`, and LiveCodeBench stores only `passed`. No speculative algorithm cause is promoted to an observed verdict.

The one definite entry-point failure is MBPP126: cached public contract requires `sum`, whereas the answer defines `sum_common_divisors`. The pinned sanitizer produces only `import math` (11 bytes), so the required callable is absent. For SciCode, all 18 have return code 1 and `timed_out=false`: 13 AssertionError, 4 ValueError, 1 IndexError allowlisted hints. Full stdout/stderr are **not** retained—only size/SHA and the allowlisted class. An assertion hint does not identify the failing input, distinguish generated assertions from harness assertions, or prove the current helper caused it.

| Failed item/step | Primary class | Retained evidence / bounded interpretation |
|---|---|---|
| HumanEval/151 | unknown | base pass / plus fail; exact-type int filter; failing input unavailable. |
| arc191_c | unknown | binary fail; bounded prime/order search; no retained per-test reason. |
| abc390_g | unknown | binary fail; nested convolution and coefficient recurrence; timeout not established. |
| arc196_a | unknown | binary fail; sorted parity pools/extreme pairing; feasibility hypothesis unverified. |
| arc190_c | unknown | binary fail; last of 2 fenced blocks selected correctly; full grid DP per query. |
| abc392_d | unknown | binary fail; pairwise frequency overlap; no concrete cause retained. |
| Mbpp/801 | unknown | base+plus fail; all-distinct maximum frequency is 1; sentinel mismatch is a hypothesis. |
| Mbpp/410 | unknown | base pass / plus fail; numeric filtering then min, without empty guard. |
| Mbpp/576 | unknown | base pass / plus fail; contiguous slice and empty-sublist=True interpretation. |
| Mbpp/129 | unknown | base pass / plus fail; adds complete 1..n² membership requirement. |
| Mbpp/103 | unknown | base pass / plus fail; boundary branches and quadratic DP; exact cause not saved. |
| Mbpp/639 | unknown | base pass / plus fail; first character must be uppercase, not merely non-lowercase. |
| Mbpp/126 | entry-point failure (high) | base+plus fail; required sum absent after pinned sanitizer (high static confidence). |
| Sci 18.1 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.249s. |
| Sci 18.2 | unknown | IndexError hint; rc=1, non-timeout, 2.716s. |
| Sci 45.1 | unknown | ValueError hint; rc=1, non-timeout, 2.698s. |
| Sci 45.2 | unknown | ValueError hint; rc=1, non-timeout, 2.315s. |
| Sci 45.3 | unknown | ValueError hint; rc=1, non-timeout, 1.933s. |
| Sci 45.4 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.594s. |
| Sci 72.5 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.322s. |
| Sci 80.3 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.363s. |
| Sci 80.4 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.574s. |
| Sci 80.5 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.569s. |
| Sci 80.6 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.706s. |
| Sci 80.7 | unknown | ValueError hint; rc=1, non-timeout, 3.250s. |
| Sci 9.1 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.107s. |
| Sci 17.1 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.566s. |
| Sci 17.2 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.080s. |
| Sci 46.3 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.223s. |
| Sci 46.4 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 2.892s. |
| Sci 35.3 | assertion hint (medium) | AssertionError hint; rc=1, non-timeout, 1.960s. |

SciCode files include previous generated helpers. A failed helper can affect later steps, but propagation is not automatic: 72.5 fails and 72.6–72.9 pass. For 45 and 80, the saved exception hints cannot isolate an earlier helper, the current function, shape/dtype incompatibility or a harness assertion. Weight norms/gradient correlation cannot resolve that missing evidence.

Rules were frozen before reading the F06 result and use precedence: explicit length → explicit timeout → missing required entry point → empty/invalid extracted code → Sci AssertionError hint → unknown. The cached official EvalPlus sanitizer and LiveCodeBench OpenAIChat extractor were called only as text parsers. Their current source hashes are pinned; the original scorer payload was temporary, so parser reconstruction is labeled static evidence rather than a retained execution trace. All 5 WSP adapter/scorer files match the archived f91 source byte-for-byte.

The 12 ordinary unknowns retain all official rejection evidence; “unknown” does not mean unscored. All 31 underlying outcomes were scored failures. The near-ceiling authored set and this benchmark set cover different failure modes; this audit selects no new objective, candidate, or benchmark-derived training example.

Execution: one CPU analysis, 1.549 s, exit 0; Ruff PASS. No broad tests, services, GPU work, benchmark rerun or parent regeneration. Initial guessed file paths were absent and one exploratory display truncated; these read limitations are preserved in the receipt, and the final extraction asserts the complete 13+18 scope.

- Source: [fresh-f04-code-failure-taxonomy-20261107.py](scripts/fresh-f04-code-failure-taxonomy-20261107.py), SHA `a707f10599abb8897d44d62e04c53fe43d19312d6bb0f5b62dcbfbfb733df8fd`.
- Records: [report.json](records/2026-11-07-f04-failure-taxonomy-v1/report.json), SHA `a435a8dd742f2b2e35aea47b76fa8e1f847f28497eaa7533c75a5a80ce86b338`; [rules.json](records/2026-11-07-f04-failure-taxonomy-v1/rules.json); [receipt.json](records/2026-11-07-f04-failure-taxonomy-v1/receipt.json). Per-row canonical file/line/raw-line SHA, selected public/integrated-code SHA, stream metadata, all 39 Sci outcomes, and exact input/source pins are included.
- Canonical input `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/results/gemma4-fresh-f04-dev30-dp4-20261107-v1/items.jsonl` SHA `112225547aa681dad402e7a3e62a2cbcafd15301e84987ad4610f2b08fffba65`. Official report SHA `1a062c114b0acae4d27930040c30ccc2e01a20c3e1062490fa1f84656278fd85`.
