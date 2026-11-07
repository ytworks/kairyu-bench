# N30: independent audit of completed AV canonical dev30

2026-11-07. **Code: 72.9365079365; overall: 76.7200592915.**
All **2,338/2,338 items and 90/90 code items** are scored; zero missing IDs,
duplicates, unknown scores, item errors or unscored items. Every frozen budget,
sample count, kind/source/area and response ID matches. The saved report
exactly equals `build_report` reconstruction; aggregate discrepancy is zero.
This is a new AV dev30 baseline, not a matched candidate comparison.
The >90 code objective remains unmet. F02's dev10 tie remains NO ADVANCEMENT;
tier10's three-source code score is not ranked against this four-source score.

| Official code source | Full / partial / zero | Score |
|---|---:|---:|
| HumanEval+ | 29 / 0 / 1 (30) | 96.666666667 |
| MBPP+ | 23 / 0 / 7 (30) | 76.666666667 |
| LiveCodeBench | 15 / 0 / 5 (20) | 75.000000000 |
| SciCode | 2 / 4 / 4 (10) | 43.412698413 |

Code is the **equal mean of these four source means**, not the 90-item mean
or a pooled SciCode-step fraction. All 90 generations finished `stop`,
with no token-cap truncation. [Every code item](evidence-2026-11-07-n30/code-90-items.json),
[all 2,338 numeric records](evidence-2026-11-07-n30/all-2338-numeric-items.json),
[all 32 sources/full summary](evidence-2026-11-07-n30/summary.json) and
[absence/unknown checks](evidence-2026-11-07-n30/data-anomalies.json) are retained.

As arithmetic constraints, fixing SciCode alone to 100 while keeping the other
three scores unchanged would yield only **87.0833333333** code. Making only
the other three sources perfect leaves **85.8531746032** with current SciCode.
Both scientific composition and other code failures therefore matter to the
90-point objective. These ceilings are not predicted gains from any edit.

## Fourteen areas and diagnostics

| Area | Score |
|---|---:|
| knowledge | 85.555555556 |
| expert | 83.333333333 |
| facts | 71.666666667 |
| math | 91.258012821 |
| code | 72.936507937 |
| instructions | 89.916666667 |
| tools | 79.166666667 |
| agents | 54.166666667 |
| long-context | 84.837524373 |
| reasoning | 84.130434783 |
| commonsense | 61.666666667 |
| multilingual | 78.872053872 |
| vision | 79.166666667 |
| safety | 57.407407407 |

Diagnostic NLL (not capability points): code **4.022500118458066**,
Japanese Wikipedia **6.628099445956357**, Wikitext **7.638729488031108**.
These dev30 selections differ from dev10; no cross-tier NLL gain is claimed.

## SciCode: all ten problems and all 39 tested steps

21/39 steps passed, but the official source score averages the ten **problem
fractions**, yielding 43.412698413 rather than the pooled 53.846153846%.
There were 18 failed steps: 12 AssertionError, two ValueError, two
AttributeError, one TypeError and one IndexError; zero step timeouts.
Exception names are allowlisted stderr hints, not proven causal diagnoses.
[All 39 step records](evidence-2026-11-07-n30/scicode-39-step-numeric.json)
retain return codes, seconds, stdout/stderr byte counts and hashes.

| Problem | Passed steps | Item score | Failed steps / exception hint |
|---|---:|---:|---|
| scicode:14 | 2/2 | 100.000000000 | none |
| scicode:58 | 5/5 | 100.000000000 | none |
| scicode:18 | 0/2 | 0.000000000 | 18.1 TypeError, 18.2 IndexError |
| scicode:45 | 0/4 | 0.000000000 | 45.1 ValueError, 45.2 AttributeError, 45.3 AttributeError, 45.4 AssertionError |
| scicode:72 | 8/9 | 88.888888889 | 72.5 AssertionError |
| scicode:80 | 2/7 | 28.571428571 | 80.3 AssertionError, 80.4 AssertionError, 80.5 AssertionError, 80.6 AssertionError, 80.7 ValueError |
| scicode:9 | 0/1 | 0.000000000 | 9.1 AssertionError |
| scicode:17 | 0/2 | 0.000000000 | 17.1 AssertionError, 17.2 AssertionError |
| scicode:46 | 2/4 | 50.000000000 | 46.3 AssertionError, 46.4 AssertionError |
| scicode:35 | 2/3 | 66.666666667 | 35.3 AssertionError |

## Runtime, requests and steps

Actual runner is f91
`sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8`.
All four ordinary replicas are bound by launch/closure receipts to AV tree
`5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`,
TP2 x DP4, MTP4, with chat/logprob/agent concurrency **256/32/4**.
Chat used the homogeneous gateway and logprobs used all four same-weight
endpoints. No selection exclusion or activation worker was attached:
observation here is **HTTP client tracing only**. Current report/catalog/
manifest/code/SciCode scorer SHA values equal the immutable f91 source archive.

Wall **2,018.9 s (33.6483 min)**, load **111.2 s**. Both saved wall and
generation counters start before dataset loading; neither is pure generation
time. Terminal-Bench finished at 2,018.8 s and RULER at 1,861.0 s.
Reported WSP API counters are 1,754 chat / 9,193 logprob calls.
The client trace has **4,850 request records plus 10 non-request records**,
zero errors/retries and no other run IDs. These counters have different
instrumentation scopes and are not equated.

Source-report usage: **5,335,614 input / 4,094,068 output tokens**.
Separate Terminal agent totals: **426,433 / 109,919**.
SWE saved assistant-message usage: **1,902,178 / 31,789**; these cover
**123 persisted assistant responses against 125 model-stat API calls**.
Ansible has 41 responses / 42 calls and OpenLibrary 16 / 17; the two missing
response usages remain unknown, not zero-filled. No exact whole-run token
total is claimed by blindly adding differently scoped counters.

| Agent task | Reward | Step/call counters | Seconds |
|---|---:|---|---:|
| bn-fit-modify | 1 | 10 episodes / 18 tool calls | 172.238 |
| cobol-modernization | 1 | 6 episodes / 35 tool calls | 430.891 |
| install-windows-3.11 | 0 | 13 episodes / 38 tool calls | 851.843 |
| polyglot-c-py | 1 | 7 episodes / 18 tool calls | 413.008 |
| portfolio-optimization | 1 | 7 episodes / 13 tool calls | 309.334 |
| reshard-c4-data | 1 | 14 episodes / 44 tool calls | 1416.782 |
| ansible | 0 | 42 API / 41 assistant / 40 tool | 1222.441 |
| qutebrowser | 0 | 42 API / 42 assistant / 41 tool | 1144.543 |
| NodeBB | 0 | 24 API / 24 assistant / 23 tool | 100.203 |
| openlibrary | 1 | 17 API / 16 assistant / 15 tool | 120.383 |

Terminal seconds are official trial timestamps. SWE seconds are the existing
harness's selection-file to verdict-file mtime span, not a new precision timing
measurement. All four SWE trajectories ended Submitted. Agent tool/call counts
retain their distinct meanings; counts are not a shared abstract step unit.

## Actual failure dependencies and uncertainty

The following uses only public prompts, generated public code, the existing
official sanitizer, and saved results. No reference implementation/test answer
was read into a calibration set. No generated solution was executed again.
The [exact one-off reader](records/n30-contract-reader.py.txt) and
[compact contract/AST facts](evidence-2026-11-07-n30/public-contract-map.json)
are retained. Direct-name AST edges are incomplete for class methods, callbacks,
aliases and redefinitions; absent edges do not prove absence of dependency.

**HE/MBPP: extraction and API adherence precede algorithm correctness.**

| Failed item | Observed evidence and limit |
|---|---|
| HumanEval/152 | Two `compare` definitions: an erroneous `abs(generator)` followed by a correction. The actual EvalPlus sanitizer retains the first definition and removes the correction. This is an observed output/extraction interaction, not a hypothetical numerical weakness. |
| Mbpp/641 | Prompt requires `is_nonagonal`; code defines `nonagonal_number`. The required callable is absent after actual sanitization. |
| Mbpp/126 | Prompt requires `sum`; code defines `sum_common_divisors`. Actual sanitized output is only `import math`. |
| Mbpp/410 | Base passes, plus fails. Generated contract narrows heterogeneous inputs to int/float and returns None otherwise. The failing plus input is not identified; a specific type-policy cause is unproven. |
| Mbpp/576 | Base passes, plus fails. Code chooses contiguous slicing and defines empty-sublist True; the brief public prompt does not settle every boundary convention. Exact failed condition remains unknown. |
| Mbpp/129 | Base passes, plus fails. Adds uniqueness and exact 1..n² range restrictions beyond checking equal row/column/diagonal sums. This exposes specification-interpretation dependence; it does not identify the stored failing test. |
| Mbpp/103 | Base passes, plus fails. Defines A(0,0)=1 and other out-of-range behavior. Boundary convention / resource behavior remains unresolved from the saved pass/fail only. |
| Mbpp/639 | Base passes, plus fails. Implements “starts uppercase” instead of “does not start lowercase”; these differ for non-letter initial characters. This is a concrete predicate difference, not proof of which plus test failed. |

**LCB: five different algorithm/output questions remain, not one generic code skill.**

| Failed item | Observable implementation and uncertainty |
|---|---|
| arc190_c | Transposes the grid when H<W, swaps position/dimensions, but continues applying original L/R/U/D to the transposed axes. Coordinate transformation and update recurrence must agree; this inconsistency is visible without a gold answer. Multiple abandoned solve variants also remain in public code. |
| 3777 | Removes zeros before alternating-sum DP. Zero-valued selected elements still change subsequent parity; for example the valid subsequence [1,0,2] has alternating sum 3 and product 0, which the zero-filtered state cannot represent. This is a public-domain counterexample, not an official hidden test. |
| arc191_c | Searches primes kN+1 and candidate bases up to one million inside an unbounded outer search. Finite resource/constructive-proof risk is visible, but the saved binary verdict does not identify timeout versus wrong output. |
| arc196_a | Sorts original odd/even-position values to optimize adjacent deletions, discarding order. The validity of that reduction and its failing case remain unresolved; no causal label is asserted from the binary verdict. |
| abc392_d | Implements pairwise matching face frequencies divided by face-count products, consistent with the stated probability expression. Exact failure remains unresolved (including possible runtime/checker/precision effects); it is not evidence of a proven probability-concept defect. |

**SciCode: inspect contracts and actual calls, not blanket propagation.**
18.1 returns scalar 0/1 branches where its header describes a 1-D array, and
indexes `Xi[xi]`; 18.2 treats coordinates as floats and reimplements a local
B-spline rather than calling 18.1. Type/index failures are consistent with
interface/shape questions, but the exact failing operand is not in the saved
stderr hint. In 45, code assumes particular grid-axis order and ndarray
`shape/size` interfaces; 45.4 reimplements the earlier initialization and
boundary updates. Early failures are therefore not automatically their cause.
No general dtype defect can be established from these records alone.

35.1/35.2 pass while 35.3 fails and reimplements energy/wavelength composition
without directly calling those helpers. 46.1/46.2 pass, 46.3 fails, and 46.4
directly calls the failed `metropolis`. In 80, failed 80.3 directly calls
successful `dist/E_ij`; later 80.6/80.7 use failed helpers. These observations
separate primitive correctness, their composition, and possible propagation.

Critically, **72.5 fails but 72.6–72.9 pass**. 72.6 actually calls the failed
function but unpacks `prob, _`, discarding its second return value.
This explains why a failed whole-function test need not imply every dependent
test fails; it does not prove that the discarded value caused 72.5's failure.
9 and 17 also fail without a simple failed-helper chain. Numerical stopping,
symbol/container contracts and problem semantics remain unresolved there.

The measurements support examining API/identifier preservation, output
sanitization, exact predicates and state representation, array/container
contracts, and per-output numerical composition as separate dependencies.
They do not localize any tensor/head or justify a specific weight edit yet.
Existing gradient proxies have not established these execution properties;
the next authored generation/execution measurement must preserve these
distinctions and all failures.

## Audit reproduction and preparation failures

The [audit source](scripts/fresh-n30-complete-audit.py) reuses `audit_run`
and `build_report`. It writes all canonical IDs and actual records before
strict completeness checks. Null/nonfinite scores, missing/duplicate rows,
unexpected IDs and API errors are retained as unknown/anomalous with no zero
imputation. A completed run containing unknowns receives
`verified_complete=false` and `full_score_ranking_permitted=false`.
Runtime mismatches also retain previously written item evidence and a failure
receipt. These branches did not trigger for N30.

CPU boundary tests initially had two passes and one JSON-read error caused by
a literal escaped newline in the new serializer. After correction, all three
passed; the tests cover real zero versus unknown/API failure, duplicate/missing/
unexpected records, and NaN/Inf preservation. Ruff/diffcheck passed.
The parent runs repository-required CI separately.

Independent audit v1 succeeded in 3.903 s. v2 succeeded in 3.702 s and preserves
the same full scores, adding the nested agent time fields and the explicit
125-call/123-response usage limitation. v1 summary/source remain in the PR
evidence. These are CPU rereads of one fixed WSP run, not two model trials.
The digest program is retained [verbatim](records/n30-result-digest.py.txt).
The public-contract reader completed once with returncode0 and no stderr.
A commentary-display attempt encountered a JavaScript quoting failure before
SSH execution; no model or file operation occurred in that attempt.

Final summary SHA:
`9d4af6e386b580dd9bf411f882bb14ad49a95292b3543f89d10436b89bd1ba31`.

```bash
R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
B=$R/worktrees/kairyu-bench-weight-tools
PYTHONPATH="$B/src" "$R/infra/weight-tools-ci/bin/python" \
  "$B/docs/experiments/scripts/fresh-n30-complete-audit.py" \
  --project "$R" --repository "$B" \
  --study "$R/analysis/fresh-purpose-20261107-av-dev30-dp4-v1" \
  --run "$R/results/gemma4-fresh-av-dev30-dp4-20261107-v1" \
  --output "$R/analysis/fresh-purpose-20261107-av-dev30-dp4-v1/independent-audit-v2"
```

The shown output already exists; any future reread requires a fresh directory.
