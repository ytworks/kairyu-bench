# 2026-11-06: official Gemma 4 base, observed WSP dev10

This is a fresh **official full dev10** run: 778 items, 29 sources and all
14 areas. It is not the earlier 95-item screening subset. No test/holdout
responses were used in this analysis.

## Identity and elapsed time

- Run: gemma4-base-observed-dev10-20261106.
- Older unobserved base reference: wsp-base-10-dev-1-20261104.
- Official Gemma 4 31B instruction checkpoint tree SHA-256:
  cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9.
- WSP manifest SHA-256:
  a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b.
- Native thinking **on**, without an additional thinking cap; vLLM 0.33,
  four TP2 replicas, MTP4. This experiment's pinned Kairyu/vLLM serving
  deployment rejected numeric reasoning_effort 100. That is a finding about
  the tested API/runtime controls, not a model-wide claim. Official WSP
  per-item maximum-output-token settings were preserved.
- Started 2026-11-05T16:05:18.901106Z; finished
  2026-11-05T16:15:33.029372Z (November 6 in Japan).
- Official status completed; **778 scored, zero unscored**.
- Reported wall time **613.8 s**. Timestamp difference 614.128 s includes
  artifact writing outside the measured interval.
- Dataset loading **106.8 s is within that wall time**, not additional.
  In [run.py](../../src/kairyu_bench/wsp/run.py), both generation_seconds
  and wall_seconds start before loading. Post-load elapsed is about 507.0 s.
- API counters: 558 chat calls and 3,062 logprob-client calls. These are
  different counting units from observed engine requests: tokenization and
  nested agent requests follow different paths.

## Official scores and repeatability

These are official area aggregates, not pooled fractions of all 778 items.
Changes are percentage points versus the older **same base** dev10 run.

| Area | Old base % | Observed base % | Change |
|---|---:|---:|---:|
| knowledge | 88.3333 | 90.0000 | +1.6667 |
| expert | 87.5000 | 81.2500 | −6.2500 |
| facts | 65.0000 | 70.0000 | +5.0000 |
| math | 97.5000 | 98.7500 | +1.2500 |
| code | 84.4444 | 87.7778 | +3.3333 |
| instructions | 95.4167 | 88.7500 | −6.6667 |
| tools | 87.5000 | 87.5000 | 0 |
| agents | 50.0000 | 50.0000 | 0 |
| long-context | 81.3779 | 81.3436 | −0.0343 |
| reasoning | 79.1304 | 71.9565 | −7.1739 |
| commonsense | 66.3333 | 67.3333 | +1.0000 |
| multilingual | 80.9848 | 80.9848 | 0 |
| vision | 87.5000 | 81.2500 | −6.2500 |
| safety | 86.1111 | 86.1111 | 0 |
| **Overall** | **81.2237** | **80.2148** | **−1.0089** |

| Source | Old base | Observed base |
|---|---:|---:|
| HumanEval+ | 10/10 | 10/10 |
| MBPP+ | 7/10 | 8/10 |
| LiveCodeBench | 5/6 | 5/6 |
| BFCL | 35/40 | 35/40 |
| IFBench | 14/15 | 12/15 |
| IFEval | 39/40 | 39/40 |
| Terminal-Bench | 1/2 | 1/2 |

MBPP/801 is the sole changed code verdict: the new **unmodified base** passes
it. Its code score, 87.7778%, equals the earlier final-norm +1% candidate's
apparent code gain. Thus the earlier small gain is not established as a
causal effect of the weight edit. The entire BFCL verdict vector is
unchanged. New candidates should be compared with this measured baseline;
historical small gains alone do not justify an improvement claim. These
data do not identify whether remaining repeat variation comes from batch
scheduling, sampling, observation overhead, or another condition.

## Public failure details

Only public tasks, official expected behavior, final code/tool calls, and
grading metadata were inspected. Hidden reasoning was not extracted.
Each code task had one answer. API steps count model requests, not the
number of tool calls inside a request.

| Problem | Requirement and observed error | Steps / official evidence |
|---|---|---|
| MBPP/410 | Find the smallest **integer** in a heterogeneous list. The answer also admits floats. An official additional input with integers 10, 5 and floats 2.5, 2.5 should return 5; the submitted logic returns 2.5. | 1; base tests pass, plus tests fail. |
| MBPP/576 | Test subsequence membership, allowing gaps. The answer tests contiguous slices instead: [1,3,5] should be a subsequence of [1,2,3,4,5]. | 1; base tests pass, plus tests fail. |
| LiveCodeBench/arc191_c | Construct any valid pair (A,M) of the required multiplicative order N. The fixed grader can reject a different valid construction; see below. | 1; passed:false, failed-case identity absent. |
| BFCL/multi_turn_base_29 | Enter VisionX, use its byte count to create number.pdf, then write the previous question. Expected state has 79.pdf; the model creates and writes 79.00.pdf. | 8; multi_turn:instance_state_mismatch. |
| BFCL/simple_java_35 | AND two Redis keys into a specified destination with Redisson. The operation is AND, but array arguments are emitted as additionally quoted string representations. | 1; value_error:others. A narrower Java-checker cause is not saved. |
| BFCL/live_irrelevance_684-222-0 | A movie request is presented with only music/weather tools. The model calls music lookup despite the absence of a suitable movie tool. | 1; irrelevance_error:decoder_success. |
| BFCL/live_multiple_423-141-12 | Obtain two tickets for a named movie at 20:30 in San Francisco. The official expected call permits an unspecified date; the model asks for the date and emits no call. | 1; ast_decoder:decoder_failed. This is a mismatch with the benchmark's optional-argument convention. |
| BFCL/multi_turn_miss_param_31 | Create Reports, move a document, read/search another file and count lines; request a character-count mean, then clarify Reports only. The model omits cat, computes a mean before the file set is specified, and after clarification states 37 without the expected mean([37]) call. | 10; multi_turn:execution_response_mismatch. Saved detail does not identify which deviation first caused failure. |

Terminal-Bench ended without saved exception information:

- portfolio-optimization: official pass, **6 model steps**, 6,464 output
  tokens; trial elapsed 497.886 s including environment/verifier work.
- install-windows-3.11: official zero, **20 model steps**, 17,748 output
  tokens; trial elapsed 458.129 s. No custom rerun or inferred score.

### Constructive-answer scoring caveat

The fixed LiveCodeBench grade_stdio compares normalized output lines or
Decimal columns. It does not run a problem-specific validity checker for
arc191_c. Public test cases are included before private cases. For public
N=1 the final answer would print "2 1", mathematically valid here, while the
stored expected output is "20250126 1". That difference alone can fail this
evaluator.

The official zero is retained. The saved result only says passed:false, so
the actual first failure and correctness on every input remain unknown.
This item must not be treated as an unqualified capability failure or a
reason to imitate a particular valid output. The grading contract is
separate from weight editing.

## Capture coverage and provenance

Capture ID: 5ce20811d57047c5b579c4d1c1a9dab4.

The stopped capture retains 39 files and 809,409,554 bytes. The drained
snapshot saves every file's size, timestamp and SHA-256. All eight queues
have depth/pending bytes zero, no drops, no errors and no exhausted quota.
**Only four of eight worker status files acknowledge closed:true.** Preserve
that partial shutdown acknowledgement; do not relabel it as complete.
Every compressed observation stream was subsequently read to valid gzip EOF.

All **778 items** link to observations across 1,615 engine requests,
including 26 requests within the two agent trials. This establishes item
coverage, not exhaustive token coverage. The analysis excludes 42 unlinked
token rows per TP rank from the pre-benchmark probe instead of assigning
them to benchmark items.

Sampling uses the first execution of the first 32 prefill positions and
these prompt-end-relative decode offsets:

    0,1,2,4,8,16,32,64,96,128,192,256,384,512,768,1024,
    1536,2048,3072,4096,6144,8192,12288,16384,24576,32768,
    49152,65536,98304,131072,196608,262143

Unreached positions are missing, not zero. The semantic boundary between
thinking and final code is not measured. Primary TP0 has 47,663 accepted
prefill positions, 6,754 accepted decode positions, and 1,847 initially
rejected speculative decode positions. The rejected positions are 21.5%
of 8,601 sampled decode positions; main contrasts exclude them.
First-execution sampling does not retry a position merely because its
first speculative value was rejected.

All ordinary code/tools requests have usage metadata for relative offsets.
The 26 agent requests have trial linkage but no per-call usage here, so
their relative-offset denominators remain unknown. The largest accepted
relative offset among requests with known usage is 16,384; coverage of
later requested offsets is not claimed.

For all 6,760 matched code/tools token rows, TP0 and TP1 have numerically equal
values for all seven metrics, no unmatched rows and no nonfinite-pattern
mismatch. This numerical equality check does not assert identity of signed-zero
or NaN bit representations. Main analysis uses **TP0 only** for full-width residual/attention/
MLP sites and never sums TP duplicates. No individual channels or heads
were selected. These data can prioritize layer/role inspection, but cannot
provide an activation-based channel/head ranking.

## Exploratory layer/role contrasts

The reproduction script streams each worker through the existing
observe.core.json_lines API. It joins acceptance sidecars, uses accepted
values, averages positions within each request/phase band, then averages
requests equally per problem. Controls are the three nearest successful
tasks within the same source/BFCL group by median log2 prompt/output length.
Prefill, early decode 0–16, middle 32–256 and later 384+ remain separate.

Examples with a consistent sign across available phase bands:

| Failure | Site to inspect | RMS / matched-success median |
|---|---|---:|
| MBPP/410 | layer 50 attention | 1.323× |
| MBPP/576 | layer 51 attention | 0.877× |
| BFCL irrelevant movie/music request | layer 6 MLP | 0.737× |
| BFCL filename state task | layer 16 MLP | 0.905× |

These are **single-failure observational associations**, not evidence that
large activations are better, causal ability localization, or instructions
to increase/decrease weights. Task semantics, input/output length, prefix
caching, accepted-position availability and phase composition remain
confounders. Output length itself is an outcome. The Java task lacks a
successful Java-specific control; its broader non-live comparison is weak.
The constructive LCB task is unsuitable as a capability-failure contrast.

Next inspection can link selected layer/role sites to their explicit
checkpoint tensor mapping, combine weight structure with independent
controlled finite interventions, and judge candidates by measured
tradeoffs. Targeted channel/head capture is needed for finer activation
claims. Known benchmark answers or hidden thinking must not become an edit
objective.

## Artifacts and reproduction

On the SSH host, ROOT is
/mnt/nvme/kairyu/wsp/gemma4-coding-20261104.

- Results: ROOT/results/gemma4-base-observed-dev10-20261106/.
- Capture: ROOT/observations/base-dev10-20261106/.
- Analysis: ROOT/analysis/base-observed-dev10/.
- capture-drained-snapshot.json SHA-256:
  129a2ddd20c73c73f250bee3ea457f4bc4404862e027ba5cb9509d7622f30545.
- activation-analysis.json: coverage, acceptance, TP audit, controls/contrasts.
- observed-request-coverage.json: metadata linkage, no prompts or answers.
- code-tools-task-phase-metrics.npz: bounded task/phase/site metric arrays.

[Reproduction code](2026-11-06-observed-gemma-analysis.py) requires a new or
empty output directory outside capture/results. It does not execute models,
regenerate responses, regrade tasks or modify the capture.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
      python docs/experiments/2026-11-06-observed-gemma-analysis.py \
      --capture "$ROOT/observations/base-dev10-20261106" \
      --results "$ROOT/results/gemma4-base-observed-dev10-20261106" \
      --snapshot "$ROOT/analysis/base-observed-dev10/capture-drained-snapshot.json" \
      --output-dir "$ROOT/analysis/base-observed-dev10/reproduction-new"

BFCL provenance is the installed
[bfcl-eval 2026.3.23](https://pypi.org/project/bfcl-eval/2026.3.23/).
Its package metadata points to the
[official BFCL repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).
Task observations came from the pinned installed package's public
data/BFCL_v4_*.json and data/possible_answer files, not the moving branch.


Code-task sources, read from the pinned cached versions:

- [MBPP+ dataset revision b2d74c91837c3f2a20c1299ae98133cbe7cfa077](https://huggingface.co/datasets/evalplus/mbppplus/tree/b2d74c91837c3f2a20c1299ae98133cbe7cfa077):
  MbppPlus-v0.2.0.jsonl, task IDs Mbpp/410 and Mbpp/576, using
  canonical_solution/base_input/plus_input.
- [LiveCodeBench dataset revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe84c3912ea0c4d4a78037083943e8f0c4dd505):
  test6.jsonl, question_id arc191_c. This file name is part of the source
  dataset release, **not an access to the WSP held-out test split**.
- [Public ARC191 C problem](https://atcoder.jp/contests/arc191/tasks/arc191_c).
- Fixed LiveCodeBench evaluator
  [output comparison, lines 370–422](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/testing_util.py#L370-L422)
  and [public/private input order, lines 106–120](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/benchmarks/code_generation.py#L106-L120).
  The WSP scorer retains the official pass boolean, not a failed-case trace.

Validation: the checked-in reproduction script was run on CPU against the
stopped capture, in a separate output directory. Coverage, row counts, TP
audit, controls, contrasts, request linkage and all 232×180×7 task/phase
metric values exactly matched the first analysis. Saved nonfinite counts
were zero in those code/tools summaries. Ruff and diff whitespace checks
passed. This adds no model calls, verifier reruns, or grading changes.


## Follow-up: 0397 hard-code dev30 subset

Run gemma4-resume-h0397-dev30-hard-code-20261106 evaluated the official
tier30/dev subset selected by **only livecodebench,scicode**: 20 LCB and
10 SciCode problems, all scored, no errors or truncation. This is a
30-item diagnostic subset, not a full 14-area tier30 result.

The standard partial comparison pairs exactly those 30 IDs with
wsp-base-30-dev-1-20261104, retaining the same manifest/tier/split.
Overall score is unavailable. Among these two sources, the code aggregate
rose from 52.6508% to 57.6508%; this is not a full-code or all-area score.

| Measure | Old official base dev30 | Candidate 0397 |
|---|---:|---:|
| LCB passed | 15/20 | 16/20 |
| SciCode mean per-problem step fraction | 30.3016% | 35.3016% |
| arc195_b | fail | pass |
| SciCode14 | 0/2 steps | 1/2 steps, problem not fully solved |
| Other 28 item scores | unchanged | unchanged |

Elapsed time was 377.0 s, including loading 7.4 s. Generation finished at
236.8 s; subsequent scoring took about 140.2 s. There were 59 model
requests: 20 single-answer LCB requests plus 39 SciCode stage requests.
All saved finish reasons were stop.

- [ARC195 B, Uniform Sum](https://atcoder.jp/contests/arc195/tasks/arc195_b):
  fill unknown nonnegative values and permute A so all corresponding sums
  agree, with N up to 2,000. The old base's public final code overcounted
  duplicate-value matches as countA × countB. The candidate instead uses
  min(countA, countB), respecting that each element can be matched once.
  It passed after one generation, 10,524 output tokens and 95.46 s API time.
  Both historical base dev30 and dev60 failed this item. The code change
  and official pass are observed; repeatability is not yet established.
- SciCode14, Brownian_motion_in_the_optical_tweezer: two stages implement
  Langevin dynamics using Mannella leapfrog integration and mean-squared
  displacement across trials. Candidate used two generations, 9,660
  output tokens and 92.72 s summed API time. Its 1/2 step result is within
  historical unedited-base variation: old dev60 already achieved 2/2.
  Canonical detail preserves the passed-step count, not which step passed.

The independent finite calibration predicted mean margin deltas of
+0.01252 for algorithm and +0.01482 for implementation; measured finite
deltas were +0.02121 and +0.02125 respectively. These are correct-versus-foil
log-probability margins, **not accuracy forecasts**. The WSP result supports
one observed algorithm-task gain; the easier screen still had 17/20 code
passes, so broad coding improvement is not established.

For a low-cost official repeat, tier30/dev with only livecodebench retains
the original manifest and repeats the gained problem together with 19
controls. It omits the 39 SciCode calls and their grading cost. The current
CLI has source filtering but no individual-item filter; a one-item custom
manifest would be a separate diagnostic with a different manifest identity.
No repeat was launched by this analysis. The planned next check, owned by the
benchmark launcher after full dev10, is this same-checkpoint 20-item repeat
followed by a fresh official-base 20-item run, both without capture, to avoid
attributing observation-condition differences to the weight edit.

Evidence is under ROOT/analysis/resume-h0397-hard30/:
official-base30-paired.json, task-deltas-public-metadata.json, and
interpretation.json. Only public final code and metadata were inspected;
no hidden reasoning, additional model calls, or verifier reruns were used.


## Follow-up: 0397 full dev10 and LCB repeat

Run gemma4-resume-h0397-dev10-full-20261106 completed all 778 dev10 items
with no unscored items or generation errors. Wall time was 636.3 s,
including 104.8 s loading. Compared with the fresh observed base, overall
was 80.2148% → 81.9196% (+1.7048 pp, paired 95% CI −0.1583 to +3.8889 pp).
Code remained 87.7778%, BFCL remained 35/40, and agents remained 50%, with
the same per-item verdicts in those three areas. IFBench rose 12/15 → 13/15
(item 259 gained), while IFEval fell 39/40 → 38/40 (item 218 lost).

Reasoning rose 5 pp: MuSR murder_mysteries205 and 243 were gained, while
object_placements187 was lost. Other changes included expert +6.25 pp,
vision +12.5 pp, facts −2.5 pp, knowledge −1.6667 pp, commonsense −0.3333 pp
and multilingual −0.2778 pp. Relative to the old original dev10 base,
overall was only +0.6959 pp (95% CI −1.3959 to +2.7931 pp), code +3.3333 pp
from MBPP801, instructions −4.5833 pp and reasoning −2.1739 pp. MBPP801
also passed in the fresh unedited base, so that historical difference is
not evidence that the edit caused the gain. These baseline-sensitive
results do not establish a broad coding improvement.

Run gemma4-resume-h0397-dev30-lcb-repeat-20261106 then repeated the same
20 LCB problems on the same checkpoint without observation. It completed
20/20 with no unscored items in 145.866 s. Accuracy changed 16/20 → 15/20:
arc195_b stayed correct, but abc390_g changed from pass to fail; the other
18 verdicts stayed the same. Thus the new algorithm-task gain repeated,
while the total gain over the historical 15/20 base did not. A fresh
unedited-base run under the same normal serving conditions is the next
comparison, owned by the benchmark launcher.

Evidence: ROOT/analysis/resume-h0397-full10/ contains the official observed
and original base comparisons and item deltas;
ROOT/analysis/resume-h0397-lcb-repeat/first-versus-repeat.json contains the
official 20-item paired partial comparison. Overall remains unavailable
for the LCB-only subset. No hidden reasoning was inspected.

The reusable [experiment evidence recipes](scripts/README.md) preserve the
official comparisons while exposing repeat ranges, fractional item
changes and recorded steps/tokens, including UUID-deduplicated DeepSWE
retry usage. They make no new model or verifier calls.


The fresh normal-serving base check,
gemma4-resume-base-dev30-lcb-matched-20261106, completed 20/20 with no
unscored items in 153.956 s. Its LCB score was 14/20, versus 16/20 and
15/20 in the two 0397 observations. Only two IDs varied across the four
observations:

| Item | Old base dev30 | Fresh normal base | 0397 first | 0397 repeat |
|---|---:|---:|---:|---:|
| arc195_b | 0 | 0 | 1 | 1 |
| abc390_g | 1 | 0 | 1 | 0 |
| Other 18 items | unchanged | unchanged | unchanged | unchanged |

The official common-20 comparisons give +10 pp (95% CI 0 to +25 pp) for
the first candidate observation and +5 pp (0 to +15 pp) for its repeat
against the fresh base. The arc195_b gain is supported by both candidate
observations and both base observations; abc390_g varies on both sides.
These are small descriptive repeat counts, not proof of a stable effect
size across unseen problems. Normal serving was used for these three new
LCB evaluations, without activation capture.

The fresh base used 20 calls, 12,686 input and 128,865 output tokens; the
candidate repeat used 20 calls, 12,686 input and 132,468 output tokens.
Summed API durations were 1,065.389 and 1,103.934 s respectively, which
overlap under parallel execution and must not be added to the wall times.
Evidence in the same repeat directory:
fresh-base-versus-first0397.json, fresh-base-versus-repeat0397.json,
and four-observation-scores.json.


## Follow-up: 0485 95-item screen

gemma4-resume-h0485-dev10-screen-20261106 completed all 95 items in
67.853 s with no unscored items. This is the official dev10 source subset
of 20 code, 40 BFCL, 15 IFBench and 20 log-probability probes; it is not a
14-area baseline or a full dev10 evaluation.

| Measure | Fresh observed base, common items | 0397 screen | 0485 screen |
|---|---:|---:|---:|
| HumanEval+ | 10/10 | 10/10 | 10/10 |
| MBPP+ | 8/10 | 7/10 | 7/10 |
| BFCL | 35/40 | 35/40 | 34/40 |
| IFBench | 12/15 | 12/15 | 13/15 |

Against the fresh observed base, MBPP801 changed to fail on both base and
plus tests. BFCL live_multiple_924-191-12 changed to fail with the official
error simple_function_checker:missing_optional; this does not establish
that the model chose the wrong function. IFBench93 and 259 became correct
under both strict/loose checks, while IFBench182 failed both. The latter
used one request, 3,346 output tokens and 33.82 s API time. MBPP801 used
one request, 1,104 output tokens and 15.69 s; the BFCL item used one request,
423 output tokens and 10.68 s. All finished normally, without API errors.

Relative to the historical base95, code and BFCL totals were unchanged:
BFCL multi_turn_miss_func_111 was gained while live_multiple_924-191-12
was lost; IFBench259 was the only instruction-score gain. The screen
therefore does not yet support code/tool improvement. There were 168
recorded chat calls and 87,227 output tokens across the 75 chat items;
20 probes are separate. Official comparisons and changed-item lists are
under ROOT/analysis/resume-h0485-screen/.


0485 also completed the official 30-item hard-code subset in
gemma4-resume-h0485-dev30-hard-code-20261106: 308.773 s, 30/30 scored,
no unscored items. LCB was 16/20; SciCode's mean per-problem step fraction
was 40.3016%. Relative to old base dev30, only arc195_b (0→1) and SciCode14
(0/2→2/2) changed. Relative to the first 0397 hard30 observation, all LCB
verdicts matched and only SciCode14 rose from 1/2 to 2/2.

Against the fresh normal base's 14/20 LCB result, the gains were arc195_b
and abc390_g, with no losses. The former is now correct in both 0397
observations and this 0485 observation; abc390_g already varied on both
base/candidate sides. SciCode14's full solution also occurred in old
unedited dev60, so it is not a uniquely new ability. These hard-code
results are encouraging alongside the screen's code/tool losses, but do
not by themselves select 0485 or establish a stable overall gain.

All 59 recorded calls finished with stop, totaling 87,821 input and 273,928
output tokens. arc195_b used one call/9,253 output tokens/84.53 s API time;
abc390_g one call/11,317 tokens/99.27 s; SciCode14 two stage calls/8,817
tokens/88.61 s summed API time. Official paired comparisons and changed
IDs are under ROOT/analysis/resume-h0485-hard30/. No additional generations
or verifier reruns were made by this analysis.

## Follow-up: 0485 full dev10

gemma4-resume-h0485-dev10-full-20261106 completed all 778 items with
zero unscored items in 620.229 s (load 104.9 s is included). Official
runtime counters recorded 555 chat calls and 3,062 log-probability calls.
The chat-call records contain 1,041,340 output tokens; Terminal-Bench
agent usage is recorded separately below.

| Comparison base | Overall base → 0485 | Delta pp | Paired 95% CI pp |
|---|---:|---:|---:|
| Fresh observed base | 80.2148% → 82.1640% | +1.9492 | -0.2719 to +4.4581 |
| 0397 full dev10 | 81.9196% → 82.1640% | +0.2443 | -2.1027 to +2.6127 |
| Old original base dev10 | 81.2237% → 82.1640% | +0.9402 | -1.3602 to +3.3699 |

| Area | Fresh observed base | 0397 | 0485 | 0485 − fresh pp |
|---|---:|---:|---:|---:|
| knowledge | 90.0000% | 88.3333% | 90.0000% | +0.0000 |
| expert | 81.2500% | 87.5000% | 81.2500% | +0.0000 |
| facts | 70.0000% | 67.5000% | 67.5000% | -2.5000 |
| math | 98.7500% | 98.7500% | 95.9375% | -2.8125 |
| code | 87.7778% | 87.7778% | 84.4444% | -3.3333 |
| instructions | 88.7500% | 90.8333% | 92.0833% | +3.3333 |
| tools | 87.5000% | 87.5000% | 87.5000% | +0.0000 |
| agents | 50.0000% | 50.0000% | 50.0000% | +0.0000 |
| long-context | 81.3436% | 81.3779% | 93.7500% | +12.4064 |
| reasoning | 71.9565% | 76.9565% | 71.9565% | +0.0000 |
| commonsense | 67.3333% | 67.0000% | 66.0000% | -1.3333 |
| multilingual | 80.9848% | 80.7071% | 83.7626% | +2.7778 |
| vision | 81.2500% | 93.7500% | 100.0000% | +18.7500 |
| safety | 86.1111% | 88.8889% | 86.1111% | +0.0000 |

The overall increase is mainly long-context and vision. Coding fell by
3.3333 pp relative to both the fresh base and 0397; tools and agents had
identical verdicts. Compared with 0397, reasoning also fell 5 pp, expert
6.25 pp and math 2.8125 pp. Relative to the old original base, code was
unchanged, instructions fell 3.3333 pp and reasoning fell 7.1739 pp.
The intervals include zero, and this is one full observation per candidate.

Concrete changed tasks and recorded work:

- MBPP801: count equal values among three integers. The final code returns
  1 when all differ, while the official reference returns 0; the public
  input [-1, -2, -3] exposes that discrepancy. Both base/plus suites fail.
  One call, 1,290 output tokens, 195.30 s API time. Fresh base and 0397 full
  passed, but old original base failed, so the historical variability remains
  relevant. No new execution was needed to identify this final-code mismatch.
- IFBench93: explain Atlantic Charter principles with every word on a
  separate line. Both strict/loose checks pass: one call, 1,479 output
  tokens, 47.47 s. This is the instruction gain over the fresh base.
- MRCR-v2:2069: the long-dialogue reference/reproduction score rose from
  0.00749 (fresh base) or 0.01024 (0397/old base) to 1. One call, 11,210
  output tokens, 275.47 s. This is an observed item gain, not proof that a
  particular edited coordinate caused better long-context handling.
- BBEH disambiguation_qa119 and MuSR murder_mysteries205 were gained relative
  to the fresh base, while BBEH nycc018 and MuSR object_placements187 were
  lost. Their respective call/output-token counts were 1/4,784, 1/2,162,
  1/5,796 and 1/6,944. Relative to 0397, MuSR murder_mysteries243 was also
  lost (one call, 4,530 tokens). Specific reasoning errors were not inferred
  from uninspected hidden reasoning.
- RULER fwe:131072:005 reached 16,384 output tokens and finish_reason=length;
  its official detail marks truncated=true and score 0. This was the only
  truncated chat call, not an unscored infrastructure error (514.80 s API
  time). It remained a failure in the compared runs.
- Terminal-Bench portfolio-optimization passed after 5 agent steps,
  31,643 input/6,344 output tokens. install-windows-3.11 failed after 12
  steps, 80,862 input/9,729 output tokens. Both had official verifier
  rewards and no exception. These agent steps are distinct from HTTP
  request counts; their usage was not invented from the WSP item summary.

Some same-checkpoint screen/full outcomes differ: the screen's BFCL
missing_optional failure recovered in full; IFBench259 went from pass to
fail, while IFBench182 recovered. Full BFCL was 35/40 and IFBench13/15.
Equal aggregate counts therefore do not imply equal item outcomes.

The three official comparisons, per-item usage/deltas and metadata-only
progress snapshots are under ROOT/analysis/resume-h0485-full10/. They were
generated with the checked-in WSP evidence recipe. Per-item usage for
log-probability tasks is unknown here and is null, not zero. API durations
include concurrent waits and must not be summed into elapsed wall time.
No additional generations, regrading, or held-out performance inspection
were performed by this analysis.
