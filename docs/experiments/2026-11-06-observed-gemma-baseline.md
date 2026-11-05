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
