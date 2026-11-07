# Fresh purpose-led weight editing, 2026-11-07

This session restarts the scientific decision process from the coding objective.
Historical hypotheses, candidate ranks, adoption policies and proposed queues
are not inherited. Existing tools and measured data may be reused only with
identified scope and checkpoint/condition evidence. The independent audit is
[recorded separately](2026-11-07-fresh-evidence-audit.md).

## Objective and evaluation contract

Improve coding above 90 using the official source mean. The first feedback
instrument is canonical WSP tier10 dev, all 778 items and all 14 domains,
without exclusions. Tier30/60 retain their entire canonical selections and
are separate score families; tier10 has no SciCode. No 90-item code-only
screen substitutes for canonical WSP. Test is reserved for final confirmation.
Use the highest completed same-condition model as the next editing parent.
Never advance a candidate based on a partial result or surrogate prediction.

The independently verified AV checkpoint is tied for the highest completed
tier10 code score (87.777778) and preserves cumulative earlier edits. In the
matched AD/AV protocol each run used one TP2 replica, MTP4, chat32/logprob8,
agent1, the same runner and all 778 items. AV has HumanEval+ 10/10,
LiveCodeBench 5/6, MBPP+ 8/10. All code outcomes match AD. AV's overall
score is 0.054368 points lower; that small difference is not proof of a causal
regression. AV is the initial cumulative parent, not an assumed better model.

## Initial environment and operational findings

All implementation, computation and benchmarks run on the specified remote
8-GPU server, as the non-root user. NVMe initially has 137 GiB free. Four
healthy TP2 Gemma replicas occupy all eight GPUs. Their checkpoint mounts
are AV, AD, original, AR, respectively. Thus the shared chat gateway cannot
be used as a single-weight evaluation endpoint. Chat and logprob must reach
the same explicitly identified replica. No preexisting benchmark lock or
active generation was found before taking ownership of a measurement.

The supplied CPU environment imports NumPy and the repository's own
safetensors reader; the optional third-party safetensors package is absent.
It is not required by these static tools. `./kairyu-bench wsp run --help`
was discovered to build a Docker image and install runtime dependencies.
The two help-only processes created by this session were identified by their
exact `wsp run --help` command and stopped; no model evaluation was made.
Future command discovery uses source or the host parser. This setup mistake
is retained and does not become a benchmark trial.

## New hypotheses and measurements

1. Observe AV's actual canonical benchmark activations, including every layer's
   attention, MLP and residual at explicitly sampled token positions. Preserve
   request/call/rank joins, quotas, missing data and writer shutdown state.
   Measure a small deterministic observation ON/OFF overhead diagnostic.
2. Reuse raw AV derivatives only after verifying the full parent identity,
   authored cases, objective frame, serial layout and stored coordinates.
   Investigate shared and conflicting directions between implementation,
   algorithms, language primitives, workflows and other authored purposes.
   Gradient geometry is an association, not a causal capability map.
3. Propose a simultaneous edit across the existing layer-stratified matrix
   sample. For F01, require each implementation and algorithm fold's public
   answer LL and correct-minus-foil margin to increase locally by 0.03 natural
   log units per answer token; code primitives LL +0.01 and margin >=0;
   permit every remaining authored fold up to 0.03 loss in each objective.
   All folds remain measured. These new numerical targets are experimental
   choices, not inherited thresholds or predicted WSP points.
4. Search all observed Q/K/V/O and gate/up/down sample coordinates jointly,
   freeze native scalar/vector weights, minimize Euclidean delta norm, and
   round once from AV to stored BF16. The continuous per-coordinate bound is
   max(0.25*abs(parent), 0.0001), with at most 16 quantization-correction rounds.
   Report actual BF16 differences and shortfalls; no claim of global optimum.
5. Review a concrete candidate with actual Claude Opus 5.5, retaining the
   hash-pinned anonymous prompt, response-model receipt and disposition.
   Save/verify candidate weights before canonical WSP comparison. Retain all
   failures, losses, unscored results and repeated runs.

A broader code benefit requires complete tier30/60 evidence, not interpolation
from tier10. The final edited checkpoint must run DeepSWE all 113 tasks once at
32 workers; append results to the wiki. Publish the edited checkpoint to
Hugging Face only if the matched original comparison improves. Completion,
score, numerator/denominator, per-item changes, agent steps, usage and elapsed
time must remain distinct from setup and surrogate measurements.

## Execution record

Append measured outcomes below; the preceding plan does not assert success.

### F01 first review and bounded next measurement

The actual second opinion returned `claude-opus-5-5` in assistant, usage and
canonical model identities, verified by the receipt. Dispatch took 90.24 seconds
and cost $0.245808. The [anonymous prompt](anonymous-fresh-code-f01-review-2026-11-07.txt)
and [verified receipt](evidence-2026-11-07-fresh-audit/f01-opus-first-review.json)
are retained. The review correctly emphasizes that neither the selected dev
maximum nor authored surrogate gains establish capability improvement.

F01 changes 201,426 stored BF16 cells in 410 matrices, delta L2 0.991527104249,
relative to the **sampled** source norm 7.388781142491 (ratio 0.134193595009).
The maximum cell change is 0.056640625. Native tensors remain frozen. Exact
source-word and target-BF16 checks passed. Nine of 76 linear targets are unmet;
all non-target -0.03 allowances remain satisfied in the prediction. These
numerical targets were declared experimental design targets, not automatic
adoption criteria; the reviewer interpreted them more strongly. No target is
relaxed after seeing the proposal and no adoption is authorized by this result.

We accept the recommendation to treat F01 first as a diagnostic finite-response
probe and not perform an amplitude/round search. Create and fully verify its
saved checkpoint, then measure all 120 authored cases AV -> fixed F01 -> AV.
Stop this direction if target folds reverse sign or non-target finite LL/margin
losses exceed 0.03 after checking A/A drift. If it transfers, inspect generation
and a complete matched WSP run, retaining every problem and all 14 domains.
The finite result cannot become a WSP score. The observer and ordinary serving
conditions must be compared explicitly; short thinking-OFF checks establish
only their own limited result. No additional optimizer is introduced.

A rejection rule for this direction is recorded before finite/official results:
no official code gain means no advancement. Any candidate that gains code but
loses more than 1 point in the 14-area overall mean, or more than 5 points in an
individual area, requires a new explicit analysis of the measured tradeoff and
cannot be silently accepted. These are operating thresholds for this experiment,
not claims of statistical significance or universal quality standards. Claim
only the named tier and evaluated split, with counts and paired uncertainty.
The final test split has already been used historically; it cannot be represented
as virgin data merely because this session restarts the scientific decisions.

### Direct stored-weight observations

Fresh CPU inspection found all 1,331,516 selected native norm/scalar values finite
(421 tensors); their L2 is 70,134.9646235 and is not a whole-model denominator.
The sampled algorithm/implementation LL gradients put the largest layer energy
at layer 25, followed by 31 and 28. This selected layer 25 for diagnostics, not for
a causal coding-neuron claim or an automatic edit mask.

Layer25's selected attention/MLP stored tensors contain 478,937,600 elements, all
finite, with L2=351.276913965. Complete row/column statistics were recorded. A
128x128 sample of the O projection has numerical rank128, stable rank
24.104732, energy-entropy rank74.874692;
these ranks describe only that sampled submatrix. Q-row/O-column correspondence
on 128 sampled channels and 128 features has mean cosine 0.000921
and mean absolute cosine 0.078597. Geometry does not identify
a functional coding circuit.

All four static outputs were tied to the fresh AV full hash by matching the
complete file inventory/stat snapshot. Static commands used `--no-hash` and did
not claim another full payload hash. Numeric outputs (gzip) and their source
hashes are retained in `evidence-2026-11-07-fresh-audit/av-static-observations.json`.

F01 saved-checkpoint creation and independent full-file verification both passed.
Creation took approximately 220 seconds and verification 173 seconds
(from output-file timestamps, one-second precision). All nine retained files
were checked, including unchanged configuration/tokenizer/native and unselected
weight bytes. The saved edit has the same 201,426 cells and L2 as the proposal.
No serving adoption or capability result follows from this integrity check.

Header inventory gives 31,273,088,876 stored floating elements, including stored
structural aliases. The 201,426 changed elements are 0.0006440873% of that
inventory. This small count fraction does not negate the 13.4194% relative
change within the sampled source values or establish a small functional effect.

### F01 finite outcome and a new dependency objective

The fixed AV -> F01 -> restored AV probe completed all 120 authored cases in
all three sweeps (20.935 / 20.364 / 20.368 seconds). Total finite runtime was
77.729 seconds; ownership and exact service restoration took 197.772 seconds.
Selected BF16 bytes were exact and restored AV had zero LL/margin/KL drift.
All protected service identities were preserved and the exact GPU6/7 chat
container returned healthy. The paired code proxy gains are not WSP points.

All algorithm/implementation fold changes were positive, but two referent
fold1 retention measurements failed the predeclared -0.03 allowance: LL
-0.037517150243 and margin -0.032255729040. Their linear predictions had been
-0.001703215622 and -0.003620151374. F01 therefore stops before serving or
canonical WSP. All 76 rows, all 120 case margins, failed conditions and saved
weight integrity remain retained; no official F01 score is available.

F02 is a newly specified joint objective on the same highest comparable AV
parent, not an increment from unadopted F01. Reference tracking is plausibly
needed to carry variable and helper relationships through code, but the authored
referent task is not proof of that dependency or of an official benchmark cause.
Observed local derivatives show both shared and conflicting directions: referent
LL and implementation LL cosines are positive, while referent/code and cross-
objective relations can conflict. Both referent folds and both objectives must
remain separate; averaging them would hide that conflict. All families remain
eligible because a negative marginal contribution does not establish that an
entire projection family should be removed.

Before computing F02, fix its targets: referent, algorithm and implementation
LL/margin +0.03 for each fold; code LL +0.01 and margin >=0; every other fold
LL/margin >=-0.03. Keep all 76 constraints, all 410 matrices / 209,920 sampled
coordinates, the same Euclidean cost, continuous bounds and at most 16 BF16
correction rounds. Solve one fixed plan from AV. This is not a search over F01
amplitude or a favorable correction round. Preserve every unmet prediction.

First measure the fixed spec in RAM with AV -> F02 -> AV and exact selected-word
proof. This phase is explicitly not a saved-checkpoint validation. Require zero
A/A drift, positive code/reference target direction and the same non-target
finite -0.03 retention checks before moving to complete canonical WSP. The +0.03
design targets remain reported even when unmet. No fixed linear prediction
warrants assuming a positive finite result: F01's largest negative referent
prediction error itself exceeded 0.03. Save and fully verify a surviving F02,
obtain an actual Opus 5.5 opinion on its measured evidence, then measure all
canonical questions and domains with matched serving/evaluation conditions.

### Runner reproducibility and CPU validation

The host wrapper now supports an explicit full local Docker image ID for reuse.
Without that opt-in, it still builds the checkout. The added end-to-end wrapper
fixture checks exact reuse, no build/pull, preserved invocation/mounts and refusal
of missing/mismatched/mutable image references. This protects the concrete risk
of changing the paired WSP runner while adding experiment scripts. The benchmark
container, scorer and mounted cache environments are separately attested.

The first full CPU validation attempt failed on environment setup: an overly
long NVMe TMPDIR exceeded Unix socket path limits, and the CPU interpreter was
invoked by absolute path without adding its bin directory to PATH for shell
fixtures. All 19 error records are retained. A second setup attempt could not
create a short directory outside the owned root and ran no tests. The third uses
a short directory within the owned NVMe root and the existing CPU venv PATH;
full unittest, Ruff, mypy, data consistency and shell syntax checks passed.
Detailed counts and raw logs are retained under
`evidence-2026-11-07-fresh-audit/validation/`; these are CPU checks, not GPU proof.

### F02 failed proxy and amended official transfer diagnostic

F02 failed its recorded proxy gate with exact A/A restoration. After a second
actual Opus5.5 review, the [frozen post-data amendment](2026-11-07-f02-transfer-amendment.md)
allows one complete official diagnostic without re-solving or erasing the gate.
It fixes candidate identity, full evaluation conditions, official tradeoff and
noise/confirmation rules before materialization or official candidate results.

### Compose deployment failure and repair

F02's first service switch and automatic Compose rollback failed before any
candidate benchmark. A stopped retired AV observer retained the target's Compose
labels. The original AV container was recovered healthy with its exact CID; all
other production containers and both checkpoint inventories match their prior
states. Only the unstarted rollback object created by this attempt was deliberately
removed after its evidence was saved. The retired observer Docker object had
already been removed by Compose; host benchmark/capture evidence survives, while
its unexported stdout/stderr may be unavailable. See the [full incident and
recovery record](2026-11-07-f02-serving-preview.md).

The existing single-replica helper now validates all project/service members,
including stopped or renamed containers, before preview and immediately before
apply. It refuses ambiguous rollback and forces orphan removal off. The CLI E2E
covers early duplicates, a late duplicate, a normal replacement, ambiguous
rollback refusal and zero-member restoration. The [independent review](2026-11-07-compose-scope-review.md)
was incorporated. A full CPU suite and Ruff/mypy/data checks passed again after
this real failure; logs are retained under validation/r4*. This fixes deployment
scope, not the candidate or official scoring. The first failed apply/rollback and
manual recovery are retained; the official F02 trial count was still zero.
