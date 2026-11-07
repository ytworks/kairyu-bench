# Coordinate sampling and coherent full MLP units

AV is the accepted cumulative parent. This comparison tests whether observing
and editing complete coupled MLP rows/columns provides a useful direction beyond
the existing 512-cell samples. It does not infer functional circuits from weight
geometry or assume that selected likelihood gains add across successive rounds.

The actual Opus 5.5 review recommended the coherent-unit mechanism alongside a
coordinate control; the larger low-rank basis proposal remains deferred. The
review used no tools. Its receipt is `6dc244fa…`, and the recorded disposition
is `6bf62d77…`. This was a method review; the actual two-candidate review will
follow both CPU proposals, before finite or ordinary generation advancement.

## Shared objective and control

AZ and BA use the same 120 authored cases, 19 purposes and 38 folds, explicit
SDPA text dispatch, serial complete answers, and the accepted AV checkpoint.
Primary purposes are implementation, algorithm, workflow, ten science/stage
purposes, and tool-call, tool-no-call, tool-arguments and referent tasks. Plain
code and fact retention remain retention constraints.

The fixed targets are correct-answer LL gain `min(0.05 * headroom, 0.05)` and
margin gain `0.02`; other lower bounds are `-0.01`. Both use Euclidean minimum
norm, realized BF16 raw-L2 cap 5, one analytic box calculation and one bounded
quantized proposal. The original small coordinate box is only a reference for
deriving the actual box. There is no amplitude sweep or automatic retry.

Native parameters and explicit V tensors are frozen. In tied K/V layers, editing
K can still alter functional value computation. The comparison keeps attention
sample coordinates fixed, so removing Q/K is not a confounding change.

AZ's CPU proposal completed in 9.387 seconds: calibration 2.303 seconds and
candidate stage 6.306 seconds. It changes 172,056 BF16 cells in 360 matrices, with
raw and declared L2 both 1.585487. Two of 76 strict predicted bounds miss by at
most `6.133e-8`; these are reported, not converted into an accuracy claim.
Completion `57b5713e…`, numeric postcheck `af101ebf…`, and spec `73f58048…`
remain immutable. No finite or generated outcome for AZ is recorded here.

The inverse-absolute-weight metric with floor `1e-6`, which is not the optimized
metric here, gives AZ norm 15,309.34. The analytic containing box has relative
radius 8,827.96 and absolute term 0.008828. A near-zero source weight can dominate
that ratio; it does not mean all weights change by this factor. One rounded cell
exceeds the continuous box by `2.248e-5`. The box is not a small-per-weight relative
guarantee. The norm claim concerns the declared linear surrogate, and does not
establish globally minimal model changes or an optimal discrete BF16 edit.

## Observed support selection

The separate UP-output gain collector observes all 1,290,240 hidden-unit gain
sensitivities in 60 matrices with frozen model parameters. Its two objectives
completed all 38 folds and 120 cases. Collection times were 65.419 and 126.954
seconds, and peak allocated memory was 84.690 GiB under the declared 88 GiB cap.
The original service was restored with the same identity. Completion
`bc75e6b7…` and output inventory `b07f4cfe…` preserve the observations.

The prospective selector divides each gain derivative by its source UP-row L2
cost, normalizes each objective/fold vector, then computes group RMS salience.
The fixed quotas are 12 science/helper, 12 code/workflow and 8 tool/reference
units, with global deduplication and at most two units per layer. Plain-code and
fact sensitivity in support selection does not change their retention targets.

The single CPU selection took 13.890 seconds and chose exactly 32 units across
24 layers: 16 layers contain one unit and eight contain two. All quotas were
filled. UP-versus-coordinate baseline LL was identical; maximum margin error
was `8.941e-7`. Selected source row norms were checked against checkpoint bytes.
Support `48c32070…` and selection report `b98e50ef…` include the selected weight
geometry; this is salience evidence, not causal attribution.

BA's policy `123d0930…` was frozen before its fresh coordinate derivatives.
Its 32 full gate/up rows and down columns contribute 516,096 eligible cells;
unchanged sampled Q/K/O add 92,160, for 608,256 eligible cells. V samples remain
diagnostic. Old coordinate gradients are not reused for BA. At this point BA
has no fitted proposal or measured finite effect.

## Validation and limits

Existing CPU fixtures cover complete-row/column derivatives against ordinary
autograd, selector-to-derived-bound-to-quantized-save behavior, unchanged native,
V and unselected bytes, and the explicit bounded editor/finite-load path. The
combined CPU suite passed 520 tests: 464 passed and 56 optional tests skipped;
source snapshots were unchanged. No new test methods were added for this mode.

The native-thinking authored baseline passed 6/6 helper stages and 6/6
oracle-helper downstream stages, but 5/6 generated-helper downstream stages.
The remaining train-chain outcome was a runtime failure; public answer bodies
were not inspected, so faulty-helper propagation is not established. Both
disjoint transfer chains passed. These mostly solved tasks are dependency and
retention anchors, not demonstrated unsolved capability deficits.

Teacher-forced empty-thought likelihood is still a different condition from
native-thinking generation. More observed coordinates, longer contexts, or
greater salience do not establish broader code improvement. D's memory fit,
nonlinear response and generated outcomes remain to be measured independently.

## First D observation and memory revision

The first full-unit observation failed under the unchanged 88 GiB allocator
limit. Margin failed in the first fold during autograd; LL had completed 34/38
folds when the owner cancelled its peer. Both partial outputs are ineligible for
BA preparation. The original service was restored at 02:42:25 UTC; failure
completion `6b0aed7d…` and analysis `9b14431d…` are retained. No solver ran.

The allocator reported a 170 MiB request, with 84.97 GiB allocated and 2.94 GiB
reserved but unallocated. The existing full-unit contraction already used
source-dtype captures and small 64-token FP32 tiles. The first fold includes a
169-token correct answer; a full FP32 vocabulary-logit tensor for that answer is
169 MiB. This matches the allocation scale but does not identify its exact
origin from the available traceback.

The authorized narrow revision offloads complete gate/up input captures to CPU
in their original dtype, restores each capture once for its backward hook, and
releases it immediately. Prior-answer temporary derivative arrays are also
released before the next forward. Products, accumulation order, frames, cases,
support, targets and the 88 GiB policy remain unchanged. The potential unique
full-axis GPU capture storage is about 312 MiB at the longest frame; actual peak
reduction is unmeasured. A new versioned complete observation is required; the
partial first attempt will not be resumed or mixed into it.

## Conditional objective follow-up

If the fixed A/D comparison does not improve functional outcomes, the preferred
next mechanism is the reviewed equivalent-reference objective, focused on
execution and public-helper-context robustness. Two or three CPU-verified
equivalent complete implementations per stage would share total stage weight
one; solved canonical cases would remain retention anchors, with semantic
wrong-program foils. Public helper implementations can vary meaningfully while
preserving their declared interface. Any later model-generated helper context
would be extracted and evaluated automatically as public code, never hidden
reasoning. No such expansion or additional collection has been executed.

The existing 6/6 helper, 6/6 oracle-downstream and 5/6 generated-context results
motivate testing context-conditioned execution, but do not diagnose a faulty
helper. The existing three-mode executor can distinguish successful canonical
execution from robust composition across equivalent contexts. A change in
reference LL without a corresponding execution gain would remain evidence of
proxy misalignment. Equivalent references alone cannot align an empty-thought
objective with native-thinking generation. This conditional choice leaves A/D
support and policy fixed; actual functional and official scores remain primary.
The follow-up must retain implementation and algorithm improvement alongside
science composition, rather than narrowing to science-reference fitting alone.
