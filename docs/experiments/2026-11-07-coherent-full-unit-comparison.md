# Coordinate sampling and coherent full MLP units

AV is the accepted cumulative parent. This comparison tests whether observing
and editing complete coupled MLP rows/columns provides a useful direction beyond
the existing 512-cell samples. It does not infer functional circuits from weight
geometry or assume that selected likelihood gains add across successive rounds.

The actual Opus 5.5 review recommended the coherent-unit mechanism alongside a
coordinate control; the larger low-rank basis proposal remains deferred. The
review used no tools. Its receipt is `6dc244fa…`, and the recorded disposition
is `6bf62d77…`. This was a method review. The actual two-candidate review followed both CPU
proposals, before finite and ordinary generation advancement, as recorded below.

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
remain immutable. The later finite outcomes are recorded separately below.

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
diagnostic. Old coordinate gradients are not reused for BA. This support and
policy were fixed before BA fitting or finite measurement.

## Validation and limits

Existing CPU fixtures cover complete-row/column derivatives against ordinary
autograd, selector-to-derived-bound-to-quantized-save behavior, unchanged native,
V and unselected bytes, and the explicit bounded editor/finite-load path. The
combined CPU suite collected 520 tests: 464 passed and 56 optional tests skipped;
source snapshots were unchanged. No new test methods were added for this mode.

The native-thinking authored baseline passed 6/6 helper stages and 6/6
oracle-helper downstream stages, but 5/6 generated-helper downstream stages.
The remaining train-chain outcome was a runtime failure; public answer bodies
were not inspected, so faulty-helper propagation is not established. Both
disjoint transfer chains passed. These mostly solved tasks are dependency and
retention anchors, not demonstrated unsolved capability deficits.

Teacher-forced empty-thought likelihood is still a different condition from
native-thinking generation. More observed coordinates, longer contexts, or
greater salience do not establish broader code improvement. Memory fit,
nonlinear response and generated outcomes require separate measurements.

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

## Completed D-v2 and BA proposal

D-v2 completed fresh LL and margin observations for all 38 folds/120 cases and
restored the same original service at 02:59:08 UTC. Reported collection times
were 42.627 and 80.453 seconds; peaks were 85.0447 and 85.0702 GiB allocated,
with 88 GiB reserved. These are not speed or peak-reduction estimates against
the incomplete first attempt. Completion `eecf0958…` and independent review
`e38cddba…` retain the evidence. The forward baselines are exactly equal to A;
paired sample/source bits, complete selected MLP axes and unchanged attention
coordinates/source bits agree. Native gradients remain unobserved.

The single BA CPU batch completed in 24.036 seconds, including calibration
5.705 seconds and candidate stage 16.413 seconds. Inputs remained stable; there
were no retries, model calls or checkpoint writes. Completion `b739192a…`,
postcheck `28f55c53…`, and spec `ecc1d988…` preserve the result.

| Realized proposal | AZ: sampled coordinates | BA: coherent units |
|---|---:|---:|
| Eligible coordinates | 184,320 | 608,256 |
| Changed BF16 cells | 172,056 | 154,318 |
| Changed matrices | 360 | 242 |
| Raw/declared Euclidean L2 | 1.585487 | 0.0310451 |
| Inverse-abs-floor relative norm | 15,309.34 | 595.122 |
| Strict predicted misses | 2/76 | 1/76 |
| Maximum predicted shortfall | 6.13e-8 | 1.187e-7 |

Both pass the fixed raw cap 5 and have no native or explicit V tensor edits.
BA's containing-box relative radius is 294.547, with absolute term 0.000294547;
one rounded cell exceeds the continuous box by `3.358e-7`. About 93% of BA's
squared raw norm lies in down-column changes, while attention changes are small
and not forced absent. The roughly 51-fold raw-norm difference mixes support
size, structure and objective-informed selection. The supports are not nested,
so this does not isolate capacity, establish a minimal model edit or predict
coding improvement.

## Actual candidate review and numerical nonidentity

The actual aggregate candidate review used verified Opus 5.5, zero tools and
public final text only; receipt `6d3722d5…` records 68.019 seconds. Both candidates
were covered. Disposition `e3d0ba57…` retains the following qualifications.

All 76 saved objective/fold arrays were compared on the 117,760 shared attention
coordinates, totaling 8,949,760 values. Coordinates and source bits match, but
Jacobians do not: aggregate relative L2 difference is 0.4098%, maximum absolute
difference 0.03084, and only four rows are bitwise equal. Median per-row relative
differences are 1.274% for LL and 2.875% for margin. Family aggregate differences
are Q 0.378%, K 0.387%, V 0.484% and O 0.334%. The existing probe tolerance is
not passed; no replacement tolerance was chosen after seeing the result.

A bounded diagnostic comparison of 34 overlapping LL arrays from failed D-v1
and complete D-v2 also found numerical nonidentity across the full observed
layout: aggregate relative L2 difference 2.391%. Failed partial arrays were
never fitted or mixed into the complete dataset. This does not identify the
cause. Capture storage, hook/probe layout and BF16 backward arithmetic differ
in execution details; no concrete semantic Jacobian error was found.

The pinned Transformers source (`d42819dd…`, lines 1231–1234) routes the K
projection output to both K and functional V before K normalization when V is
tied. The projection hook is upstream of both paths, with no detach between
them. D's actual zero-additive checks covered four sampled K and V cells, with
K error at most `1.46e-11`; both probed layers are sliding attention. This is not
an empirical probe of every tied layer. The existing CPU factor fixture also
covers attention coordinates.

The resulting comparison uses separately observed, numerically nonidentical
Jacobians. It makes no causal capacity attribution or universal significance/
zero-loss requirement. Existing norm/rounding caveats remain unchanged. Root
started the bounded realized-weight finite comparison at 03:13:06 UTC. Its
outcomes follow below. The ordinary round retains a fixed primary control, both
candidates and a descriptive empty repeat, with no averaging or result-based
control selection.


## Realized finite outcomes

The single finite sweep completed in 97.764 seconds. Base, AZ, BA and restored
base each matched the expected selected BF16 words across the same 360-tensor,
325,313-coordinate union. Restored-base NLL, margin and KL differences were zero.
Baseline correct-LL fold difference was zero against the D observation, and the
maximum per-case margin difference was `2.384e-7`. The original O2 service was
healthy with the same identity at 03:16:50.478 UTC; source, accepted AV state and
checkpoint file statistics were unchanged. Completion `ef935844…`, postchecks
`d2f2be34…`, replay summary `ada31f9f…` and independent review `72ae4e9f…`
preserve the result. No additional optimization followed the measurements.

The table reports equal-fold mean changes in correct-answer LL and margin
(nats/token); each purpose has two folds. Old science, short stages and long
stages remain separate. These are authored teacher-forced diagnostics, not
execution success or official coding scores.

| Purpose group | AZ correct LL | AZ margin | BA correct LL | BA margin |
|---|---:|---:|---:|---:|
| Implementation / algorithm / workflow | +0.059006 | +0.035557 | +0.043147 | +0.003633 |
| Original six science purposes | +0.044568 | +0.010528 | +0.044375 | +0.012559 |
| Short helper / composition stages | +0.089710 | +0.008918 | +0.072733 | +0.016659 |
| Long helper / composition stages | +0.040931 | +0.007326 | +0.041784 | +0.001565 |
| Plain-code retention | +0.197734 | +0.057355 | +0.120601 | +0.783643 |
| Tool call / no-call / arguments | +0.080441 | +0.023714 | +0.084982 | +0.035540 |
| Referent | +0.044755 | +0.044107 | +0.041643 | +0.019718 |
| Fact retention | +0.035120 | +0.007087 | +0.015846 | +0.009378 |

Both candidates increased correct-answer LL in all 38 folds. This did not mean
all desired margins improved. AZ's short-composition purpose mean margin fell
by 0.017429, and its plain-code fold-zero margin fell by 0.074526. BA's workflow
mean margin fell by 0.011523 despite positive workflow LL; AZ's workflow mean
margin increased by 0.092077. Each candidate also had one negative long-stage
margin fold. These tails remain visible alongside the positive group means.

Prediction errors were material. For the original science group, predicted
LL/margin gains were 0.054785/0.020011 for AZ and 0.054066/0.020041 for BA,
versus the realized values above. AZ's long-stage predicted margin gain was
0.061825 versus 0.007326 realized; BA's was 0.022825 versus 0.001565. The actual
strict bound counts were 44 misses out of 76 for AZ and 41/76 for BA, with
maximum shortfalls 0.064526 and 0.044350. These are descriptive nonlinear
prediction misses, not an added all-constraints acceptance gate.

BA's large plain-code margin gain decomposes into correct LL +0.120601 minus
foil LL -0.663042, with replay KL 0.203681. AZ's corresponding foil LL rose
by 0.140378 and KL was 0.009780. Workflow KL was 0.180510 for AZ and 0.170895
for BA. Thus BA's much smaller raw weight norm does not imply uniformly smaller
functional movement, and its foil suppression does not establish coding
ability. The finite evidence supports measuring both frozen candidates in the
predeclared ordinary round; it neither selects a new parent nor isolates the
causal benefit of coherent-unit capacity.

The fixed four-arm native-thinking comparison began at 03:20:36 UTC with AV as
the primary control, AZ, BA and the descriptive AV repeat. Its scores were
pending when this note was frozen. Conditional equivalent-reference work
remains unimplemented until actual transfer results inform the next decision.
## Realized BA unit geometry

A bounded 0.330-second CPU calculation reconstructed BA's exact BF16 deltas
from its saved spec and same-AV selected source arrays, with no model, solve or
checkpoint payload scan. Receipt `ba-unit-delta-geometry.json` under
`current-av-full-unit-d-v1` has SHA
`dc8250c67f7f8548de3c269459cacd79761299b7565138387627ca26b28631d7`.
Down columns contain 93.24% of total squared delta norm, versus 3.49% in UP rows
and 3.26% in gate rows. All 32 selected units changed, but the top eight account
for 81.76% of MLP squared norm; its participation ratio is 9.45 units. Changed
fractions are 50.42% of eligible down cells, 19.52% of UP and 19.15% of gate.

The down-column delta's cosine with its own source column ranges from -0.0332
to +0.0291, with median -0.00065. The realized direction is therefore nearly
orthogonal to its source columns, rather than a simple column-gain rescaling;
this is geometric evidence, not functional attribution. The largest floored
relative change is 294.88 on a source value of magnitude `7.56e-7`, below the
`1e-6` denominator floor. This concretely illustrates why that ratio does not
measure a typical relative perturbation.

Within each of the eight layers containing two selected units, the down-delta
span has numerical rank two; paired cosines range from -0.0798 to +0.4339.
Different layers were kept in separate coordinate blocks. A direct-sum rank
of 32 follows from that parameter support and does not establish a common
feature space, independent causal circuits or additive capability gains.
Official native-thinking outcomes remain the primary transfer evidence.

### Reproducing the geometry calculation

The calculation used only NumPy and four existing files. Set `A` to the local
analysis root, then run the recipe below with the saved inputs (no model or
checkpoint payload load). It prints numeric JSON to stdout; save that output
to a fresh path if desired. The per-layer Gram matrices deliberately omit
cross-layer feature comparisons. The direct-sum spectrum is the concatenation
of their eigenvalues, not a rank estimate for shared representations.

```sh
SPEC="$A/current-av-full-unit-d-v1/ba-run/BA/candidate/proposals/av-long-full-unit-ba-v1.json"
COVERAGE="$A/science-gpu-runs/current-av-full-unit-d-v2/matrix-correct-logprob/coverage.json"
SAMPLE="$A/science-gpu-runs/current-av-full-unit-d-v2/matrix-correct-logprob/sample-plan.npz"
SUPPORT="$A/current-av-full-unit-d-v1/selected/support.json"
OPENBLAS_NUM_THREADS=1 python - "$SPEC" "$COVERAGE" "$SAMPLE" "$SUPPORT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

spec_path, coverage_path, sample_path, support_path = map(Path, sys.argv[1:])
spec = json.loads(spec_path.read_text())
coverage = json.loads(coverage_path.read_text())
support = json.loads(support_path.read_text())
with np.load(sample_path, allow_pickle=False) as z:
    coordinates = np.asarray(z['coordinates'], dtype=np.int64)
    source = np.asarray(z['source_values'], dtype=np.float64)
target = source.copy()
entries = {e['tensor']: e for e in coverage['entries']}
for edit in spec['edits']:
    e = entries[edit['tensor']]
    start, count, width = e['start'], e['count'], e['shape'][1]
    c = coordinates[start:start + count]
    changed = np.asarray(edit['coordinates'], dtype=np.int64)
    keys = c[:, 0] * width + c[:, 1]
    wanted = changed[:, 0] * width + changed[:, 1]
    order = np.argsort(keys)
    indices = order[np.searchsorted(keys[order], wanted)]
    assert len(np.unique(wanted)) == len(wanted)
    assert np.array_equal(keys[indices], wanted)
    target[start + indices] = np.asarray(edit['set_values'], dtype=np.float64)
assert np.all((target.astype(np.float32).view(np.uint32) & 65535) == 0)
delta = target - source
families, units, down_vectors = {}, [], {}
for e in coverage['entries']:
    sl = slice(e['start'], e['start'] + e['count'])
    d, s = delta[sl], source[sl]
    f = families.setdefault(e['family'], {'eligible': 0, 'changed': 0, 'squared_l2': 0.})
    f['eligible'] += len(d)
    f['changed'] += int(np.count_nonzero(d))
    f['squared_l2'] += float(d @ d)
for group in support['units']:
    layer = group['layer']
    for unit in group['indices']:
        row = {'layer': layer, 'unit': unit, 'families': {}}
        for family in ('gate', 'up', 'down'):
            e = entries[f'model.language_model.layers.{layer}.mlp.{family}_proj.weight']
            sl = slice(e['start'], e['start'] + e['count'])
            c = coordinates[sl]
            mask = c[:, 1 if family == 'down' else 0] == unit
            axis = c[mask, 0 if family == 'down' else 1]
            order = np.argsort(axis)
            assert np.array_equal(axis[order], np.arange(e['shape'][0 if family == 'down' else 1]))
            d, s = delta[sl][mask][order], source[sl][mask][order]
            dn, sn = float(np.linalg.norm(d)), float(np.linalg.norm(s))
            relative = np.abs(d) / np.maximum(np.abs(s), 1e-6)
            row['families'][family] = {
                'eligible': len(d), 'changed': int(np.count_nonzero(d)),
                'squared_l2': dn * dn, 'raw_l2': dn,
                'delta_source_cosine': float(d @ s / (dn * sn)) if dn and sn else None,
                'max_relative_floor_1e6': float(relative.max()),
                'source_abs_at_max_relative': float(abs(s[int(relative.argmax())])),
            }
            if family == 'down':
                down_vectors[(layer, unit)] = d
        row['squared_l2'] = sum(v['squared_l2'] for v in row['families'].values())
        row['changed_fraction'] = sum(v['changed'] for v in row['families'].values()) / sum(v['eligible'] for v in row['families'].values())
        units.append(row)
layer_grams = []
for group in support['units']:
    x = np.stack([down_vectors[(group['layer'], j)] for j in group['indices']], axis=1)
    gram = x.T @ x
    singular = np.sqrt(np.maximum(np.linalg.eigvalsh(gram), 0))[::-1]
    tolerance = max(x.shape) * np.finfo(np.float64).eps * singular[0]
    cosine = float(gram[0, 1] / np.sqrt(gram[0, 0] * gram[1, 1])) if x.shape[1] == 2 else None
    layer_grams.append({'layer': group['layer'], 'units': group['indices'], 'gram': gram.tolist(), 'rank': int(np.count_nonzero(singular > tolerance)), 'rank_tolerance': float(tolerance), 'pair_cosine': cosine})
squares = np.sort(np.array([u['squared_l2'] for u in units]))[::-1]
mlp_squared = float(squares.sum())
total_squared = float(delta @ delta)
for f in families.values():
    f['raw_l2'] = float(np.sqrt(f['squared_l2']))
    f['changed_fraction'] = f['changed'] / f['eligible']
    f['share_total_squared_l2'] = f['squared_l2'] / total_squared
print(json.dumps({
    'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (spec_path, coverage_path, sample_path, support_path)},
    'raw_l2': float(np.sqrt(total_squared)), 'changed': int(np.count_nonzero(delta)),
    'families': families, 'unit_rows': sorted(units, key=lambda u: -u['squared_l2']),
    'top1_top4_top8_mlp_shares': [float(squares[:n].sum() / mlp_squared) for n in (1, 4, 8)],
    'unit_participation_ratio': float(mlp_squared ** 2 / (squares @ squares)),
    'layer_local_down_grams': layer_grams,
}, indent=2))
PY
```


## 2026-11-07: AV / AZ / BA / identical AV repeat

One exploratory load-time-overlay round completed at 03:33:28 UTC. All four original containers returned healthy with the same CIDs; accepted AV remains unchanged. All 90 items per arm are scored, with no unknown, empty or truncated official result. These rows represent logical overlays, not separately saved AZ/BA checkpoints.

| Arm | HE /30 | MBPP /30 | LCB /20 | SciCode mean /10 | Equal-source code % | Delta vs fixed primary pp |
|---|---:|---:|---:|---:|---:|---:|
| Primary empty AV | 28 | 22 | 16 | 0.403015873 | 71.742063 | — |
| AZ | 29 | 21 | 14 | 0.423015873 | 69.742063 | −2.000000 |
| BA | 29 | 23 | 16 | 0.403015873 | 73.408730 | +1.666667 |
| Identical empty AV repeat | 29 | 22 | 16 | 0.423015873 | 73.075397 | +1.333333, descriptive only |

BA has three binary gains (HumanEval/152, HumanEval/95, Mbpp/801) and one loss (HumanEval/151). Its code delta CI is [−1.666667, +5.000000] pp; SciCode and LiveCodeBench are unchanged. AZ gains HumanEval/95, loses abc390_g, arc195_b and Mbpp/792, and improves SciCode/58 from 0.8 to 1.0; its CI is [−6.250000, +1.750000] pp. The repeat gains HumanEval/95 and the same SciCode partial credit. The repeat is not substituted for, or averaged with, the declared primary, and one repeat is not an IID variance estimate.

Each arm records 119 formal HTTP attempts, no retries/errors, plus one separate one-token startup call. Total formal input/output tokens are 377,322/1,351,034; the four startup calls add 84/4 tokens. Serving counters include the four startup length finishes, while official scored truncation is zero. Agent steps remain unknown.

Root selected only BA for physical save and fresh ordinary AV/BA full14 tradeoff evaluation; this is not adoption. AZ's negative result remains in the catalog. No capacity-only interpretation is supported because eligibility, step scales and measured Jacobians differ.

Numeric closure: current-av-az-ba-overlay-dev90-v1/numeric-final-summary.json a049be093a44e31a83cda4b280c6df98ef52f1b8218919e6d02175bb56ed10fd. Independent arithmetic review a30900bc26553cb2626c6bcff73bb6bc92f066f7a5963a02c90afdb47565a35d. Completion 937b2d37c8d008c711e01ae86440ecdba20fa57ef6282043050735dfc8e132b9.

Append-only v15: 151 rows, all 147 previous row objects unchanged.

- catalog.json: e45bc9c5d097d04cbdfaca9e0dde3a3ac8a95e3a227fff6839714f2109a4c648
- preservation.json: cd7ca95e9e322cc303c7a755478afe71f6b631acd5322f65987eff7488df45f0
- export-reproduction.py: cca97be80de5d5b5be7cae993675573d31086c3cac33a330ac93a5133894129e

The prepared [ordinary AV/BA full14 recipe](scripts/launch-av-ba-retention-full14.py) preserves tier10/dev, the fixed 778-item cohort (746 capability items plus 32 likelihood probes), c32/logprob8/agent1, native generation, archived benchmark source and symmetric numeric observers. AV0 is the retained control and saved BA3 is the candidate. Its default unbound preview launches nothing; BA's verified freeze, deployment, fresh endpoint snapshots and a separate root decision must bind before execution.

Future helper diagnostics now retain only an allowlisted exception category and setup/execution phase. Prior runtime-only artifacts cannot recover their class, and no new model diagnostic has run.


## Current parent: highest recorded code90 score

The user's explicit policy now selects the highest measured score on the fixed four-source official code90 cohort, superseding the earlier local-pair gain rule. The151-run ledger contains59 code90 runs. Saved AD's74.742063% is the highest ordinary code90 score; the AR logical overlay ties that value, while separately saved AR scores73.075397%. Different cohorts (screen95, hard30, known64, full14 code26) are not ranked together.

The working-parent metadata was restored to saved AD, tree27629fd3, with no weight or service change. AV's prior accepted state and all AV-anchored AZ/BA measurements remain historical evidence. BA materialization and the prepared AV/BA full14 comparison are HELD: no copy, deployment, or full14 launch occurred. The old AV-pinned templates remain unchanged and cannot validate against the new parent state.

Current AD state: b70fe5e9d519c7b4459af26f505471649c3313483058558e0796be3dad64440f. Policy: working-incumbent-policy-highest-code-v2.json eaf2112552ba45b26c913756ee128111848b64b7184afd74da3ae425886e85de. Decision: current-ad-highest-code-policy-v1/working-decision.json a9eaa1fd499b7363a2f44f375fbcf7bd3d2732ca28c8b79129f26ee228e5bc7b. This selects the observed record; it does not prove a stable expected-score ordering or code>90.
