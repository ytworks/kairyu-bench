# Referent and code composition: independent F01 evidence

All results are from the same AV raw gradients and the fixed F01 finite replay. No previous candidate ranking, old adoption decision or prior target was used. This analysis makes zero model calls, edits no checkpoint and generates no candidate. The [recipe](../../scripts/experiments/gemma4-purpose/fresh_joint_referent_analysis.py) checks all raw gradient/sample hashes against the F01 report and reconstructs the exact F01 BF16 delta. It passes Ruff. The first auxiliary command used a mistyped input root and stopped before reading data; it was corrected. An initial lint check found ambiguous local names, corrected before successful analysis.

[Full evidence](evidence-2026-11-07-f01/referent-dependency-analysis.json) contains all 4,352 cosine rows: four referent objective/fold vectors against all sixteen referent/algorithm/implementation/code vectors, separately over the entire sample, seven families and sixty layers. It also records all 5,168 F01 linear contributions (76 vectors × 68 groups), plus every one of the 76 finite/predicted errors. Sample scope is 209,920 stratified coordinates from 410 matrices, not the complete model. The calculation took 2.879 seconds. Family/layer sums recover every original prediction within 1e-12.

## Evidence and limits

Referent correct-answer LL shares local directions with implementation LL in all four fold pairings (cosines 0.168–0.242). This support appears in q, k, o, gate and down, while v and up include conflicts. Algorithm relationships are mixed. Fourteen of sixteen referent-versus-code objective/fold pairings are negative globally; all sixteen are negative within q and o. Those conflicts argue for explicit joint constraints, not omitting code or removing a matrix family.

Referent LL fold-to-fold cosine is 0.614; its margin fold-to-fold cosine is only 0.139. Referent LL fold 0 against referent margin fold 1 is −0.212. Averaging those folds/objectives would obscure competing derivatives.

For F01, referent fold 1's k-family contribution is −0.004402 LL and −0.007188 margin; its o-family margin contribution offsets +0.004830. The LL losses include layers 23 (−0.004076), 31 (−0.003918) and 35 (−0.002317), offset by layers 28 (+0.004263), 29 (+0.003847) and 33 (+0.002388). These are first-order dot products of one fixed edit, not causal attributions of the actual finite loss; they do not justify dropping k or those layers.

The complete finite error mean absolute value is 0.027207 and median is 0.007621; the maximum, +0.889774, occurs in code fold 0 margin. Non-code mean absolute error is 0.009364 and maximum is 0.035814, the referent fold 1 LL miss. Eleven nonprimary constraints had a positive prediction and a negative finite response, even though only the two referent fold 1 retention bounds failed. Positive linear directions alone cannot certify a candidate.

## Concrete next hypothesis

The parent fixed one new objective: retain independent algorithm and implementation LL/margin +0.03 targets and code LL +0.01/margin 0, add referent LL/margin +0.03 for both folds, and retain every other LL/margin −0.03 constraint. Preserve AV parent, all 410 matrices / 209,920 sample cells / seven families, native fixed, Euclidean cost and the same box/round policy. This is a new joint solve from AV, not an amplitude change or addition of F01.

The +0.03 referent targets are an explicit new capability hypothesis, not a calibrated safety margin. F01's worst negative error exceeded 0.03, so the same error carried over would erase a +0.03 prediction. We have only one finite perturbation and cannot infer that the error will transfer unchanged or disappear. The fixed all-case AV → candidate → AV finite experiment must therefore precede promotion; unmet predictions and all finite losses must remain visible.

The authored set has six referent cases total (three per fold). Reusing both folds to redesign the objective makes both development data, not a holdout. Matrix sampling, teacher-forced public-answer scores, one base point and single examples also limit generalization. Canonical full WSP dev, other-domain results and later final test are needed to assess ability. No source/problem exclusion or after-the-fact per-fold averaging is warranted.
