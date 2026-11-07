# F03 proposed native scope: independent static review (2026-11-07)

This reviews the parent's provisional F03 design before freeze. **No solver, candidate plan/spec, gradient recomputation, model/GPU call, service operation or checkpoint edit was performed.** Only current solver source and the completed dependency report were read. Feasibility has not been tested.

The proposed 180 tensors (post-attention norm, post-feedforward norm and layer scalar at all 60 layers) are representable by the existing **schema 1**, `kind=native_coupled_objectives_v1` and an exact `native_tensors` allowlist. Schema 2 expects matrix observations and is unnecessary; explicit `observed_blocks=["native"]` is not supported. Full observation/validation remains all 421 native tensors; eligible variables become **645,180 coordinates**, with represented source L2 **960.7201571980261**, not the full-native 70,134.9646. Other 241 native tensors and matrices remain ineligible.

The three selected families contain **98.5174% / 99.2470%** of A1/F2 correct-LL gradient squared norm. Across all 24 rows the range is **96.9323–99.4823%**. This is raw-coordinate derivative energy, not ULP-weighted energy or proof that excluded coordinates are behaviorally unimportant. The parent independently verified the actual HF residual structure: post-normalization scales a branch before residual addition, but `layer_scalar` multiplies the whole layer output including residual. These three operators therefore are not identical branch gains. Their joint selection is a concrete structural hypothesis, not established causal localization.

## Constraint units and intended losses

Current `minimum_delta` fields are changes in teacher-forced nat/token objectives. They are not absolute final values or WSP points. Rows must be complete and ordered: all 12 correct-LL folds first, then all 12 margin folds; no per-case reordering or omitted guards.

- A1 margin baseline is **−0.260095477104187**. An absolute final target of **+0.02** requires `minimum_delta=+0.28009547710418703`, not +0.02.
- A1/F2 correct LL +0.05, A2/F1 >=0, other eight >=−0.02 are unambiguous as delta constraints. This preserves sibling LL only locally; it does not prove preserved generated behavior.
- Under the same delta interpretation, F2 margin >=−0.10 leaves a predicted floor **+0.4364991426467896**. It does not optimize the unrelated foil direction.
- Remaining margins >=−0.05 permit B2 **+0.045379→−0.004621** and E1 **+0.040020→−0.009980**. That is an explicit permitted pair-ranking loss, not retention of every positive ranking. If pair-positive preservation is intended instead, the smallest change is to set these two lower deltas to `-baseline` (nonnegative final margin) or a separately frozen positive epsilon. Otherwise keep −0.05 and state the allowed flips. No constraint was changed by this review.

A +0.280095 A1 margin delta is substantially larger than the +0.05 LL goals, but scales are normalized by gradient norms inside the continuous solver. Magnitude alone does not prove infeasibility. Static evidence is insufficient to certify feasibility after clipping/BF16 realization; the parent's frozen single solve must retain any failure.

## Euclidean cost, box and 16-round behavior

`cost_metric={"type":"euclidean"}` minimizes raw weight L2 in each continuous correction. This is a declared parameter-space regularizer, not an equal functional effect across scalar and vector operators. The implementation does **not** prove a globally minimum-L2 BF16 solution with box constraints.

`relative_radius=.01, absolute_radius=1e-6` means per-coordinate continuous radius `max(abs(base)*.01,1e-6)`. It is neither a global 1% L2 cap nor a strict bound on stored BF16 differences. The absolute floor applies below `abs(base)=1e-4`; the selected post-attention minimum 2.95639038e−5 permits about **3.3825%** continuous relative change before rounding. The other two families' minima exceed 1e−4.

Each iteration adds an unconstrained minimum-L2 correction to an accumulated F64 proposal, clips that proposal, then realizes **original AV + F64 proposal → F32 → BF16**. It never successively rounds and edits the last BF16 checkpoint. Clipping is outside the convex solve; it can undo feasibility. A representable example is base 1.25, continuous +0.0125 (1%), rounding 1.2625 to 1.265625: actual +0.015625 (1.25%). A strict actual-value box would require an explicitly different algorithm; it is not what current settings mean.

16 is an **upper bound**, with early stop if all predicted shortfalls <=1e−10. Otherwise the saved best round minimizes shortfall-vector L2 first, then realized weighted L2. This is a fixed diagnostic heuristic, not a discrete optimum or infeasibility proof. No amplitude/round search is needed to test this one frozen design.

## Failure visibility and minimum procedural requirements

The solver can emit a spec and exit successfully when `predicted_constraints_satisfied=false`. Final prediction/minimum-delta vectors retain all 24 rows. Each historical round retains **maximum** prediction shortfall, norms, change count and dual diagnostics; **it does not retain all 24 historical residuals**. Do not describe max-only history as full per-round constraint vectors.

A continuous dual nonconvergence explicitly says it is not proof of infeasibility. Timeout/nonconvergence may occur before the output directory/report is created. The parent execution wrapper must retain command, pinned inputs, timing, exit status and stderr/failure receipt separately. No optimizer exception should erase the attempted trial.

Minimum corrections are therefore to freeze the exact delta interpretation, chosen-scope denominator and permitted pair flips; call .01 a pre-round continuous radius; inspect actual stored differences and all 24 final residuals independently; and distinguish unsuccessful solver status from measured capability. The provisional scope/cost/16-round choices themselves are supported by the API and have no demonstrated static impossibility. No claim of behavioral gain or optimum is made.

Reviewed implementation SHA256:

- `scripts/experiments/gemma4-purpose/propose_native_objectives.py`: `9633698b9eab5d1e9c5e49d399f897d9147ab37f32c3ecd4b61232e3dc5e2d83`.
- `scripts/experiments/gemma4-purpose/joint.py`: `6480e16c8f453b4a8f3914a502e1abd6ce31e21ef01759f258715baf557e703c`.
- `src/kairyu_bench/wsp/weights/finite_response.py`: `2f4b474089290d66a291d61eea176743d2502303943c6fded5b4331ebbff77af`.

All arithmetic uses the saved report; [facts and exact read-only calculation](records/2026-11-07-f03-static-review/) retain the selected denominator/energy and two near-zero-margin counterexamples. No new reusable tool or tests were needed for this source/document review.
