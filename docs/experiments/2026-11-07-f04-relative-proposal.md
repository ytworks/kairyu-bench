# F04: one relative-cost vector proposal, not measured

One existing CPU solver preview and one solve completed successfully in 11.651 seconds total (10.986 seconds inside the solver). All 421 native observations and the same 24 F03 constraints are retained; eligibility is the 361 complete vectors, with all 60 whole-layer scalars fixed. The original F03 eligibility was 180 tensors, while its observation set was 421. Checkpoint, authored cases, both collections and rounding bounds are unchanged: continuous relative radius 0.01, absolute radius 1e−6 and at most 16 rounds. F04 changes only scope, relative cost with floor 1e−4, identifier and the explicitly pinned parent-state rationale.

AV is the latest completed comparable parent at authorization. F03 was still running before and after this CPU operation and is not ranked here. **Before any F04 checkpoint materialization or evaluation, check the parent again: if F03 has become the highest model, retain this proposal as unscored with `unscored_reason=parent_changed` and redesign from that new parent.** No checkpoint was created or served.

The relative geometry motivated testing a different metric, without assuming superiority: full-421 relative gradients shifted energy from postnorms into whole-layer scalars. Fixing those scalar controls allows all seven vector families to vary jointly. The solver retained 429 rounded BF16 changes across 140 tensors; all seven families have at least one change. Native scalar changes are zero. ΔL2 is 2.0357791145; relative weighted ΔL2 is 0.1167916772. The 361-vector source denominator is 70,134.9643459, dominated by large norm families and not the whole-model norm. No small relative-to-that-denominator safety conclusion is made.

Round 9 met all 24 *linear predicted* bounds; rounds 1–8 shortfalls and every round remain in the report. Continuous L2 is 6.8455045, rounded-to-continuous cosine 0.2217555 and rounding-relative L2 error 0.9780312. Thus even though constraints use the rounded BF16 delta, these local predictions are not a measured finite response or capability gain. F2’s authored foil also differs from its observed off-by-one native failure.

| Case | Predicted correct-LL delta | Required minimum | Predicted margin delta | Required minimum |
|---|---:|---:|---:|---:|
| A1 | +0.125929100 | +0.050000000 | +0.281535693 | +0.280095477 |
| A2 | +0.051445283 | +0.000000000 | -0.024985056 | -0.050000000 |
| B1 | +0.003252991 | -0.020000000 | -0.009074366 | -0.050000000 |
| B2 | +0.001842854 | -0.020000000 | -0.003542894 | -0.050000000 |
| C1 | +0.014419794 | -0.020000000 | -0.003692707 | -0.050000000 |
| C2 | +0.010512267 | -0.020000000 | -0.001019640 | -0.050000000 |
| D1 | +0.015275780 | -0.020000000 | -0.001006637 | -0.050000000 |
| D2 | +0.015073961 | -0.020000000 | +0.001741858 | -0.050000000 |
| E1 | +0.012318025 | -0.020000000 | +0.003895109 | -0.050000000 |
| E2 | +0.021639908 | -0.020000000 | -0.001425364 | -0.050000000 |
| F1 | +0.027923242 | +0.000000000 | -0.003792397 | -0.050000000 |
| F2 | +0.050216079 | +0.050000000 | -0.003931544 | -0.100000000 |

Changed coordinates by family: `input_layernorm.weight` 67, `norm.weight` 3, `post_attention_layernorm.weight` 52, `post_feedforward_layernorm.weight` 34, `pre_feedforward_layernorm.weight` 18, `self_attn.k_norm.weight` 149, `self_attn.q_norm.weight` 106.

OPENBLAS, OMP and MKL thread limits were all 1, with idle-class I/O. The recorded launcher used nice +15 and the child command added +15 again, reaching the lower-priority nice 19 ceiling; no evaluation CPU priority was raised. This background CPU solve overlapped F03 WSP and must be included when interpreting its elapsed time rather than calling that latency unloaded. No GPU, model/API request, extra test, retry, new solver feature or service operation was performed.

[Pinned plan](records/2026-11-07-f04-relative-v1/proposal-plan.json), [parent state](records/2026-11-07-f04-relative-v1/parent-state.json), [one-time commands/source](records/2026-11-07-f04-relative-v1/prepare-and-solve.py), [complete report and round history](records/2026-11-07-f04-relative-v1/proposal/report.json), [exact proposed BF16 spec](records/2026-11-07-f04-relative-v1/proposal/fresh-native-relative-f04.json) and [completion receipt](records/2026-11-07-f04-relative-v1/completion.json) preserve the entire trial. Preview and solver stdout/stderr plus timings are alongside them. There were no failed solver attempts.
