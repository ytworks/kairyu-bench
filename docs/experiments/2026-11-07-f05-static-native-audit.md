# F05 static native-spec audit

One existing native_spec_audit invocation independently reconstructed F05's exact BF16 spec against F04 in3.588316s (wall3.759590s). It read all421 native tensors/1,331,516 source words and checked them against the saved measurement source. Matrices were not read or edited. This audit made no solver/model/Docker call, checkpoint write or service change.

Eligible scope is361 vectors/1,331,456 coordinates;60 scalars stay unchanged. **26,417 coordinates in361 tensors change**, with delta L2 **52.2944053830**. Relative L2 **0.0007456253031** uses source L2 **70,134.9661329 over all eligible coordinates**, not only changed coordinates and not the full model. Whole native source L2 including60 unchanged scalars is70,134.9664104. The solver's declared relative-metric weighted norm is0.936642257920, a distinct metric.

After BF16 rounding, **653 coordinates exceed the declared continuous box** (relative1%, absolute floor1e-6). Maximum excess1.56, actual/radius ratio1.333333, maximum per-coordinate relative change1.333333%, maximum absolute delta8. No sign flips. These are preserved facts; no clamp, rescale, plan rewrite or rerun was performed.

| Native family | Changed coordinates | Delta L2 | Box exceedances |
|---|---:|---:|---:|
| input_layernorm.weight | 4536 | 42.4409865 | 127 |
| layer_scalar | 0 | 0 | 0 |
| post_attention_layernorm.weight | 3481 | 0.853037433 | 87 |
| post_feedforward_layernorm.weight | 4292 | 2.82757183 | 50 |
| pre_feedforward_layernorm.weight | 5559 | 29.4352202 | 81 |
| self_attn.k_norm.weight | 4193 | 0.0472233859 | 127 |
| self_attn.q_norm.weight | 3832 | 0.399049775 | 170 |
| norm.weight | 524 | 7.62568061 | 11 |

All38 ordered predictions independently match the original solver exactly (max discrepancy0);36 meet their frozen constraints. Margin A1 predicts0.0998884202013 against0.1 (shortfall0.000111579798704); J1 predicts0.0999481658819 against0.1 (shortfall0.0000518341181305). Chosen round15 of16. Reported rounding relative L2 error is0.820878101690 and cosine to the continuous direction0.615749115111. Neither linear predictions nor static norms establish capability improvement.

The original parent-state lacked the audit tool's dependency field. The new audit-input-v1 derives only an opaque parent-state reference: all original plan numeric fields, constraints, costs, rounding, eligibility, collection paths, checkpoint and cases are exactly unchanged. Original report/spec and solve-execution bytes are copied unchanged under the tool's expected names. The new dependency joins all19 F04 measured correct-logprob/margin rows; no AV values are used. A later cross-check matches all19 values exactly to dependency report d1126f39…6ea. The first optional cross-check used the wrong external key; its KeyError is recorded, and the actual audit was not rerun.

Report SHA256:4fe03834011c41512674fc5febd9238a65349f57bbd1d8d2b9dd5d9495afe468. Derived plan:0981476fd24eea3c43a3aea87e3550aec8066f934830f4096afd51198405cfd1. Equality proof:f3cd6bb60780953aa515a5c20894b756495192ee3d6db46e232a954a985d79c2. Original files and every source pin named by the solver report remained unchanged.

Root separately saved and verified F05; the preserved faststore verification reports the same26,417/361 and L2, with tree fcc4681343c1a0c338157bee767df2024f37290469e55dbc417723717f4ac176. This static audit itself reconstructed the spec and is not a fresh hash/read of the saved candidate. No finite/model response was measured.

[Every38 row,421 tensor, family/layer metric, original/derived inputs and execution source](records/2026-11-07-f05-static-native-audit-v1/) is retained, using lossless gzip where needed. The unchanged tool contains a historical text saying24-row histories; actual cohort/row counts are38 and complete. Per-round per-constraint vectors were never stored and remain unknown. Shared source was not modified.
