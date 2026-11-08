# AV relative-coordinate sensitivity: one CPU pass

Existing AV data only: 12 singleton cases × correct-LL/margin, all 421 native tensors and 1,331,516 coordinates. One CPU execution took 2.817 s with one BLAS thread; all 44 used input/source pins matched. No model, GPU, solver, candidate, new test or baseline measurement was run. F03 WSP30 remained incomplete at this analysis, so no model ranking or next-base decision is made.

`g_relative = g × max(abs(weight), 1e-4)` is the derivative in coordinates `x = delta_weight / max(abs(weight), 1e-4)`. A Euclidean norm of x corresponds to relative-change cost with this explicit floor. Each energy is a sum of squared derivative coordinates; its fraction divides by the whole 421-tensor energy of the same case/objective. This is full coverage of the native scalar/vector subset, not the 31B model matrices. The floor affects 1,425 coordinates.

| Native family | A1 LL raw % | relative % | F2 LL raw % | relative % | Success10 mean raw % | relative % |
|---|---:|---:|---:|---:|---:|---:|
| input_layernorm.weight | 0.0069 | 4.8270 | 0.0027 | 1.7352 | 0.0029 | 5.7982 |
| layer_scalar | 1.4338 | 65.0647 | 3.2275 | 87.1264 | 0.2186 | 53.5654 |
| norm.weight | 0.0000 | 0.1295 | 0.0001 | 0.3926 | 0.0003 | 1.1530 |
| post_attention_layernorm.weight | 62.6063 | 6.1403 | 56.3055 | 1.6836 | 57.4260 | 4.6614 |
| post_feedforward_layernorm.weight | 34.4773 | 3.9246 | 39.7141 | 2.2691 | 41.1204 | 7.3905 |
| pre_feedforward_layernorm.weight | 0.0057 | 3.3072 | 0.0094 | 2.3030 | 0.0084 | 6.8353 |
| self_attn.k_norm.weight | 1.4550 | 8.5416 | 0.7337 | 2.3154 | 1.2132 | 10.6361 |
| self_attn.q_norm.weight | 0.0150 | 8.0651 | 0.0071 | 2.1747 | 0.0103 | 9.9600 |

Relative cost removes most postnorm concentration but makes the 60 layer scalars dominant: 65.06% of A1 LL energy, 87.13% of F2, and a 53.57% mean across the ten successful cases. These are equal-weight averages of per-case fractions; all individual values remain in the JSON. Thus changing the metric moves concentration rather than demonstrating a balanced or safer direction.

| Geometry across the same rows | Raw | Relative |
|---|---:|---:|
| A1/F2 correct-LL cosine | 0.318507 | 0.012068 |
| A1/A2 correct-LL cosine | 0.499058 | 0.375263 |
| F2/F1 correct-LL cosine | 0.448718 | 0.084268 |
| A1 own LL/margin cosine | 0.621530 | 0.866225 |
| F2 own LL/margin cosine | 0.055294 | 0.317445 |
| Positive/negative LL pairs (66) | 66 / 0 | 58 / 8 |
| Positive/negative margin pairs (66) | 37 / 29 | 33 / 33 |
| Positive/negative all-objective pairs (276) | 173 / 103 | 160 / 116 |

A1 LL now conflicts with successful B1 (−0.009783) and B2 (−0.131130); F2 LL conflicts with successful E1 (−0.157082). These are coordinate-metric alignments, not a proof that simultaneous constraints are infeasible. They refute the simple claim that relative cost is uniformly better aligned. If later considered, all-family simultaneous constraints and per-tensor/family realized BF16 changes would need to be judged explicitly; this analysis selects neither a support set nor a direction.

The evidence is local AV teacher-forced sensitivity, with forced empty thought and authored public-answer tokens. A1 had 0/16 and F2 4/16 native execution trials; the other ten had 16/16, retaining the B1/B2 method-requirement caveat. F2’s foil removes zeros, whereas its generated failure was an off-by-one starting index. Neither this metric change nor margin alignment establishes finite BF16 response, native generation improvement or a WSP gain. This does not re-solve F03 or explain its eventual score.

[Full small JSON](records/2026-11-07-native-relative-energy-v1/report.json) retains eight-family energies/fractions for every one of the 24 rows, both full 24×24 cosine matrices, A1/F2 against each successful case, coordinate counts and source denominators. [One-time source](records/2026-11-07-native-relative-energy-v1/analyze.py) invokes only the existing `joint.read_collection` loader. [Receipt](records/2026-11-07-native-relative-energy-v1/receipt.json) records the one execution and hashes.
