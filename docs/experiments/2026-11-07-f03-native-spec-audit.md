# F03: independent BF16 native-spec audit (2026-11-07)

The single fixed F03 spec was reconstructed once on CPU in **2.877 s**, without a new solve, rescaling, checkpoint write, service action or model call. Intended eligibility is **all 60 layers / 180 tensors / 645,180 coordinates**. Actual BF16 differences affect **59 layers / 112 tensors / 60,473 coordinates**; 68 eligible tensors are unchanged, including all eligible tensors in layer 59. All 421 observed native tensors are reported, including the 241 ineligible tensors with zero differences. Matrix payloads were not read or edited. This is a spec-value audit, **not a saved-candidate or finite-response claim**.

The audit compared all **1,331,516 current native source words**, prior full attestation and current checkpoint stat continuity. No whole-model payload rehash was performed. Exact eligible-target F32/BF16 representation SHA matches the solver: `232fcba434c74422a29f2326538ba977edc4c9e79c8a5c182a0a5abffdd0c184`. Independent dot products reproduce all 24 saved predictions with maximum absolute disagreement **0**.

## Actual differences and denominators

Actual L2 is **0.0043893681725189855**. Its denominator is the complete selected-native source L2 **960.7201571980252**, giving relative L2 **4.56883114155e−6**. It is not divided by the full-native 70k norm or only changed cells. Positive/negative coordinate deltas are **29,982 / 30,491**; 584,707 eligible coordinates remain unchanged. No sign flips occurred.

Maximum absolute difference is **0.00048828125**; maximum nonzero-source relative difference is **3.225806%**. The declared pre-round box is `max(.01*abs(source),1e-6)`: its absolute floor is active at 1,406 eligible coordinates. **16,754** actual BF16 differences exceed this continuous box, by up to **3.0517578125e−5**, with max actual-delta/box ratio **1.3333333333333333**. Relative-only 1% is exceeded at 17,711 coordinates. These are recorded rounding effects, not silently clamped or declared within 1%.

| Native family | Eligible / changed tensors | Changed cells | Actual L2 | Full family source L2 | Relative L2 | Box-exceed cells |
|---|---:|---:|---:|---:|---:|---:|
| input_layernorm.weight | 0 / 0 | 0 | 0 | 40876.2542763 | 0 | 0 |
| layer_scalar | 60 / 2 | 2 | 0.000690533966002 | 6.23943785358 | 0.000110672464765 | 0 |
| post_attention_layernorm.weight | 60 / 58 | 46834 | 0.0027385114604 | 558.637724885 | 4.90212411088e-06 | 14893 |
| post_feedforward_layernorm.weight | 60 / 52 | 13637 | 0.00336009981652 | 781.580566671 | 4.29910870332e-06 | 1861 |
| pre_feedforward_layernorm.weight | 0 / 0 | 0 | 0 | 56934.2805684 | 0 | 0 |
| self_attn.k_norm.weight | 0 / 0 | 0 | 0 | 14.7536933445 | 0 | 0 |
| self_attn.q_norm.weight | 0 / 0 | 0 | 0 | 134.59325826 | 0 | 0 |
| norm.weight | 0 / 0 | 0 | 0 | 2364.63083799 | 0 | 0 |

All eight families, all 60 layers plus final norm and all 421 native tensor rows are in [report.json](records/2026-11-07-f03-native-spec-audit/report.json). Each row includes positive, negative and unchanged counts, full-source denominator, actual norm/maximum, floor and box-excess counts, and all 24 signed prediction contributions. Zero-difference families/tensors are retained; they do not indicate absent ability.

## All 24 predictions, including unmet and nonpositive values

The fixed solver completed with **22/24 predicted constraints met**; `predicted_constraints_satisfied=false` remains unchanged. F2 LL is +0.0446587314292 versus +0.05 (shortfall 0.00534126857082). A1 margin changes by +0.267741950406 versus +0.280095477104 (shortfall 0.0123535266982): predicted final margin +0.00764647330180 is positive but below the frozen +0.02 target. No post-data threshold change is made.

| Objective | Case | Baseline | Required delta | Predicted delta | Predicted after | Shortfall | Met |
|---|---|---:|---:|---:|---:|---:|---|
| correct-logprob | A1 | -1.37358403 | 0.05 | +0.115549309 | -1.25803472 | 0 | yes |
| correct-logprob | A2 | -1.64746261 | 0 | +0.0284756532 | -1.61898695 | 0 | yes |
| correct-logprob | B1 | -1.72073543 | -0.02 | +0.0109354195 | -1.70980001 | 0 | yes |
| correct-logprob | B2 | -1.10795224 | -0.02 | +0.00519979694 | -1.10275244 | 0 | yes |
| correct-logprob | C1 | -2.42754483 | -0.02 | +0.0285094638 | -2.39903537 | 0 | yes |
| correct-logprob | C2 | -1.2776953 | -0.02 | +0.0124711622 | -1.26522414 | 0 | yes |
| correct-logprob | D1 | -1.51190495 | -0.02 | +0.0168610455 | -1.49504391 | 0 | yes |
| correct-logprob | D2 | -1.10669422 | -0.02 | +0.0128985866 | -1.09379563 | 0 | yes |
| correct-logprob | E1 | -1.08664787 | -0.02 | +0.0114501453 | -1.07519772 | 0 | yes |
| correct-logprob | E2 | -1.85462403 | -0.02 | +0.0169266034 | -1.83769743 | 0 | yes |
| correct-logprob | F1 | -2.10941315 | 0 | +0.0342379788 | -2.07517517 | 0 | yes |
| correct-logprob | F2 | -1.88131273 | 0.05 | +0.0446587314 | -1.836654 | 0.00534126857 | NO |
| margin | A1 | -0.260095477 | 0.280095477 | +0.26774195 | 0.0076464733 | 0.0123535267 | NO |
| margin | A2 | 0.228590012 | -0.05 | -0.0203960832 | 0.208193928 | 0 | yes |
| margin | B1 | 0.116192222 | -0.05 | -0.00100005467 | 0.115192167 | 0 | yes |
| margin | B2 | 0.0453791618 | -0.05 | +0.000669269809 | 0.0460484316 | 0 | yes |
| margin | C1 | 0.633769274 | -0.05 | -0.000495107266 | 0.633274166 | 0 | yes |
| margin | C2 | 0.416755795 | -0.05 | -0.000619727107 | 0.416136068 | 0 | yes |
| margin | D1 | 0.286189437 | -0.05 | -0.00126095186 | 0.284928485 | 0 | yes |
| margin | D2 | 0.45308423 | -0.05 | -0.000573389672 | 0.452510841 | 0 | yes |
| margin | E1 | 0.0400202274 | -0.05 | +0.00200453399 | 0.0420247614 | 0 | yes |
| margin | E2 | 0.208760023 | -0.05 | -0.000144881163 | 0.208615142 | 0 | yes |
| margin | F1 | 0.557504654 | -0.05 | +0.00355068598 | 0.56105534 | 0 | yes |
| margin | F2 | 0.536499143 | -0.1 | -0.000825905572 | 0.535673237 | 0 | yes |

All 12 LL predictions are positive. Margin has four positive and eight negative deltas; all are reported. Predicted objective changes are not measured behavior or official WSP scores. Numerical finite response remains unknown in this audit.

## Original 16-round history

The fixed solver chose round 16; there was no additional solve or amplitude/round search. The history records maximum shortfall, not all 24 individual historical residuals. Those missing vectors remain `null`; the audit did not reconstruct them by rerunning. The accumulated F64 proposal was also not saved, so its geometry is only the solver's reported record, not an independently reconstructed artifact.

| Round | Maximum shortfall | Actual delta L2 | Changed cells | Dual iterations |
|---:|---:|---:|---:|---:|
| 1 | 0.201272095262 | 0.00112955566975 | 30646 | 4 |
| 2 | 0.166149583513 | 0.00169128915495 | 39588 | 4 |
| 3 | 0.135776963281 | 0.00219857206328 | 44895 | 3 |
| 4 | 0.111179322356 | 0.00261852398893 | 48456 | 3 |
| 5 | 0.0918879544672 | 0.00295315274879 | 51041 | 3 |
| 6 | 0.0751041855651 | 0.00324621145963 | 53108 | 3 |
| 7 | 0.0622609006939 | 0.00347276271943 | 54763 | 3 |
| 8 | 0.0519423329349 | 0.00365548637023 | 56012 | 3 |
| 9 | 0.0438954465369 | 0.00380047584325 | 57020 | 3 |
| 10 | 0.0367006963636 | 0.00392992174017 | 57833 | 3 |
| 11 | 0.0299476196175 | 0.0040664370424 | 58511 | 3 |
| 12 | 0.025794078401 | 0.00414418231876 | 59009 | 3 |
| 13 | 0.0215867011791 | 0.00422030235067 | 59427 | 3 |
| 14 | 0.0180735308993 | 0.00428346187928 | 59817 | 3 |
| 15 | 0.0144199154481 | 0.00435042871268 | 60209 | 3 |
| 16 | 0.0123535266982 | 0.00438936817252 | 60473 | 3 |

The original rounding record reports continuous L2 0.00457984606549, relative rounding error 0.636836652781 and continuous/realized cosine 0.789322237389. The stored spec/realized norm/predictions were independently verified; the missing continuous vector was not.

## Validation history and complete records

Initial focused validation had **2 import errors**: the test loader could not resolve a sibling recipe import used only for a three-line assertion helper. The ordinary CLI audit nevertheless completed successfully on that original source. Both failures, exact pre-fix source and successful audit output are preserved. The unnecessary import was removed, using the AST-identical local helper; no numerical logic changed, and the already completed audit was not rerun. Final focused tests: **2 passed in 0.011 s**, Ruff passed. Full CI was not repeated.

This helper is [native_spec_audit.py](../../scripts/experiments/gemma4-purpose/native_spec_audit.py). It adapts the existing F01 independent-word-check method; that tool and the existing roster summarizer require matrix samples and do not support this native-only input. Common solver/editor/sandbox sources were unchanged.

[Complete records](records/2026-11-07-f03-native-spec-audit/) include the fixed plan, original solver report/history, lossless spec, all 421-row audit results, exact command, before/after source, failed/successful test logs and checksums. Source before: `93ccadd4f34ffd5cec06c52eeea5a90ece095f79ee6d3ae151139a3753b4e672`; final source: `f75f9204516133894d0f3ae1b400ae1ed890ada6f05ad17b44fb09db2db68f29`. The result pins the source actually executed, with its exact snapshot preserved after the import cleanup. Audit report SHA: `9f595cdcbc87b97eba388092fdaf4bde902b708a609bd28e6981da91cfe4c159`. No attempt or failure was overwritten.

The box-excess counts were obtained **after** the parent's Opus-review prompt was frozen/sent; they are additional audit evidence, not information that the earlier prompt necessarily included.

## All layers with selected-scope denominators

The original report's layer denominator covers all native weights in that layer and is explicitly labeled. The [additional selected-denominator table](records/2026-11-07-f03-native-spec-audit/layer-selected-denominators.json) also uses only the three eligible families, retaining every eligible coordinate. This avoids hiding a local change behind large ineligible input/pre-FF norms. It is computed from the preserved per-tensor norms without rerunning the audit or reading gradients. Both full-native and selected denominators remain available; final norm has no eligible coordinates and its selected-relative value is unknown/null, not zero.

| Layer | Changed tensors / eligible | Changed cells | Actual L2 | Eligible source L2 | Relative to eligible | Box-exceed cells |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1 / 3 | 1 | 0.00048828125 | 465.186923 | 1.04964526e-06 | 0 |
| 1 | 2 / 3 | 174 | 0.000527256853 | 525.813433 | 1.00274512e-06 | 1 |
| 2 | 2 / 3 | 4628 | 0.000876036293 | 6.45691458 | 0.000135674134 | 780 |
| 3 | 2 / 3 | 5562 | 0.0012090898 | 12.0918117 | 9.99924433e-05 | 3266 |
| 4 | 2 / 3 | 1597 | 0.00145525029 | 7.87563114 | 0.000184778879 | 143 |
| 5 | 2 / 3 | 4278 | 0.000970888699 | 23.6367072 | 4.10754632e-05 | 234 |
| 6 | 2 / 3 | 3852 | 0.000918944261 | 23.2868056 | 3.94620146e-05 | 2231 |
| 7 | 2 / 3 | 4011 | 0.00096831903 | 30.9170767 | 3.13198767e-05 | 245 |
| 8 | 2 / 3 | 3342 | 0.000841075869 | 34.6480683 | 2.42748272e-05 | 2291 |
| 9 | 2 / 3 | 3094 | 0.00068450435 | 31.7498768 | 2.15592758e-05 | 282 |
| 10 | 2 / 3 | 2921 | 0.000496878047 | 33.2363822 | 1.49498235e-05 | 1606 |
| 11 | 2 / 3 | 2673 | 0.000748735773 | 40.0895637 | 1.86765757e-05 | 885 |
| 12 | 2 / 3 | 2829 | 0.00100850958 | 45.6008489 | 2.21160263e-05 | 282 |
| 13 | 2 / 3 | 2501 | 0.000972104561 | 16.5328212 | 5.87984681e-05 | 382 |
| 14 | 2 / 3 | 2782 | 0.00106248361 | 15.8093166 | 6.72061692e-05 | 121 |
| 15 | 2 / 3 | 2744 | 0.00113191781 | 35.5648405 | 3.18268771e-05 | 266 |
| 16 | 2 / 3 | 1193 | 0.00170042725 | 18.5494419 | 9.16699951e-05 | 318 |
| 17 | 2 / 3 | 1771 | 0.000577334614 | 43.4566663 | 1.32852946e-05 | 455 |
| 18 | 2 / 3 | 729 | 0.000920564484 | 23.4501802 | 3.92561795e-05 | 61 |
| 19 | 2 / 3 | 1274 | 0.000217320595 | 36.1570426 | 6.01046377e-06 | 624 |
| 20 | 2 / 3 | 739 | 0.000142068668 | 40.7420496 | 3.48702801e-06 | 216 |
| 21 | 2 / 3 | 48 | 1.80946979e-05 | 34.0763256 | 5.3100496e-07 | 0 |
| 22 | 2 / 3 | 128 | 4.46906071e-05 | 47.3129586 | 9.44574351e-07 | 8 |
| 23 | 2 / 3 | 71 | 6.79187539e-05 | 84.5664958 | 8.03140218e-07 | 2 |
| 24 | 2 / 3 | 29 | 2.25680534e-05 | 69.9843588 | 3.22472818e-07 | 0 |
| 25 | 1 / 3 | 34 | 2.32038976e-05 | 79.6965339 | 2.91153159e-07 | 0 |
| 26 | 1 / 3 | 5 | 1.70598448e-05 | 108.682305 | 1.56969847e-07 | 0 |
| 27 | 1 / 3 | 17 | 3.14567995e-05 | 109.294549 | 2.87816727e-07 | 0 |
| 28 | 1 / 3 | 1 | 1.52587891e-05 | 154.868556 | 9.85273542e-08 | 0 |
| 29 | 1 / 3 | 15 | 3.23687797e-05 | 111.463002 | 2.90399317e-07 | 0 |
| 30 | 2 / 3 | 109 | 0.000119174952 | 94.100778 | 1.26646086e-06 | 44 |
| 31 | 2 / 3 | 23 | 5.46181529e-05 | 82.04132 | 6.65739567e-07 | 10 |
| 32 | 2 / 3 | 91 | 9.82978724e-05 | 83.1779744 | 1.18177766e-06 | 17 |
| 33 | 2 / 3 | 169 | 0.000122427418 | 79.3490184 | 1.5428977e-06 | 33 |
| 34 | 2 / 3 | 165 | 0.000135247024 | 88.5166383 | 1.5279277e-06 | 25 |
| 35 | 2 / 3 | 162 | 0.000171788486 | 118.135103 | 1.45416969e-06 | 47 |
| 36 | 2 / 3 | 244 | 0.000202358814 | 87.7727245 | 2.3054863e-06 | 107 |
| 37 | 2 / 3 | 283 | 0.000229516738 | 86.2586237 | 2.66079759e-06 | 106 |
| 38 | 2 / 3 | 294 | 0.000259399414 | 93.2508338 | 2.78173828e-06 | 116 |
| 39 | 2 / 3 | 323 | 0.00025880971 | 129.266702 | 2.00213748e-06 | 29 |
| 40 | 2 / 3 | 208 | 0.000320230146 | 162.084807 | 1.97569502e-06 | 26 |
| 41 | 2 / 3 | 233 | 0.000266838006 | 143.531281 | 1.85909304e-06 | 48 |
| 42 | 2 / 3 | 299 | 0.000224516891 | 108.346777 | 2.07220646e-06 | 110 |
| 43 | 2 / 3 | 211 | 0.000276269425 | 94.844194 | 2.91287652e-06 | 15 |
| 44 | 2 / 3 | 298 | 0.000155574654 | 84.1972261 | 1.84774085e-06 | 104 |
| 45 | 2 / 3 | 300 | 0.000145734417 | 75.9781686 | 1.9181091e-06 | 36 |
| 46 | 2 / 3 | 333 | 0.000166857579 | 69.7122021 | 2.39352042e-06 | 57 |
| 47 | 2 / 3 | 300 | 0.000151751131 | 73.53182 | 2.06374779e-06 | 57 |
| 48 | 2 / 3 | 332 | 0.000154577634 | 73.6883172 | 2.0977224e-06 | 106 |
| 49 | 2 / 3 | 302 | 0.000239355155 | 72.2233047 | 3.31409863e-06 | 59 |
| 50 | 2 / 3 | 316 | 0.000153159026 | 77.9345162 | 1.96522714e-06 | 151 |
| 51 | 2 / 3 | 333 | 0.000129951905 | 72.9044094 | 1.78249719e-06 | 26 |
| 52 | 2 / 3 | 297 | 0.000210509429 | 100.609843 | 2.09233434e-06 | 191 |
| 53 | 2 / 3 | 317 | 0.000177496187 | 84.6643188 | 2.0964698e-06 | 33 |
| 54 | 2 / 3 | 348 | 0.00015417702 | 64.7674945 | 2.38046911e-06 | 126 |
| 55 | 2 / 3 | 315 | 0.000176200202 | 72.6677134 | 2.42473849e-06 | 39 |
| 56 | 2 / 3 | 309 | 0.000176941927 | 77.9019324 | 2.27134195e-06 | 41 |
| 57 | 2 / 3 | 318 | 0.000230844349 | 97.5660823 | 2.36603073e-06 | 180 |
| 58 | 2 / 3 | 198 | 0.000348455421 | 116.299941 | 2.99617882e-06 | 116 |
| 59 | 0 / 3 | 0 | 0 | 267.39901 | 0 | 0 |
| global | 0 / 0 | 0 | 0 | 0 | null | 0 |
