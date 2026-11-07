# F04 native19 collection and local dependency observations

All 19 singleton cases completed once for correct-logprob and once for margin on F04. The owner completed in 279.955 seconds, restored the same ordinary gemma-2 CID, verified the other 26 snapshots, and released its locks. Failures: 0. The final target was healthy with running/waiting queues 0; both collector clients and owned containers had finished. This is teacher-forced sensitivity measurement, not another WSP or native-generation baseline.

| Measurement | Folds | Forward calls | Collector seconds | GPU peak allocated GiB | CPU copied bytes, cumulative |
|---|---:|---:|---:|---:|---:|
| correct-logprob | 19 | 19 | 43.970 | 62.256745 | 100,941,375,488 |
| margin | 19 | 38 | 73.084 | 62.256745 | 201,885,896,704 |

Collector timing starts after model loading and frame preparation, so it excludes those phases and Docker startup; complete Docker jobs took 67.409 and 94.593 seconds. Owner elapsed also includes stop and same-CID health restoration. CPU copy totals count repeated save/unpack events, not resident host memory. Image af9, GPU4 only, text SDPA, serial one-answer VJP, maximum frame 1024, FP32 activation offload, 88 GiB allocator cap and 128 GiB host cap match the approved plan. No fallback or retry was used.

All 421 native tensors/1,331,516 coordinates were observed and their source values remained unchanged. Analysis retains every measured row. The predeclared eligible set removes 60 whole-layer scalars, leaving 361 vectors/1,331,456 coordinates. For both complete and eligible supports, raw and relative gradients (`g * max(abs(weight), 1e-4)`) have full 38×38 cosine tables, eight-family energy tables and explicit per-row norms. None of these norms assigns an ability to a weight family.

| Case | Native execution | Correct mean logprob | Correct-minus-foil margin |
|---|---|---:|---:|
| behavior-A1 | fail | -1.375041604 | -0.253932714 |
| behavior-A2 | pass | -1.628711462 | 0.223465681 |
| behavior-B1 | pass | -1.738438964 | 0.069040060 |
| behavior-B2 | pass | -1.105060935 | 0.062152982 |
| behavior-C1 | pass | -2.408848286 | 0.632292986 |
| behavior-C2 | pass | -1.265420198 | 0.403710842 |
| behavior-D1 | pass | -1.511942863 | 0.285267711 |
| behavior-D2 | pass | -1.091444731 | 0.450174928 |
| behavior-E1 | pass | -1.072220206 | 0.037589431 |
| behavior-E2 | pass | -1.836114168 | 0.213594198 |
| behavior-F1 | pass | -2.045177698 | 0.564146280 |
| behavior-F2 | fail | -1.811716199 | 0.563709140 |
| behavior-G1 | pass | -1.987506866 | 0.683558226 |
| behavior-G2 | pass | -1.781924605 | 0.435414910 |
| behavior-H1 | pass | -1.740635872 | 1.495291471 |
| behavior-H2 | pass | -2.248062611 | 1.176909447 |
| behavior-I1 | pass | -1.786936402 | 0.103530765 |
| behavior-I2 | pass | -1.783743143 | 0.060129404 |
| behavior-J1 | fail | -1.819816947 | -0.144932747 |

Native execution labels come from the separately completed 19-case CPU report (16 pass, 3 fail, 264/304 fixtures). A1 failed 0/16; F2 and J1 failed 4/16 each. G/H/I all passed. No missing score was filled with zero.

F2 and J1 share their public prefix and correct answer, but have different foils: 49 versus 43 foil tokens, with prefix 103 and correct answer 45. Their maximum padded widths are 152 and 148. Their correct-LL values differ by −0.008100748 (J1−F2); the gradients differ at 1,311,419 coordinates, raw difference L2 16.050859, and eligible-relative cosine 0.985021393. Thus their semantic correct-LL targets are duplicated, but the observed numerical rows are not identical. Different BF16 execution shapes are a possible contributor; this analysis does not prove the cause.

Their margin gradients have eligible-relative cosine 0.178451908. F2 prefers the correct answer to the older zero-removal foil (+0.563709), while J1 prefers the actual off-by-one foil (−0.144933). This is why the foil distinction matters; it is not evidence that either local margin predicts a WSP gain.

| Failure margin vs all 16 successful margins, eligible relative metric | Minimum cosine | Maximum cosine |
|---|---:|---:|
| behavior-A1 | -0.064307909 | 0.085300819 |
| behavior-F2 | -0.054741414 | 0.058661814 |
| behavior-J1 | -0.061154924 | 0.102208508 |

These mixed local signs justify retaining the success constraints alongside failure objectives. They do not establish causal ability dependencies or finite BF16 improvement. All 38 normalized Gram eigenvalues and a stated 1e−8 rank threshold are retained; numerical rank 38 is compatible with near-duplicate rows. No solver, candidate or additional model call was made by this analysis.

The first CPU analysis entry failed before any gradient read because the collector-only snapshot did not contain joint.py (0.122 seconds). Its exact source and failure are retained. The second entry used existing repository joint.read_collection, pinned every imported local source and input, and completed the single data aggregation in 3.628 seconds (process 3.788 seconds), with one CPU thread, nice 15 and idle I/O priority.

Completion SHA: `b92b26603c4040139ed3043e45c3f740173a8664d148045434fecf5ba71864d3`. Analysis report SHA: `d1126f39833bc48709e9e5afee76262614335fe12de78210fec6007ee29ac6ea`. [Commands, all per-case values and cosine tables, source pins, raw collector metadata, logs and both CPU attempts](records/2026-11-07-f04-native19-sensitivity/retention-index.json) are in the PR. All gradient/source NPZ files remain at their original NVMe paths with hashes; the PR stores the complete numeric summaries instead of duplicating 118 MB of raw arrays.
