# F05 proposal, saved weights and actual Opus 5.5 review

This record closes the F05 proposal/save/review phase. The parent is the completed F04 dev30 result: code 72.9365079365, overall 76.9910864328, all 2338 items. F05 has no official capability result in this snapshot. Code above 90, final60/test, final DeepSWE 113 at 32 workers, wiki and conditional publication remain unfulfilled. Deployment and subsequent candidate screening are separate records.

The [fixed pre-observation policy](records/2026-11-07-f05-proposal-save-opus-v1/pre-observation-policy.json) uses all 19 authored rows from F04 (16 numerical passes, A1/F2/J1 failures), two teacher-forced objectives and 38 simultaneous constraints. All 361 norm vectors are eligible; 60 layer scalars are frozen. Failure LL targets require +0.1; previously successful LL rows allow -0.01. A1/J1 margin targets require at least +0.2 and a final margin of 0.1; all other margins allow -0.02. Relative metric floor is 0.0001; the continuous box is 1% relative plus 0.000001 absolute; all 16 BF16 rounds are retained. F2/J1 share their prompt/correct answer and are not independent behaviors. The original F2 foil is preserved. Nine transfer cases remain unused.

The CPU solver ran once: preview 0.297543 s, solve process 26.341675 s (internal calculation 26.018707 s), owner 206.664382 s including waiting. It selected round 15: 26,417 saved BF16 cells across 361 tensors, delta L2 52.2944053830, relative metric norm 0.9366422579. It meets 36 of 38 linear predictions; A1/J1 margin shortfalls are 0.000111579799 and 0.000051834118. These are proxy predictions, not execution gains. The [independent static audit](2026-11-07-f05-static-native-audit.md) gives complete denominators, all 38 predictions, 653 continuous-box overruns and the rounding discrepancy; that evidence is not duplicated here.

| Round | Changed cells | Actual delta L2 | Maximum predicted shortfall |
|---:|---:|---:|---:|
| 1 | 3895 | 13.7477623 | 0.212770917 |
| 2 | 9765 | 28.2642844 | 0.134517869 |
| 3 | 15504 | 38.4755211 | 0.0809938001 |
| 4 | 19903 | 44.6923903 | 0.0458435915 |
| 5 | 22657 | 47.6631157 | 0.0259229871 |
| 6 | 24282 | 49.6201562 | 0.0143262241 |
| 7 | 25221 | 50.8439076 | 0.00775598227 |
| 8 | 25796 | 51.7203387 | 0.00397058373 |
| 9 | 26087 | 52.0269153 | 0.00209500165 |
| 10 | 26243 | 52.0454652 | 0.00111829911 |
| 11 | 26328 | 52.2816216 | 0.000784938245 |
| 12 | 26367 | 52.2853111 | 0.000509566591 |
| 13 | 26390 | 52.2859916 | 0.000492364597 |
| 14 | 26405 | 52.2889958 | 0.000392668378 |
| 15 | 26417 | 52.2944054 | 0.000111579799 |
| 16 | 26421 | 52.2944439 | 0.000215352091 |

The [materialization completion](records/2026-11-07-f05-proposal-save-opus-v1/materialization/completion.json) records one preview/create/verify sequence: 3.551943 / 3.900203 / 4.326436 s, total 13.938368 s including checks. Faststore preserves logical F04 ancestry and its unchanged shared payload. Eight private files received fresh hashes; the shared file uses its prior full hash plus exact inode/stat/link proof. Shared payload bytes read were zero; this is explicitly not a fresh whole-model hash. Verification confirms 26,417 changed cells / 361 tensors and original-byte restoration.

| Saved evidence | SHA256 |
|---|---|
| Original proposal plan | f0ca2df4c3a73bd1f032e9602aed42733afc2b9fa919324fc81feef648569ffa |
| Original specification | 48e095d3de304767410ac0956e390ad6db4d211457683dabdc779a4f6bb48df3 |
| Complete solver report | 4d2e57e4c92dff749e0da7b052cbadd435bc4486145ebda5d476c12cefdad793 |
| Saved manifest | 09ea569a08f138efd438055d7149f0697dc1d6edd08332f2b6c7c09b1da7c24c |
| Verify output | 1e9150f21a53af2e96d82c65297a9ce0dffb71d21ea59ea03d3d02a45ca4c427 |
| Candidate checkpoint tree | fcc4681343c1a0c338157bee767df2024f37290469e55dbc417723717f4ac176 |

The [anonymous prompt](records/2026-11-07-f05-proposal-save-opus-v1/anonymous-opus55-prompt-v1.txt), SHA 57c6e4de0b23e1dc69aa01398a4ab16feed4d13f70acb46ec272059a3d61dcfc, was previewed before each launch. Attempt v1 failed before starting the CLI with FileNotFoundError (noninteractive PATH), 0.000764 s and zero model calls. Its prompt, launch, receipt and logs remain. Attempt v2 used the absolute CLI in the existing authenticated environment; no environment or credentials were recorded. Actual response model and canonical usage are both claude-opus-5-5, no fallback/tool use, strict JSON with label A, exit 0: 52.396717 s and $0.120192. Receipt SHA c19a3f3ae455c537c0764ce9348615ae918671e98a5c8bf1e0287f647076b19e. The prompt used a rounded save time; the actual completion above is authoritative.

[The complete public review](records/2026-11-07-f05-proposal-save-opus-v1/review-public-response.json) conditionally supports a full canonical trial, but estimates weak evidence for native coding benefit. It recommends one candidate-only native 19-row generation/execution screen first, given prior proxy/native mismatch, the larger all-layer change, BF16 box overruns, and duplicate/padding sensitivity. It treats tiny missed bounds as disclosed limitations rather than a reason to re-solve. Its proposed screen is advice, not a benchmark result or an adoption decision.

Root recorded the [screen decision before any candidate forward](records/2026-11-07-f05-proposal-save-opus-v1/candidate-native-screen-decision.json). It accepts the candidate-only screen and avoids any repeated parent measurement/replay. Go requires at least one of A1/F2/J1 fully passing, at most one loss among the previously passing 16, at least 16/19 fully passing and 264/304 fixtures, all 19 complete with zero unknown/API error/truncation. Allowing one loss instead of Opus's zero-regression preference implements the user's measured tradeoff instruction. On failure, F05 stays officially unscored and F04 remains parent; on success, the unchanged full 2338-item canonical dev30 is next. No candidate has been adopted by this packaging work.

[Records and checksums](records/2026-11-07-f05-proposal-save-opus-v1/SHA256SUMS) preserve every solver round, exact full specification/report/manifest (lossless gzip where large), all original source/input hashes, source snapshots, both review attempts, public response, operator decision, and one-off launch/packaging sources. Large gradient arrays remain at their existing locations with their original source hashes; they were neither copied nor rehashed. No checkpoint payload, model request, service, solver or live source was changed during packaging. This snapshot excludes ongoing deployment and future screen results.
