# Scientific calibration and runtime pilot (2026-11-06)

This extends the code90 cycle. These observations are calibration/runtime evidence, not a new WSP score.

## Frozen starting point

The previous candidate0397 remains the serving control. Development code90 starts at70.0753968 on90items; the previously reported known64 code score82.547619 is a different cohort. The primary target remains unrounded official source-mean code>90 on that known64 reference, with separate transfer results.

The author-owned science corpus has16training cases and8family-separated transfer cases. Trusted correct programs passed192variable-input checks; every foil failed at least one check. Training cases occupy141–255tokens under the shared native-thinking chat/public-answer framing. Their gradients are calibration, not an estimate of benchmark success.

## Runtime attempts

1. Host Python attempt stopped before model load because accelerate was absent. No packages were changed. Subsequent attempts reuse the already-pinned vLLM image af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe as a non-root offline HF runtime.
2. Matrix pilot loaded the original checkpoint but stopped before gradients: Transformers5.20.2 exposes full-attention geometry through per_layer_config rather than legacy num_global_key_value_heads. Actual full layers have4KV heads of width512, sliding layers16KV heads of width256. Sampling must resolve actual layer configuration and check projection shapes.
3. Native16case collection exceeded memory on its first forward with the previous1600paired-token packing:91.50GiB allocated and1.21GiB reserved. Existing unrelated embedding work was retained. A one-case microbatch cap preserves fold weighting while reducing activation memory; no truncation or hidden checkpoint recomputation is introduced.

Only serving replica0 was stopped after all four replicas reported zero running/waiting requests. Its GPU0/1pair is used for collection; the original and0397 checkpoints are mounted read-only. Collection containers use UID1003, no network and no new model downloads. WSP is not run through the reduced replica group.

## Independent generated-code baseline

The0397 control generated all16training and8transfer public answers using native thinking ON, checkpoint generation defaults, seed0 and an8192output-token cap. All24finished with stop, with no API errors. Output-token counters are15921train and13864transfer, including inference-token usage; hidden reasoning text was not persisted. Execution verdicts are pending. These generated results are neither added to WSP nor used as benchmark reference answers.

Evidence resides under analysis/code90-resume-20261106: immutable source/case hashes, failed-attempt receipts, actual-container-text-config.json, source/gpu handoff, science-gpu-runs and science-generation-h0397-r2. Failed attempts remain inspectable rather than being replaced by successful reruns.

## Completed measurements and first proposals

The corrected matrix pilot completed4folds in1.93seconds after model loading, using67.61GiB peak allocated and69.17GiB reserved. Eight zero-additive probe coordinates agreed with the factored derivatives to maximum absolute error2.73e-12. Full16case matrix collection took5.62seconds, peak70.99GiB allocated. It covers209,920sampled coordinates in410projections, not all model elements. The native16case collection took7.50seconds, peak74.90GiB allocated/78.96GiB reserved. Both report unchanged source values.

The first six local-gradient plans were numerical-priority, contract-priority and balanced objectives, each in native-only and matrix-only scope. With radius max(.025*abs(W),1e-6), all plans retained unmet linear targets. Native emitted36,532/49,053/21,392changed values with absolute delta L2 .002714/.003736/.001480, respectively. The three matrix plans needed199,565/205,055/202,154changes, exceeding the then65,536tool cap; no matrix specs were emitted. Their predicted responses were only approximately.001–.003nats against.02/.05targets. These are recorded failures, not measured coding gains.

A second matrix proposal stage keeps purpose targets and sampled geometry fixed, uses radius max(.5*abs(W),1e-5), and expands the tool budget to262,144total matrix cells while retaining512per matrix and the replay1million total-coordinate bound. The radius is a limit, not the achieved edit. Increasing this tool limit does not change any official benchmark grading or question set.

## Generated execution baseline

After fixing the sandbox image entrypoint from python to python3, trusted correct/foil fixtures were checked in actual restricted containers. The original all-unknown attempt is retained as an infrastructure failure. Re-evaluating the same24public answers produced15/16training and7/8transfer passes, unknown0, with8variable-input trials per case.

The training failure is batched linear solve: generated np.linalg.solve(a,b) does not implement the declared batch-vector contract under the sandbox NumPy version. The transfer failure is Kalman covariance update: the response assumes a two-dimensional observation vector while the fixture supplies a vector. The original public prompt does not explicitly state that vector shape, so that failure also has task ambiguity and is weak evidence of a model defect. The existing prompt, tests and score are preserved. The short training kernels are already mostly solved; their margin gains cannot stand in for difficult scientific benchmark performance.

Candidate finite measurement uses120authored cases in34folds covering17purposes: scientific numerics/contracts plus code, tools, instruction, general knowledge, reasoning, lookup, Japanese, implementation, workflow, tool call/no-call/arguments, algorithms, reference tracking and fact retention. It is a bounded diagnostic before generated execution and official WSP, not an adoption rule by itself.

## Fixed realized proposals before official scoring

| Proposal | Changed values | Absolute delta L2 |
|---|---:|---:|
| science-matrix-numerics-v2 | 192,623 | 1.037222928 |
| science-matrix-contracts-v2 | 202,196 | 1.611345760 |
| science-matrix-balanced-v2 | 190,994 | 0.803666688 |
| science-native-numerics-v1 | 36,532 | 0.002714207 |
| science-native-contracts-v1 | 49,053 | 0.003735684 |
| science-native-balanced-v1 | 21,392 | 0.001479884 |
| science-joint-matrix-numerics-native-contracts-v2 | 241,676 | 1.037229655 |

All seven exact specs were re-read against original source coordinates before replay. No proposal is yet an adopted checkpoint. The one-load replay completed all eight sets including0397 in86.64seconds, with original/restored margin, correct-answer NLL and sampled endpoint KL differences exactly zero. KL samples only the first/last correct-answer token positions, so small endpoint KL does not prove that the entire answer distribution is unchanged.

The code90 science/tool changes passed419CPU tests (377pass,42optional skips), whole-tree Ruff and mypy73source files. Actual GPU evidence and restricted Docker fixture validation are reported separately from this CPU suite.
