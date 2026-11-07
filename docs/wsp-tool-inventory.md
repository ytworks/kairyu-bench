# WSP tool inventory and experiment provenance

This inventory maps reusable tools created during the Gemma experiment to their
reviewable implementation. Candidate weights, generated specs, benchmark data,
private runtime configuration and model reasoning are separate artifacts.

| Task | Included implementation | Evidence and boundary |
|---|---|---|
| Checkpoint identity, streamed statistics, differences and explicit groups | [weight reader and CLI](../src/kairyu_bench/wsp/weights/) | CPU reads; distinguish full counts, samples and denominators |
| Focused all-tensor BF16 delta footprints | [delta-only recipe](../scripts/experiments/checkpoint-delta-footprint.py), [guide](checkpoint-delta-footprint.md) | Exact bit/finite-numeric counts and delta L2, bounded one-pair reads; prior full SHA plus stat continuity, explicit nonfinite semantics; comprehensive statistics remain in the existing weights CLI |
| Selected tensor axes, structural alignment and low-rank statistics | [matrix diagnostics](../src/kairyu_bench/wsp/weights/matrix.py) | Geometry is not a causal assignment of coding ability |
| Native scalar/vector and sparse matrix checkpoint edits | [editor](../src/kairyu_bench/wsp/weights/editor.py), [editing guide](wsp-weight-editing.md) | Explicit application; original-byte restoration and complete file verification |
| Verified shard reuse, frozen file fingerprints and followup receipts | [shard pool](../src/kairyu_bench/wsp/weights/shard_pool.py), [receipts](../src/kairyu_bench/wsp/weights/receipts.py) | Explicit inputs; edited shards remain private; receipt checks do not authorize an automatic benchmark skip |
| Exact BF16 LoRA merge, joint native scale, and adapter diagnostics | [Gemma LoRA recipes](../scripts/experiments/gemma4_lora/README.md) | Pinned 410-text-projection recipe; compiled kernel source and tests included, binary excluded |
| Purpose gradients across native gains, virtual branch directions and sampled matrix coordinates | [purpose collection recipes](../scripts/experiments/gemma4-purpose/README.md) | Cases and frames are explicit; new code validated on CPU fixtures, not claimed as a new real-model measurement |
| Authored scientific-library training and unused transfer families | [train builder/validator](../scripts/experiments/gemma4-purpose/library_train_cases.py), [transfer builder/validator](../scripts/experiments/gemma4-purpose/library_transfer_cases.py), [training guide](wsp-library-science-training.md), [transfer guide](wsp-library-transfer-cases.md) | Sixteen new training variants from eight families; append-only inherited136 preservation; distinct transfer8 stays out of gradients; SciPy/SymPy CPU references, foils and execution invariants are authored calibration, not official scores |
| Offline scientific-library CPU executor staging | [staging helper](../scripts/experiments/stage_science_library_executor.py), [Dockerfile](../scripts/experiments/science_library_executor.Dockerfile), [guide](wsp-library-transfer-cases.md) | Copy an installed SciPy distribution only after exact Python ABI/NumPy/SymPy compatibility; source/copy hashes and fresh disjoint output; no RUN/download hook, serving change or model call; actual sandbox compatibility is separately measured |
| Offline full16 helper-stage diagnostics from the complete72 response envelope | [adapter](../scripts/experiments/gemma4-purpose/helper_stage_offline.py), [guide](wsp-helper-stage-offline.md) | All16 exact authored stages; fixed public helper/support only; existing score/Docker worker; complete72 input/response SHA,56 not targeted, missing/error unknown,3 dtype caveats; no model request or generated-code host execution |
| Joint science/code sampled-matrix groups | [group-plan builder](../scripts/experiments/gemma4-purpose/prepare_science_code_groups.py), [guide](wsp-shared-science-code-groups.md) | Exact matrix-name eligibility is shared by target normalization and solving; current-parent/fold/source pins, native frozen, existing optimizer; group alignment is not causal attribution or a capability gain |
| Joint edits from saved local gradients | [local-G solver](../scripts/experiments/gemma4-purpose/joint.py) | Explicit purposes, fold constraints, permitted losses and cost metric; BF16 predictions require finite measurement |
| Coupled native correct-logprob and margin proposals | [native coupled recipe](../scripts/experiments/gemma4-purpose/propose_native_objectives.py), [guide](wsp-native-coupled-objectives.md) | Same complete incumbent identity/cohort/serial layout, explicit two-objective bounds, existing BF16 solver, actual changed-coordinate cap; predictions are not capability |
| One continuous calibration and a derived matrix-only BF16 box | [derived-radius recipe](../scripts/experiments/gemma4-purpose/derive_matrix_radius.py), [guide](wsp-derived-matrix-radius.md) | Two explicit bounded CPU stages, same fixed observations/bounds, source-native and QKV exclusions, realized-box audit; no model call or local-response guarantee |
| Current-parent normalized coding targets | [plan preparation](../scripts/experiments/gemma4-purpose/prepare_normalized_objective_plan.py), [guide](wsp-normalized-objective-targets.md) | Matched row identities, selected-matrix dual norms and headroom; no solver/model call, global BF16 cap remains an explicit owner postcheck |
| Load-time sparse BF16 overlay (CPU-tested, rollout pending) | [separate vLLM plugin](../scripts/experiments/vllm_sparse_overlay), [guide](wsp-vllm-sparse-overlay.md) | Read-only physical parent plus exact sparse logical identity; per-tensor clone, post-TP all-rank selected-word proof, pinned unchanged MTP; no enabled serving path |
| Matched cost metrics on saved local gradients | [cost comparison](../scripts/experiments/gemma4-purpose/compare_costs.py) | Identical constraints, explicit near-zero floors, output-name collision checks; no new forward |
| Saved finite-response analysis and quantized joint-set proposals | [finite-response guide](wsp-finite-response.md) | Endpoint secants, active-cone projection and Pareto reports are predictions, not scores |
| Original-base composition | [composition](../src/kairyu_bench/wsp/weights/composition.py) | Report overlap, cancellation and actual BF16 deltas; no assumed additive benefit |
| Exact composition of verified saved checkpoints | [byte-splice composer](../scripts/experiments/compose-verified-checkpoints.py), [guide](wsp-checkpoint-composition.md) | Disjoint full-tensor bytes, full/partial virtual reversion, fresh output hashes; no GEMM or rerounding and no assumed functional additivity |
| Carry a complete sparse edit onto the current parent | [rebase helper](../scripts/experiments/gemma4-purpose/rebase_sparse_composition.py), [guide](wsp-cumulative-sparse-composition.md) | Exact selected BF16 source-word match or whole-edit rejection, retained unrelated increments, metadata-only preview and explicit relative cost; interaction unmeasured |
| Fixed native/sparse-matrix RAM replay | [finite replay](../src/kairyu_bench/wsp/weights/finite_replay.py) | Original → up to eight fixed native/matrix sets → restored original; explicit serial settings and selected-BF16 word readback, no persistent weight writes or automatic adoption |
| Live activation collection and benchmark/request joins | [observation package](../src/kairyu_bench/wsp/observe/), [guide](wsp-weight-observation.md) | TP, call/trial identity, missing data, drop counts and incomplete capture status remain visible |
| Large offline activation overview | [report generator](../src/kairyu_bench/wsp/observe/report.py) | Task-wise compressed loading, bounded cache, explicit errors; all tasks retained |
| Observer deployment/restore and an isolated trace gateway | [deployment guide](wsp-observation-deployment.md) | Explicit apply and ownership-bound rollback; supported example remains responsible for model serving |
| Anonymous method review and reserve metadata audit | [review tool](../scripts/claude-method-review.py), [reserve audit](../scripts/wsp-reserve-audit.py) | Review transmission is explicit and hash-pinned; reserve audit does not inspect answers |
| Explicit task-image cache preparation and verification | [prefetch tool](../scripts/prefetch-task-images.py), [guide](task-image-prefetch.md) | Preview first; pinned daemon and registry digests; bounded retries and disk checks; no container execution |
| Recorded operational item exclusions | [WSP CLI](../src/kairyu_bench/wsp/cli.py), [guide](wsp.md) | Explicit selected IDs recorded in run metadata; manifest and scoring rules unchanged; compare common scored items with disclosed denominator |
| Baseline observation analysis | [experiment analysis recipe](experiments/2026-11-06-observed-gemma-analysis.py) | One-pass read of retained capture; associations are not edit directions |
| Repeated WSP gain/loss ranges and DeepSWE trial usage | [evidence recipes](experiments/scripts/README.md) | Official scoring is reused; fractional changes, retry UUIDs, steps and missing usage are preserved; opt-in bounded first-info metadata adds agent-recorded API counters separately from steps |
| Serving counter baselines and deltas | [tool](../scripts/experiments/serving-counters.py), [guide](wsp-serving-counters.md) | Whitelisted process counters; resets/missing series explicit; no per-item, retry or thought-token attribution |

The source migration also covers the reusable parts of the original
`edit_checkpoint`, dense/native LoRA recipes and BF16 kernel, purpose-dependency
collectors, joint solver, verified shard pool, restoration/provenance helpers,
screen range/repeat reports and trial-usage summarizer. Reusable functions are
consolidated where the new reader or editor already provides the operation.
The task-local source inventory retains source hashes for that mapping.

Machine-specific launch queues, one-time mount repair, verifier artifact
recovery, retirement commands, deployment snapshots and publication scripts
are operational records. Their methods, failures and safeguards are recorded
in the [first experiment](experiments/2026-11-04-gemma4-weight-editing.md) and
[resumed experiment](experiments/2026-11-06-observed-gemma-baseline.md),
[cost comparison and generated outcomes](experiments/2026-11-06-local-gradient-costs.md),
[composition method review](experiments/anonymous-composition-review-2026-11-06.md),
and [measured composition and sparse-matrix response](experiments/2026-11-06-composed-and-matrix-finite.md).
The [frozen confirmation record](experiments/2026-11-06-frozen-confirmation.md) preserves the common-item comparison, operational exclusion, and environment limits.
They do not define an additional portable tool API. Serving changes use the
existing Gemma example's supported control command; the benchmark repository
does not gain a second serving implementation.

Implementation identity changes when a task-local recipe is moved or hardened.
Preserve the original source hashes and receipts for historical measurements.
CPU fixture validation of the new entrypoints is distinct from the earlier
real-model runs, and the inventory does not claim those runs used later code.

The [migration ledger](experiments/reusable-tool-inventory.md) records the original operation groups and their closed or method-only disposition.

The opt-in sampled-matrix editor and finite replay support 512 coordinates per matrix and 262,144 matrix coordinates per candidate. RAM replay retains its one-million total edited-coordinate limit. This storage/validation bound is not a scientific edit-size threshold.

## Science calibration and candidate diagnostics

The [science recipe guide](../scripts/experiments/gemma4-purpose/README.md) includes executable author-owned train/transfer cases, preview-first public-answer generation, and restricted CPU-container execution. Generation records the declared checkpoint and decoding frame, persists public answers and numeric usage only, and does not establish serving identity by itself. The sandbox keeps expected values outside the generated-code process and reports infrastructure failures as unknown. Neither result is a WSP score. The [gateway request compatibility fix](experiments/2026-11-07-science-generator-gateway-compatibility.md) adds explicit server-default thinking control while preserving the direct-vLLM request default; thinking attestation remains external. HTTP failures retain numeric status without bodies and remain unknown in execution reports.

The same guide documents current/legacy attention geometry, one-case gradient accumulation and native/matrix joint proposals. [Stored donor alignment](../scripts/experiments/gemma4_lora/README.md) compares saved BF16 bytes with sampled scientific derivatives; it cannot infer the full donor direction from those cells. See the [code90 protocol](experiments/2026-11-06-code90-plan.md), [GPU pilot and actual failures](experiments/2026-11-06-science-matrix-pilot.md), and [candidate-method review](experiments/anonymous-science-candidate-review-2026-11-06.md).

The [J final comparison](experiments/2026-11-06-weighted-reasoning-c1-review.md) separates saved calls from the source-defined startup probe and process counters. The [fixed B+H composition plan](experiments/2026-11-06-native-coherent-splice-k.md) records contemporaneous four-arm comparisons; its implementation proof does not establish a score gain.

The [lossless text adapter extractor](wsp-text-adapter-extraction.md) explicitly partitions text and vision factors, preserves raw F32/BF16 bytes and unchanged config, and distinguishes original repository payloads from derived text-only exports. It creates no checkpoint or benchmark call.

The [K1 exported-text trial record](experiments/2026-11-06-k1-exported-text-c1-review.md) records the actual lossless F32 selection, 499-test CPU validation and fixed-unit cross-base recipe; checkpoint quality and official scores remain separate outcomes.

The [paired score-only contrast tool](wsp-paired-score-contrast.md) adds explicitly weighted multi-arm contrasts with shared source-stratified row draws, preserving covariance. It reads SHA-pinned numeric exports only and does not alter official scoring or declare adoption. The subsequent full CPU check passed 503 collected tests (451 pass, 52 optional skips), Ruff and mypy73 with unchanged source hashes; the earlier 499-test extraction result remains its separate historical validation.

The focused BF16 footprint tool passed three behavior fixtures and an actual 1,188-tensor scan in 83.388 seconds (different scope from comprehensive statistics; official-run overlap disclosed). Its [guide](checkpoint-delta-footprint.md#actual-public-base-validation-2026-11-06) records the prior-hash/stat proof, imported-reader pin and nonfinite semantics. The subsequent complete CPU check passed 506 collected tests (454 pass, 52 optional skips), Ruff and mypy73 with unchanged source hashes; all earlier validation counts remain historical records.

The [native faststore](wsp-native-faststore.md) adds an opt-in byte-preserving repack and fixed prelinked slots for cumulative scalar/vector edits. Shared payload links are created before freezing and never changed during a generation; private native files receive fresh byte verification while bulk shards retain explicitly labeled prior-hash/inode/stat proof. Existing editor/pool defaults are unchanged. The new two-generation fixture and complete 510-test CPU suite (458 pass, 52 optional skips), Ruff/mypy74/data/shell checks passed with unchanged source hashes; real checkpoint preparation and timing remain pending.

The purpose matrix collector's optional correct-logprob/serial/coherent-triplet
mode and the coupled adapter's explicit v2 native-plus-O/MLP plan are documented
in [the coupled-objective guide](wsp-native-coupled-objectives.md#optional-native-plus-sampled-matrix-plan).
They reuse the existing objective, optimizer, BF16 editor and sparse finite
replay; Q/K/V are observed but excluded from v2 edits. Legacy defaults remain.

The same v2 coupled adapter accepts explicit matrix-only block eligibility:
native columns are excluded before solving, with unchanged default behavior
and existing saved-output byte verification. This is an opt-in hypothesis,
not an automatic additional search.

The combined library/group-tool CPU validation passed 518 collected tests (463 passed, 55 optional skips), whole-tree Ruff, mypy74, data checks and 31 shell syntax checks, with unchanged source hashes. The managed suite lacks the optional scientific libraries; the train/transfer reference and foil E2E checks also passed separately in the recorded SciPy/SymPy environment, and the offline executor passed its actual CPU sandbox checks. Receipt: A/cpu-integration-library-science-v1/ci-results.json, SHA256 94822c7a5dbb131df8e1949235d5604f6a3b6a71d29b774327a43ed80ab488ac. CPU validation overlapped the separately owned expanded-cohort collection; no speed comparison is claimed.

## Bounded GPU evaluation owner recipes

The [owner guide](wsp-bounded-gpu-evaluation-owners.md) covers two explicit-plan recipes: [matrix-overlay transfer screening](experiments/scripts/screen-matrix-overlays.py) and [a TP2×DP2 full14 pair](experiments/scripts/full14-tp2-dp2-pair.py). The published screen is a formatting-only copy of the exercised owner, with AST equivalence after splitting imports: six candidates plus repeated control, 64 ordinary-generation requests, all original containers restored. No candidate exceeded its paired control; these are authored execution outcomes, not official scores. The DP2 recipe has syntax/inert-preview and source-review evidence only; its GPU topology, smoke and full14 execution remain unvalidated. The port adds no solver, sampler change or automatic adoption rule. These publication checks are separate from the preceding 518-test CPU suite.

The [official overlay screen](wsp-overlay-official-screen.md) reuses the same lifecycle with the immutable official code90 runner: up to two rounds of empty-parent control plus three logical overlays, identical request observation, source/OCI capture and numeric paired evidence. The control server is reused but its official run is fresh each round. Physical saved-weight confirmation remains required before adoption. The first AN–AS execution completed all eight original reports and same-CID restoration, with the artifact-persistence correction and logical-overlay limits disclosed in the guide.

The functional-target extension of the [science/code group builder](wsp-shared-science-code-groups.md)
keeps requested LL-headroom and margin gains fixed across support and cost
choices. [run_derived_matrix_batch.py](../scripts/experiments/gemma4-purpose/run_derived_matrix_batch.py)
runs its pinned reference plans through the existing analytic box derivation,
BF16 proposal and exact raw-cost postcheck with two bounded CPU workers.
Per-method failures are retained without retries. It does not run a model or
adopt weights. The existing builder and derived-box E2Es cover the new
Euclidean/exact-mask/raw-cap paths through saved checkpoint bytes.

The [authored finite owner](wsp-authored-finite-owner.md) publishes the executed six-candidate AN–AS lifecycle as a host-bound, AST-equivalent formatted recipe. It preserves one parent load, selected-BF16 apply/restore proof, bounded cleanup and same-CID restoration; its finite gains are distinct from the subsequent exploratory official overlay scores.

The [sparse delta geometry guide](wsp-sparse-delta-geometry.md) compares explicit same-parent BF16 changes using declared numeric costs. Overlapping absolute set-values are not additive composition; geometry does not establish functional independence or authorize edits.

The [saved-finalist confirmation guide](wsp-saved-finalist-confirmation.md) documents the reviewed physical save/verify/deploy and matched ordinary dev90 recipes. Exploratory overlay gains do not skip saved-byte verification or fresh ordinary-serving comparison; source parent and protected service identities remain explicit.

- [Ordinary full14 finalist comparison](wsp-ordinary-full14-finalists.md): prepared ordinary TP2 fresh-control comparison for one or two saved finalists; fixed778 cohort, symmetric numeric observers and writable-results mount proof. Not executed.

[Helper-chain curriculum](wsp-helper-chain-curriculum.md), [saved-parent observation owner](wsp-observed-parent-purpose.md), and [disjoint-purpose collection composition](wsp-purpose-collection-composition.md) preserve the accepted parent while adding only new authored-purpose observations. The CPU composer keeps the full cohort and explicit selected optimization views with exact copied gradients and original segment provenance; it adds no model run or solver.

Combined validation: 520 collected tests (464 passed,56 optional skips), whole-tree Ruff, mypy74, data checks and31 shell syntax checks passed with source_changed[]. Receipt A/cpu-integration-helper-composition-v1/ci-results.json SHA256 a2475dc95fcb9473b060f8b6a262d37ad31a61a9a5ef74a5f326bdd2ad760bd6. Compared with the previous518-test snapshot, the already-published output-mount regression and the necessary helper-chain E2E add one method each; composer and LL-cap behavior extend existing end-to-end methods. The helper scientific-library E2E also passed separately in its recorded actual environment.

The [helper-capped functional batch](experiments/2026-11-07-helper-capped-functional-targets.md) retains112-case fit and160-case diagnostic identities, an explicit LL cap, and versioned Q/K compatibility without rewriting historical runtime paths. Q/K load-time overlay support separately proves every fused destination, including tied K-to-V copies; direct V/native edits remain excluded. CPU image proof is distinct from an actual engine rollout.

[Opt-in long-context authored observations](wsp-long-context-observation.md) retain legacy frame/backend defaults while adding complete 1024/2048-token validation, text-only SDPA dispatch and an optional CUDA allocator cap. Shared policy identity rejects incompatible collector/composer mixtures. CPU contracts and a complete 120-case matrix-only 31B observation are recorded; native 88/92 GiB attempts failed, and fused-kernel savings remain untested.

- [Saved AV ordinary all14 confirmation](wsp-av-full14-confirmation.md): fixed778 AD/AV TP2 comparison after explicit exploratory selection and saved-word/full-file verification; no intervening dev90 claim or automatic adoption.

[Dependency-rich longer helper chains](wsp-long-helper-chains.md) add four authored training families (eight stages) and two disjoint transfer chains using the existing public-code evaluator and isolated executor. Complete interfaces and semantic foils are CPU-verified; the selected112 case objects remain unchanged in the120-row bundle. Longer input frames are not evidence of greater model capability.

The [matrix-only observation contract](wsp-matrix-only-observation.md) supports explicitly frozen, unobserved native parameters with complete matrix LL/margin provenance; it does not fabricate missing native gradients.

[UP-output unit sensitivities](wsp-up-unit-gains.md) select bounded coherent support through the [full-MLP-unit method](wsp-coherent-full-mlp-units.md). The [explicit full-unit editor mode](wsp-full-unit-edits.md) permits selected gate/up rows and down columns with an unchanged one-million-coordinate ceiling, preserves native/V and unselected bytes, and retains legacy sampled-coordinate bounds. Unit salience is a support choice, not circuit attribution or a measured candidate improvement.

The [fresh F01 CPU record](experiments/2026-11-07-fresh-f01-cpu-proposal.md) adds a fixed-purpose all-fold geometry/plan recipe, a nonroot network-isolated exact historical-input namespace owner, and an independent sampled-source/BF16-spec postcheck. All 76 rows and 410 matrices remain; unmet bounds and setup failures are preserved. These tools perform no model call or adoption.

The same F01 record includes a preview-first GPU6 finite replay owner with a pinned sublease, exact serving-CID restoration, protected-service checks and full 120-case frame verification. Its real CPU-container preview and one full 120-case AV/F01/AV GPU6 run passed, with exact selected-byte restoration, zero A/A error, healthy same-CID service recovery and unchanged protected snapshots. The F01 finite result exposed two referent retention violations, retained without a retry.

- `scripts/experiments/gemma4-purpose/fresh_joint_f01_summary.py`: fixed F01 finite-response audit; retains all 76 objective/fold comparisons and all 120 case margins, verifies zero-error base restoration and fold means, and records input/source hashes. Authored proxy values are not WSP scores.

- `scripts/experiments/gemma4-purpose/fresh_joint_referent_analysis.py`: read-only all-fold referent/code derivative cosines, full family/layer F01 dot-product contributions, and all 76 finite prediction errors; checks original gradient SHA and exact BF16 target bytes.


The [official host wrapper](../kairyu-bench) now supports explicit immutable
`KAIRYU_BENCH_RUNNER_IMAGE_ID` reuse, documented in [WSP](wsp.md#measured-runner-image-reuse).
It checks a complete local Docker image ID and preserves the normal launch path;
default commands still build the current checkout. This prevents experiment-only
source changes from silently changing a paired measurement's runner. Cache and
serving provenance still require separate evidence.

- The fresh F01 CPU/finite recipes accept explicit experiment root and candidate ID while preserving F01 defaults. Fixed F02 additionally selects referent as a target purpose. Finite `--unsaved-spec --exclusive-global` uses the original AV checkpoint plus a CPU-verified edit spec in RAM, with no claim of a saved candidate, and holds both global and GPU6 locks. All input/source hashes and exact target restoration are retained.

- [Fresh AV full-manifest live observation](experiments/2026-11-07-fresh-av-live-observation.md): an explicit remote observer/WSP owner accepts a plan directory and CPU-only preview, verifies the frozen observer source and exact baseline runner, keeps the 778-item / 14-area scope and exact pre-attachment CID restoration. Its paired code-capture audit retains every code question, request/rank/position/acceptance identity, all-item activation coverage and unmatched pre-run records. The completed AV baseline, measured four-request overhead, per-item scores and layer-25 fixed-offset RMS table are retained; unmatched records remain partial, and CPU preview does not deploy candidate weights.

- `docs/experiments/scripts/fresh-av-repeat-audit.py`: independent all-run identity scan, exact same-checkpoint dev10 repeat count, all 26 code outcomes and capture/runner/source condition comparison. Tied different models are never counted as repeats.

- `docs/experiments/scripts/fresh-materialization-evidence.py`: binds completed full editor verification to all nine source/candidate hashes, canonical trees and postverify full stat fields; records timing bounds and exact code hashes without redundant checkpoint payload hashing.

- [30/60 全問監査](experiments/2026-11-07-wsp-large-tier-audit.md): `docs/experiments/scripts/fresh-wsp-large-tier-audit.py` は現 manifest・旧 full dev/test・AV 反復を読み取り照合し、全 source 件数と個別 code 得点、入力 hash、実測と区別した時間概算を保存する。新規 WSP / GPU 実行なし。


Single-replica redeployment now checks all Compose project/service members,
including stopped retired observers, before preview/apply and guarded rollback.
The [saved-finalist guide](wsp-saved-finalist-confirmation.md#compose-membership-before-replacement)
records the real duplicate-label failure and exact-AV recovery. The CLI regression
fixture proves rejection before mutation and refusal of ambiguous rollback.

- [著者120ケースの事後監査](experiments/2026-11-07-authored-case-audit.md): `scripts/experiments/gemma4-purpose/authored_case_audit.py` は固定入力と保存有限値を CPU 再集計し、限定 fixture で契約反例を記録する。新候補・モデル評価・公式 benchmark 採点を行わない。

- [F02全778公式比較](experiments/2026-11-07-f02-official-results.md): `docs/experiments/scripts/fresh-f02-official-comparison.py` は既存 audit_run/strict compare を再利用し、入力・runtime receipts・全問題・source平均・凍結guardrail・usage/agent手数を保存する。capture本体やモデルを実行しない。

- [F02 matched observed diagnostic](experiments/2026-11-07-f02-observed-diagnostic.md): full778 result, incomplete close acknowledgement preserved, one-pass record/trace/rank audit, and a guarded CPU-only owned-retired log export with in-memory credential redaction. No metadata flag patch, retry, deletion or new model execution is introduced by the audit/export.

- [AV/F02 four-task observation display](experiments/2026-11-07-av-f02-four-code-observation.md): recorded one-off bounded extraction and existing offline viewer; all-code/all-item results retained, absent pairing and F02 close acknowledgement explicit. No new model run.

- `docs/experiments/scripts/fresh-wsp-cache-preflight.py`: pinned-runner CPU cache preflight for full canonical selections, package/source identities, H5, source rows and task-image names. Can relocate advisory locks while keeping data read-only. See `docs/experiments/2026-11-07-av-dev30-cache-preflight.md` for limitations and all preparation failures; no generation, scoring, prepare or pull.

- [AV full dev30 ordinary-replica run](experiments/2026-11-07-av-dev30-dp4-run.md): `docs/experiments/scripts/fresh-av-dev30-20261107.py` is the preview-first canonical-wrapper/flock owner, with pinned runner, input/closure identity and exact healthy idle replicas. `fresh-av-dev30-progress-20261107.py` reads response, verifier, trajectory, HTTP and capacity evidence, separating completed responses from scores. The actual full run completed all 2,338 scored items with zero unscored/errors; owner closure and exact healthy AV identities passed. Client HTTP traces only, no model deployment or activation capture.

- `docs/experiments/scripts/fresh-n30-complete-audit.py`: CPU audit for completed canonical AV dev30, using existing `audit_run` and exact `build_report` reconstruction; checks all 2,338 IDs/budgets, 90 code items, 14 areas/32 sources, immutable runner/serving receipts, HTTP request counts and ten agent trajectories. Refuses partial runs; writes fresh numeric evidence only, with no model/API/GPU calls. The actual complete-run audit and exact report reconstruction passed; missing/error/duplicate/nonfinite evidence remains explicit on anomalous inputs. Three boundary tests and the full 525-test CPU suite passed with 56 optional-dependency skips.

- [Authored 120-case native generation owner](experiments/2026-11-07-av-authored120-native-generation.md): preview-first, fixed four input batches through the existing public-answer generator; pinned AV/4-CID identity and shared evaluation lock; no scoring or service changes.

- [AV authored48 fixed CPU execution](experiments/2026-11-07-authored48-execution-results.md): all three existing catalogs, immutable commit4e088df sandbox, 46/48 passed and 375/384 numeric tests; separate 3/45 strata, full failure/unknown records and exact source/commands. No model call or new oracle; remaining72 unscored at this stage.

- [Native-answer / finite-margin case contrast](experiments/2026-11-07-native-finite-case-contrast.md): one read-only all120 ID join, fixed48 execution outcomes and explicitly scoped guard10 literal checks; no model calls, code execution or candidate selection.

- [Behavior18 fixed train/transfer calibration](experiments/2026-11-07-behavior18-implementation.md): new case recipe and thin offline adapter reuse the existing Docker worker; 18 correct programs passed288 numeric fixtures, all18 foils rejected by fixed witnesses or explicit entrypoint contracts. Train12 first; transfer6 only after candidate freeze. All revisions, raw final worker output and the common empty-array JSON boundary fix are retained; method requirements are separately unverified.

- [AV train12 native GPU4測定owner](experiments/2026-11-07-native-gpu4-measurement-plan.md): plan/SHAに束縛したscalar/vector全量2目的収集。既定preview、GPU4専有とgemma-2 exact CID復帰、全失敗保存。GPU実測は別承認・未実行。

- [Behavior18 tokenizer and fixed train12 owner](experiments/2026-11-07-behavior18-frame-and-train12-readiness.md): local CPU-only complete-frame audit (18/18, maximum290), explicit exact-plan approval, all4 AV serving identity, and global-lock retention through issued requests/idle closure; three focused CPU guards passed. No new model generation is implied.
