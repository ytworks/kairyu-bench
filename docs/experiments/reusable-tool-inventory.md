# Reusable experiment tools: inclusion ledger

This ledger distinguishes functioning tools included in this PR from
experiment-specific operations retained as method documentation. "Closed" means
the reusable function is available here; it does not claim that every temporary
script or its host state is shipped unchanged.

| Earlier function/source family | Included implementation | Status and boundary |
| --- | --- | --- |
| Safetensors identity, bounded norms, matrix/head/FFN geometry, lexical rows | wsp/weights/checkpoint.py, statistics.py, matrix.py | Closed for reusable reads/metrics; Gemma tensor/head selection remains explicit input. |
| Focused exact BF16 checkpoint delta observation | scripts/experiments/checkpoint-delta-footprint.py; tests/test_checkpoint_delta_footprint.py | Closed; all matching BF16 tensors, bit/finite-numeric changes, finite/nonfinite counts and delta L2; existing comprehensive reader/statistics retained. |
| Native scalar/vector and sparse matrix checkpoint create/verify; original-byte restoration | wsp/weights/editor.py | Closed; schema 1, default dry-run, explicit apply, full-file restoration proof. |
| Verified unchanged-shard pool | wsp/weights/shard_pool.py and editor --shard-pool | Closed; explicit donor/source map, new pool only, UID/inode/SHA guards. |
| Pinned whole-text LoRA recipe, native norm combination, exact/realized adapter audits | scripts/experiments/gemma4_lora | Closed as a pinned Gemma experimental recipe, explicit compiled-library build; no binary or model weights committed. |
| Whole-projection donor sensitivities and saved-purpose profiles | gemma4_lora/donor_direction.py and profile_donor_directions.py | Included; zero-gate first-order custom VJP over all 410 text projections, explicit authored folds, cost/family profiles. Continuous derivatives do not substitute for merged BF16 or execution results. |
| Signed per-module donor solve and persistent recipe | gemma4_lora/solve_module_directions.py and module_lora_recipe.py | Included implementation and CPU contract validation; explicit six scopes/bounds, FP32 coefficients, one BF16 realization from original weights. Two of four requested module proposals have actual RAM finite evidence; two remain unmeasured after the bounded run. No persistent checkpoint or capability adoption followed. |
| Exact signed-module BF16 RAM finite replay | gemma4_lora/finite_module_replay.py; shared weights/finite_replay.py PublicAnswerEvaluator | Included implementation and eight Torch CPU contract tests; actual GPU replay completed two proposals before the 20-minute budget, then restored all 410 matrices byte-exact with margin/NLL/KL A/A differences 0. The four-proposal run remains incomplete; no capability claim. |
| Saved module checkpoint / RAM byte-equivalence verifier | gemma4_lora/verify_saved_checkpoint.py | Included CPU-only tool and four fixture tests; actual saved candidate verified in 111.681 seconds: all 410 BF16 tensor SHA values and 12 files, finalized staged-int16 proof, creator inventory and redeploy-compatible freeze. No GEMM, model call, deployment or capability claim. See [verification contract](../wsp-saved-module-verification.md). |
| Exact composition of saved nonoverlapping checkpoints | scripts/experiments/compose-verified-checkpoints.py | Included CPU byte-splice tool: source/component pins, selected-tensor SHA, original and component virtual-reversion hashes, private output and redeploy freeze; no GEMM or second rounding. See [contract](../wsp-checkpoint-composition.md) and [K experiment plan](2026-11-06-native-coherent-splice-k.md). |
| Native-purpose and stratified matrix-coordinate local derivatives, local joint solve | scripts/experiments/gemma4-purpose | Closed as bounded experimental collectors/solver; caller provides the synthetic cases and model. |
| Continuous-calibrated matrix-only radius and BF16 proposal | scripts/experiments/gemma4-purpose/derive_matrix_radius.py; [guide](../wsp-derived-matrix-radius.md) | Included portable two-stage composition of existing solvers, explicit pins and 120/180-second budgets, one E2E save/verify and foreign-output regression; no fresh real-model execution. |
| Normalized coding/science LL target preparation | scripts/experiments/gemma4-purpose/prepare_normalized_objective_plan.py; [guide](../wsp-normalized-objective-targets.md) | Portable fixed-policy preparation using existing observations and numerical primitives; generated plan only, with external realized-norm postcheck and one E2E adapter/save contract fixture. Whole CPU 512 = 460 passed + 52 optional skips; source unchanged. |
| Optional load-time sparse BF16 overlay | scripts/experiments/vllm_sparse_overlay; [guide](../wsp-vllm-sparse-overlay.md) | Separate formal vLLM loader/plugin, one real safetensors/TP-boundary CPU fixture, prior-hash/stat identity and fresh all-rank selected-word proof; real model rollout remains unvalidated. |
| Saved finite response, Gram-based joint solve, BF16 projection and set composition | wsp/weights/finite_response.py and composition.py | Closed; proxy predictions remain distinct from actual model responses. |
| Fixed candidate finite replay and original-base repeat | wsp/weights/finite_replay.py | Closed for up to eight fixed native/sparse candidate specs per load, including restored-base measurement after late failures; no checkpoint switch or benchmark call. |
| Metadata freeze and conditional followup evidence | wsp/weights/receipts.py | Closed as explicit fingerprint and read-only ticket validation. Live snapshot collection, locks, and any skip policy remain external. |
| WSP capture, trace propagation, accepted-token attribution, lazy offline report | wsp/observe and supported agent adapters; observe-docker.py and observe-trace-gateway.py | Closed for documented runtime versions and coverage; observations do not establish causal ability. |
| Paired outcome/base-repeat evidence and recorded DeepSWE usage | docs/experiments/scripts | Closed as read-only reporting; official scoring is unchanged and unknown usage is not zero. Opt-in bounded trajectory-info API counters join existing trial UUIDs without reading messages or claiming HTTP telemetry. |
| Anonymous method review and metadata-only reserve audit | scripts/claude-method-review.py and wsp-reserve-audit.py | Closed; review is preview-first with explicit dispatch, and reserve audit does not read held-out answers. |
| Task-image prefetch, paced retry, and existing-cache digest audit | scripts/prefetch-task-images.py; docs/task-image-prefetch.md | Closed; explicit socket and daemon identity, expected registry digests, preview receipt, bounded downloads and retries. No container or daemon configuration changes. |
| Existing serving/control backend | Operator-provided Kairyu example control.py and Compose API | Existing external backend, not reimplemented or bundled as a generic serving system. Host paths/admin state remain operator inputs. |
| Guarded single-replica experiment redeployment | docs/experiments/scripts/redeploy-replica.py | Included thin helper over the existing backend; preview first, explicit apply, target identity/idle/runtime checks, no-deps service recreation and rollback attempt. Two real applies completed for B and original-base with healthy/read-only checkpoints; no gateway comparison with mixed weights. |
| Candidate-specific queue, scheduled continuation, freeze-to-holdout-to-DeepSWE orchestration and publishers | Experiment methods and evidence, not a new general deployment service | Method-only: fixed study policy and one-off authorized operations; no implicit automatic publication. |
| Exact named-shard retirement and one-off nested-Docker mount/reward recovery | Experiment record | Method-only: target-specific destructive/repair operations are not generalized or distributed as automatic tools. |
| Historical prompt/case authoring and named amplitude/line-search recipes | Experiment methods; explicit data/spec inputs to included collectors and replay | Method-only: case data and selection decisions are not represented as universal benchmark or model-improvement tooling. |
| Backups, source-deployment snapshots, local environments, binary kernels, checkpoint shards, private model outputs | None | Not reusable source deliverables; preserved where needed as local provenance. |

The prior same-checkpoint dense reuse optimization depended on completed dev
evidence, fresh full weight hashes, the same healthy replicas, and a held
evaluation lock. The included validator preserves those metadata comparisons
without silently weakening the editor's full verification or implementing a
serving/queue shortcut.

The audit inventory records source line counts, dependencies, file hashes, and
host-specific/credential-handling flags locally. No credentials, raw trajectories,
model weights, environment dump, or signed download URL is included in this PR.

The completed evaluation and per-task official outcomes are recorded in [Frozen 0397 final DeepSWE comparison](2026-11-06-deepswe-frozen-h0397.md). Both models scored 1/113; the edited checkpoint did not satisfy the strict-gain HF publication condition.

Sampled matrix create/verify and RAM replay now admit up to 262,144 matrix coordinates (512 per matrix), covering the 410 × 512 stratified projection sample without selecting cells to fit the former 65,536 implementation bound. Explicit opt-in and sample-plan provenance remain required for persistent editing; the one-million total RAM edit cap remains unchanged.

Additional code90 tools are included in source: science_cases.py (trusted self-authored programs and executable train/transfer validation), science_generate.py (explicit endpoint, native-thinking public-answer generation), science_sandbox.py (restricted non-root CPU Docker execution, numeric comparison and unknown infrastructure outcomes), and gemma4_lora/donor_gradient_alignment.py (actual stored-BF16 sampled-coordinate alignment). Their focused tests protect execution, provenance, cleanup and split boundaries. The matrix/native collectors now accept current per-layer attention geometry and optional one-case microbatches. Failed runtime attempts, proposal bounds and the distinction between calibration outcomes and official scores are retained in the [science pilot record](2026-11-06-science-matrix-pilot.md).

The [single-replica helper](single-replica-comparison.md) has two completed real
apply receipts under the experiment-relative paths
analysis/code90-resume-20261106/broad14-native-deployment/receipt.json and
analysis/code90-resume-20261106/broad14-base-deployment/receipt.json. These record
the actual named service changes and retained runtime settings. Candidate
identity reused creator hashes with a documented ctime-interval assumption;
original-base identity reused prior full hashes with exact stat continuity.
Neither apply is described as a fresh full-payload hash.

The [module RAM replay contract](../wsp-module-finite-replay.md) distinguishes
all-answer margin/NLL from first/last-answer-token diagnostic KL. Its CPU
fixtures and source review are complete. The owner-run v2 proof measured two
of four requested proposals, then hit its time budget. All410 matrices restored
byte-exact, source continuity held, and restored-base margin/NLL/KL differences
were 0. Two other proposals remain unmeasured; incomplete run status is retained.
Measured correct-answer NLL worsened in 33/34 and 32/34 folds despite margin gains,
so these proposals were not adopted or sent to WSP. A successful preview or
continuous solve alone is not this proof. The new anonymous
[structured-donor review](anonymous-structured-donor-review-2026-11-06.md)
and [matched-screen outcomes](2026-11-06-code90-matched-screens.md) retain these
limits and the moderate-tradeoff selection policy.


追加のadvanced科学校正は [science recipes](../../scripts/experiments/gemma4-purpose/README.md#code90-advanced-自作科学校正) の advanced_science_cases.py に収録した。旧basic16学習/8転移はbyte不変で、新たに16学習/8転移を追加し、共有validator/exportと既存sandboxの明示catalogで評価する。CPU correct192/192・全24foil失敗、別seed契約、trainの公式chat frame最大288tokensを検証済み。元モデルのadvanced v1生成実行は生結果21/24だが、2問の公開文に曖昧さがあり能力失敗とは断定しない。明示revision2/catalog advanced-v2へ2promptだけを修正し、v1出力のbyte一致と全24prompt/sample契約を独立確認した。公開dev概念を参考にした校正であり、公式WSPや未見評価ではない。旧120行を保持したcombined136 v2は21目的42fold、最大505tokens、転移はgradientへ入れない。

correct-logprob目的をnative/donorで明示するobjective_variants.py、対になる42foldから6scopeを解くsolve_positive_module_directions.py、既存の決定的module merge経路のCPU所要時間を測るprofile_merge.pyを収録した。正答logPとmarginの符号・frame・coordinate/source対応、許容損失、FP32係数と実BF16変位の境界を保持する。[目的補正の記録](2026-11-06-positive-correct-donor-amendment.md)と[匿名Opusレビュー](anonymous-positive-correct-review-2026-11-06.md)を参照。solverは連続予測のみで、実モデル/WSPを起動せず採用を自動決定しない。

今回の全CPU検証は448件（399 pass、49 optional skip）、Ruff合格、mypy73 files合格で、検証前後source変更なし。advanced v2の共通136件でnative/donorそれぞれmargin/correct-logprobの4収集が全42foldを完了した。新solverの6案は連続制約内で解が得られた段階であり、実BF16の品質・生成正答・WSP向上を確認したという意味ではない。

公開回答の境界診断public_answer_diagnostic.pyを追加した。最大3自作caseで旧assistant-startと新forced-empty-thought、公式full-model/submodel、BF16/FP32 softcap、paddingを分離して数値だけ記録する。2件のTorch CPU契約・Ruff・独立source reviewを完了。実GPUの効果は別receiptで確認する。旧frameの6positive案は未適用retiredであり、448件CIの記録はこの後続2件を含まない。

public_answer_diagnostic.pyの親実行3caseは25.062秒・24forwardで完了し、12条件すべてで公式full-modelと同じBF16 softcapを使うsubmodelのlogits差0を確認した。[境界の発見・実数・限界](2026-11-06-public-answer-frame-audit.md)を独立記録した。コード生成やWSP能力の検証ではない。

### Final public-answer frame validation

The final frame/serial/headroom snapshot passed **453 collected tests: 403 passed, 50 optional-dependency skips**, whole-tree Ruff, and mypy over 73 source files. Source hashes were unchanged during checks (analysis/code90-resume-20261106/cpu-integration-final-frame-v2/ci-results.json). A same-tick filesystem fixture flake was reproduced and fixed by explicitly advancing the test mtime; the deployment helper was not changed. Three-case actual-model framing parity and matched native/donor baseline values are recorded in [the frame audit](2026-11-06-public-answer-frame-audit.md). The first new-frame module replay is running; no capability improvement is claimed.

### Saved verification and joint replay validation

The saved-checkpoint verifier, joint native/donor proposal and one-shot residual repair, direction probes, and optional joint RAM replay passed the complete CPU suite: **466 collected, 414 passed, 52 optional-dependency skips**, whole-tree Ruff and mypy over 73 source files. Source hashes were unchanged during the checks (analysis/code90-resume-20261106/cpu-integration-joint-v1/ci-results.json). The persistent balanced checkpoint was actually verified against all 410 measured tensors and all 12 files, then evaluated with the official dev90 protocol; it did not improve the official score and is not adopted. The joint candidate is still an experiment.

## Cross-donor factor-space observation

The generic CPU-only [LoRA direction observer](../lora-direction-comparison.md) compares explicitly aligned ordinary LoRA factors of different ranks and F32/BF16 dtypes without dense updates or model calls. It uses FP64 Gram products, records all input pins and exclusions, and separates unrounded direction geometry from actual BF16 effects. Three contract fixtures and an independent review passed before the [two-donor actual observation](2026-11-06-two-donor-factor-observation.md); the measured 410-module comparison took 15.232 seconds. This does not establish capability complementarity.

## Serving counter baselines

The [counter snapshot/diff tool](../wsp-serving-counters.md) keeps a bounded whitelist of process metrics, refuses identity changes and observed resets, and reports missing series as unknown. Its nine CPU fixtures include the actual vLLM label schema (including repetition), exact large-counter subtraction, and source-field filtering. The complete CPU suite passed **489 collected: 437 passed, 52 optional skips**, with whole-tree Ruff, mypy73 and unchanged source hashes. The first actual J/fresh-original pre-run captures both confirmed zero running/waiting requests; later deltas remain endpoint aggregates rather than per-item or retry evidence.

The completed [J comparison](2026-11-06-weighted-reasoning-c1-review.md) used the counter tool and the unchanged source-defined startup probe. Its +3.055556 pp fresh-original difference has a 95% CI crossing zero and adds no success above all prior B/H/I outcomes; it does not establish code90 or adoption. K therefore tests accumulation of the fixed B/H bytes with fresh component reruns, not summed historical gains.

The lossless [text-factor extraction tool](../wsp-text-adapter-extraction.md) and its four CPU fixtures are included. Its actual K1 extraction preserved 820 F32 text factors and the original config bytes while explicitly excluding 378 vision factors. The [L trial record](2026-11-06-k1-exported-text-c1-review.md) retains source-versus-derived hashes and the unknown Heretic base revision. Complete CPU validation passed 499 collected tests (447 pass, 52 optional skips), Ruff and mypy73; this does not certify capability transfer.

The [M four-condition preregistration](2026-11-06-public-base-alignment-fourway-m.md) separates public modified-base and fixed-adapter effects under the original protocol; it is a plan and anonymous method review, not a new capability result.

The [paired score-only linear contrast CLI](../wsp-paired-score-contrast.md) and four behavioral CPU tests are included. Joint row draws preserve cross-arm covariance, source means receive equal weight, and fractional scores remain numeric. Pair parity is tested against the existing official comparator. The prior K exercise is post hoc and changes no adoption decision; M remains a separate preregistered comparison. The subsequent full CPU check passed 503 collected tests (451 pass, 52 optional skips), Ruff and mypy73 with unchanged source hashes; the earlier 499-test extraction result remains its separate historical validation.

The [focused footprint recipe](../checkpoint-delta-footprint.md) completed one 1,188-tensor paired read in 83.388 seconds, with prior full-file SHA/stat continuity and a separately pinned reader. Nonfinite operands make full delta L2 null; finite-subset L2 is labeled separately, and signed-zero differences count as bits only. The preceding full-statistics attempt was interrupted without a result; its broader moments/cosine/quantile implementation remains available. The two workloads and overlapping official-run timings are not a like-for-like speed comparison. Complete CPU validation passed 506 collected tests (454 pass, 52 optional skips), whole-tree Ruff and mypy73, with unchanged source hashes; the three new behavioral fixtures are included in that count.


## Library-composition training and transfer extension

The [training builder](../../scripts/experiments/gemma4-purpose/library_train_cases.py) includes all16 new authored prompt/correct/foil definitions and independent numeric references: eight families with two interface/composition variants each. Its [guide](../wsp-library-science-training.md) records append-only merging of the exact existing136 objects into152 examples, preserving21 purposes/42folds. A CPU-only check found new frames at most311 tokens and inherited frames at most509. NumPy2.2.6/SciPy1.13.1/SymPy1.14.0 reference validation passed every correct implementation and detected every foil; this is not a model or official-score result.

The separate [transfer module](../../scripts/experiments/gemma4-purpose/library_transfer_cases.py) defines eight unused families with gradient_allowed=false, independent reference checks and metamorphic relations. The existing [sandbox](../../scripts/experiments/gemma4-purpose/science_sandbox.py) adds explicit library-only catalogs and a declared underflow allowance while retaining nonfinite/overflow/divide/invalid checks and complete catalog/content identity. Existing NumPy catalogs keep their defaults. Training and transfer definitions are reviewable source; generated response bodies remain outside the repository.

The [science/code group-plan builder](../../scripts/experiments/gemma4-purpose/prepare_science_code_groups.py) and [method guide](../wsp-shared-science-code-groups.md) use exact selected matrix names consistently for target dual norms and proposal eligibility, with the existing constrained BF16 solver. No new optimizer or benchmark scoring rule is introduced. Prospective method choices, actual BF16 support and finite/ordinary-generation outcomes remain separate evidence.

The offline [scientific-library staging helper](../../scripts/experiments/stage_science_library_executor.py) and [COPY-only executor Dockerfile](../../scripts/experiments/science_library_executor.Dockerfile) reproduce an already installed SciPy distribution after exact ABI/dependency checks. The helper rejects output overlap with source packages or pinned inputs and verifies copied bytes; it performs no installation hook, download, Docker action or model call. Installed-image CPU execution is separately recorded in the [transfer/executor guide](../wsp-library-transfer-cases.md). This executor is distinct from active inference containers.

The combined library/group-tool CPU validation passed 518 collected tests (463 passed, 55 optional skips), whole-tree Ruff, mypy74, data checks and 31 shell syntax checks, with unchanged source hashes. The managed suite lacks the optional scientific libraries; the train/transfer reference and foil E2E checks also passed separately in the recorded SciPy/SymPy environment, and the offline executor passed its actual CPU sandbox checks. Receipt: A/cpu-integration-library-science-v1/ci-results.json, SHA256 94822c7a5dbb131df8e1949235d5604f6a3b6a71d29b774327a43ed80ab488ac. CPU validation overlapped the separately owned expanded-cohort collection; no speed comparison is claimed.

## Bounded evaluation owners

The [matrix-overlay screen](scripts/screen-matrix-overlays.py) is an AST-equivalent formatting copy (after normalizing split imports) of the exercised owner for the six-candidate library transfer screen. The [TP2×DP2 full14 pair recipe](scripts/full14-tp2-dp2-pair.py) is a plan-bound portability copy of a prospective owner, with no actual DP2/model evaluation yet. The [shared guide](../wsp-bounded-gpu-evaluation-owners.md) records exact-CID independent restoration, selected-word/draft proof boundaries and explicit authorization. The [experiment result](2026-11-07-ad-science-dependency-sets.md#ordinary-transfer-screen-no-gain-over-the-matched-control) records no transfer improvement, the repeated-control variability and AL's selection for a later saved-weight comparison; it claims no adoption or official-score gain.

[The derived matrix batch runner](../../scripts/experiments/gemma4-purpose/run_derived_matrix_batch.py)
orchestrates at most six pinned functional-target plans in two CPU workers,
with one 120-second analytic stage and one 180-second BF16 stage per method.
Completed candidates receive the existing exact numeric postcheck; failed
stages remain in the batch report. There is no replacement optimizer, model
call, automatic retry or adoption. See the [functional-target guide](../wsp-shared-science-code-groups.md).
