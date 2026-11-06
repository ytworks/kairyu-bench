# Reusable experiment tools: inclusion ledger

This ledger distinguishes functioning tools included in this PR from
experiment-specific operations retained as method documentation. "Closed" means
the reusable function is available here; it does not claim that every temporary
script or its host state is shipped unchanged.

| Earlier function/source family | Included implementation | Status and boundary |
| --- | --- | --- |
| Safetensors identity, bounded norms, matrix/head/FFN geometry, lexical rows | wsp/weights/checkpoint.py, statistics.py, matrix.py | Closed for reusable reads/metrics; Gemma tensor/head selection remains explicit input. |
| Native scalar/vector and sparse matrix checkpoint create/verify; original-byte restoration | wsp/weights/editor.py | Closed; schema 1, default dry-run, explicit apply, full-file restoration proof. |
| Verified unchanged-shard pool | wsp/weights/shard_pool.py and editor --shard-pool | Closed; explicit donor/source map, new pool only, UID/inode/SHA guards. |
| Pinned whole-text LoRA recipe, native norm combination, exact/realized adapter audits | scripts/experiments/gemma4_lora | Closed as a pinned Gemma experimental recipe, explicit compiled-library build; no binary or model weights committed. |
| Whole-projection donor sensitivities and saved-purpose profiles | gemma4_lora/donor_direction.py and profile_donor_directions.py | Included; zero-gate first-order custom VJP over all 410 text projections, explicit authored folds, cost/family profiles. Continuous derivatives do not substitute for merged BF16 or execution results. |
| Signed per-module donor solve and persistent recipe | gemma4_lora/solve_module_directions.py and module_lora_recipe.py | Included implementation and CPU contract validation; explicit six scopes/bounds, FP32 coefficients, one BF16 realization from original weights. Two of four requested module proposals have actual RAM finite evidence; two remain unmeasured after the bounded run. No persistent checkpoint or capability adoption followed. |
| Exact signed-module BF16 RAM finite replay | gemma4_lora/finite_module_replay.py; shared weights/finite_replay.py PublicAnswerEvaluator | Included implementation and eight Torch CPU contract tests; actual GPU replay completed two proposals before the 20-minute budget, then restored all 410 matrices byte-exact with margin/NLL/KL A/A differences 0. The four-proposal run remains incomplete; no capability claim. |
| Saved module checkpoint / RAM byte-equivalence verifier | gemma4_lora/verify_saved_checkpoint.py | Included CPU-only tool and four fixture tests; actual saved candidate verified in 111.681 seconds: all 410 BF16 tensor SHA values and 12 files, finalized staged-int16 proof, creator inventory and redeploy-compatible freeze. No GEMM, model call, deployment or capability claim. See [verification contract](../wsp-saved-module-verification.md). |
| Native-purpose and stratified matrix-coordinate local derivatives, local joint solve | scripts/experiments/gemma4-purpose | Closed as bounded experimental collectors/solver; caller provides the synthetic cases and model. |
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
