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
| Native-purpose and stratified matrix-coordinate local derivatives, local joint solve | scripts/experiments/gemma4-purpose | Closed as bounded experimental collectors/solver; caller provides the synthetic cases and model. |
| Saved finite response, Gram-based joint solve, BF16 projection and set composition | wsp/weights/finite_response.py and composition.py | Closed; proxy predictions remain distinct from actual model responses. |
| Fixed candidate finite replay and original-base repeat | wsp/weights/finite_replay.py | Closed for up to eight fixed native/sparse candidate specs per load, including restored-base measurement after late failures; no checkpoint switch or benchmark call. |
| Metadata freeze and conditional followup evidence | wsp/weights/receipts.py | Closed as explicit fingerprint and read-only ticket validation. Live snapshot collection, locks, and any skip policy remain external. |
| WSP capture, trace propagation, accepted-token attribution, lazy offline report | wsp/observe and supported agent adapters; observe-docker.py and observe-trace-gateway.py | Closed for documented runtime versions and coverage; observations do not establish causal ability. |
| Paired outcome/base-repeat evidence and recorded DeepSWE usage | docs/experiments/scripts | Closed as read-only reporting; official scoring is unchanged and unknown usage is not zero. Opt-in bounded trajectory-info API counters join existing trial UUIDs without reading messages or claiming HTTP telemetry. |
| Anonymous method review and metadata-only reserve audit | scripts/claude-method-review.py and wsp-reserve-audit.py | Closed; review is preview-first with explicit dispatch, and reserve audit does not read held-out answers. |
| Task-image prefetch, paced retry, and existing-cache digest audit | scripts/prefetch-task-images.py; docs/task-image-prefetch.md | Closed; explicit socket and daemon identity, expected registry digests, preview receipt, bounded downloads and retries. No container or daemon configuration changes. |
| Host-specific checkpoint switch wrapper | Existing Kairyu example's supported up/checkpoint API, documented operator boundary | Method-only: wrapper added no new model-switch implementation; host paths/admin state are not imported into this repository. |
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
