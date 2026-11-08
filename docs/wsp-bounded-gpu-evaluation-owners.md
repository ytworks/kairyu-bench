# Bounded GPU evaluation owners

These experimental recipes manage a temporary evaluation topology and restore the exact original containers. They do not edit checkpoint files or select/adopt a model. JSON plans carry machine paths, immutable inputs, current service identities and time budgets; credentials are inherited and never written into a plan.

- [screen-matrix-overlays.py](experiments/scripts/screen-matrix-overlays.py) was exercised on eight GPUs with one saved-parent empty-overlay control and six matrix candidates.
- [full14-tp2-dp2-pair.py](experiments/scripts/full14-tp2-dp2-pair.py) is **prospective only**. CPU syntax and inert preview passed; no TP2×DP2 server, MTP workload, smoke or official run has validated it.

Both recipes are scoped to pinned Gemma4 images, UID 1003, four existing TP2 services and the repository's existing checkpoint/redeployment/counter tools. They are experiment recipes, not a general scheduler. A different image, protocol or topology needs a new concrete plan and review.

## Invocation and ownership

Default invocation is inert and does not inspect Docker or contact a model:

```sh
python -B docs/experiments/scripts/screen-matrix-overlays.py \
  --plan /absolute/path/plan.json --plan-sha PLAN_SHA256
```

The DP2 script has the same CLI. Execution adds `--authorization /absolute/path/authorization.json --authorization-sha AUTH_SHA256 --apply`. Authorization declares `status=authorized`, `issued_by=root` and the exact plan/runtime SHA256. Here root identifies the coordinating experiment agent; execution still requires ordinary UID 1003.

Use a fresh exclusive output directory containing the plan. Locks, exclusive status files, fresh outputs and exact temporary-container labels prevent reusing another run. Bind the source actually invoked, not a historical hash. Finalizers independently attempt child cleanup, temporary cleanup and every original-container restoration. Failed receipts remain evidence; neither recipe retries automatically.

| Plan scope | Required binding |
| --- | --- |
| Inputs | Current incumbent, selection, source/draft freezes, exact specs and code hashes, protocol metadata, current file-stat inventories |
| Services | Four exact original IDs and semantic snapshots; screen also pins full Config/HostConfig; embeddings preserve identities and lifetimes |
| Tools | Absolute repo/Python/tool paths with hashes, pinned images and existing generation/execution/counter tools |
| Outputs | Fresh canonical receipt/cache/results paths and unique temporary names |
| Limits | Startup, active work, child execution and restoration deadlines |
| Authorization | Exact plan/runtime hashes and allowed temporary topology/restoration scope |

The source itself defines the complete plan schema. Prior full-file hashes plus current stat continuity establish saved-source identity; this is not a fresh full-payload hash pass. Logical overlay identity is separate from the physical checkpoint tree.

## Exercised six-candidate screen

The screen reuses the [sparse overlay loader](wsp-vllm-sparse-overlay.md) with a saved current parent and at most six ordinary BF16 O/MLP specs. Native and Q/K/V edits are rejected, and overlay-parent chains are unsupported.

Four TP2 slots use GPUs 0–1, 2–3, 4–5 and 6–7. Each round evaluates control plus up to three candidates. The same empty-overlay control process stays loaded across at most two rounds; only candidate temporaries are replaced. All arms use the same pinned overlay image, static MTP4, protocol metadata and cache policy. Writable NVMe paths include FLASHINFER_WORKSPACE_BASE. Canonical source/draft/proof mounts are read-only.

Before generation, both ranks must publish matching post-TP selected BF16-word readback and pinned-draft bypass receipts. Empty-overlay proof has zero selected words and is not a full-model GPU hash.

The executed protocol used [library transfer8](wsp-library-transfer-cases.md), native thinking ON, seed 0, concurrency 8, maximum 32,768 tokens, request timeout 1,200 seconds and retries 0. Temperature/top-p were omitted, retaining server defaults. The immutable CPU scientific executor used eight trials/task, seed 20261109, 30 seconds/case, 512 MiB, one CPU, no network, read-only filesystem and UID 1003. Correctness is recorded, never an 8/8 gate.

Public summaries contain numeric execution fields, usage, finish reasons and code hashes. Startup diagnostics are private and captured only before generation. Model answers and hidden reasoning are excluded from public evidence.

### Actual result and limits

The November 6, 2026 run took 685.8306 seconds including loading, both rounds and restoration. All 64 requests ended with stop; errors/unknowns were zero. Selected-word checks, draft bypass and restoration of all four healthy original IDs passed.

| Arm | Tasks passed | Trials passed | Output tokens |
| --- | ---: | ---: | ---: |
| Control, round 1 | 7/8 | 56/64 | 17,577 |
| AH | 7/8 | 56/64 | 18,040 |
| AI | 7/8 | 56/64 | 18,568 |
| AJ | 7/8 | 56/64 | 19,853 |
| Control, round 2 | 7/8 | 56/64 | 16,800 |
| AK | 6/8 | 48/64 | 18,619 |
| AL | 7/8 | 56/64 | 20,016 |
| AM | 7/8 | 56/64 | 20,250 |

Every arm failed filter-state-response. AK also failed symbolic-axis-jacobian. No candidate improved its paired control on these eight tasks. Repeated control binary outcomes matched but generated-code hashes matched 0/8; no exact functional/deterministic parity claim is made. This is authored-task execution evidence, not official accuracy or an unseen-evaluation claim. A finalist still needs physical saving and fresh official evaluation before adoption.

Evidence under analysis/code90-resume-20261106/current-ad-library-overlay-screen-v1:

| Artifact | SHA256 |
| --- | --- |
| Plan | `83bc589f9c547bccc40e611c505b536e0dcb4e99a5615b7e664510bd0a31ad5a` |
| Completion | `b3ac854aa7e35230812f7ede1a87538f25d4af5a6d7e4c55eb48a6c53789382b` |
| Numeric summary | `2ba543be9c84dd8b19e7ccca841e77c84c67c980275c68ef562a7270e9fbd0c6` |
| Derived owner postchecks | `22e4588174fb462883fbdd49c9e1ce6929c7d3551063b03be1862c4507228e1e` |

## Prospective matched TP2×DP2 pair

This unexecuted recipe uses two ordinary saved checkpoints, four GPUs each, vLLM internal DP2, TP2 and static MTP4. It uses the base image, not overlays. Compared arms retain identical source/image/sampler/protocol with global chat concurrency 32, logprob 8, agent 1 per arm and independent identical numeric-observer policies.

An existing eight-request authored smoke records correctness and requires observable MTP draft activity, not 8/8 correctness. The subsequent prospective path launches the immutable official tier 10/dev full14 pair, captures actual runner image/source parity and counters, then restores all originals. No custom router or performance claim is introduced.

The published port adds plan fields `repo`, `python`, `results_dir`, and `tools.redeploy={path,sha256}`. Snapshot lookup uses exact plan IDs; helper import happens only for apply. The official HF_TOKEN is inherited and must never be printed or saved.

## Source relationship and checks

The screen was exercised as runtime
`f8f6c1ad0f1ebc6d2d8278b0c397175c77d99c74a11e29de9cb56f2bbf2a6b45`.
Its published version is
`f5e7a4ea498cca565644833479931d6cc642fb02433d4dd80a7210f79cafdf3e`:
Ruff formatting and splitting grouped imports repaired the publication's CI
failure. Its parsed AST is identical after normalizing that import split.
The exercised runtime remains preserved separately.

The DP2 port derives from reviewed but unexecuted `cdac49afd3d15f6c958a3f8938559f7eecea1f378b3a0014c2ae282f582fdfdb`. Only path/helper binding, exact-ID lookup, deferred import and flat numeric trial-count fields changed. Mathematics and lifecycle ordering are unchanged. The port was then formatted
from `8021731899fde53d893a8259d5a34e7856dbfd5802eab94a71a316a2ad040956`
to `958404efe2f5966dce29847a0ee6a7ed9f16612a93a069c8f759b56f28e18439`
with the same AST-equivalence check. It remains unexecuted.

CPU syntax and both inert previews passed without Docker/model operations. Receipt: analysis/code90-resume-20261106/overlay-owner-publication-v1/validation.json, SHA256 `2f7f99ba7a91749ec8c1485f93dc4f9c2d04c99effbc1ff429f56b1ccc67aa44`. No model reruns or new static-registry tests were added.

The formatting repair passed Ruff for both published scripts without a model
rerun. The exact before/after hashes and AST check are recorded in
`analysis/code90-resume-20261106/published-owner-ruff-repair-v1.json`.
