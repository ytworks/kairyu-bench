# F04 LoRA optimizer recipe: implementation and CPU checks

The bounded experiment recipe is implemented; this record contains no Gemma
forward, GPU training, model deployment or dependency installation. The parent
is completed F04. Rejected F05 and its native screen remain unchanged.

[train_lora.py](../../scripts/experiments/gemma4-purpose/train_lora.py) freezes
the BF16 parent, wraps all410 actual text projections with ordinary rank8/alpha8
FP32 A/B, and performs actual AdamW steps (LR1e-4, no weight decay, clip1,
accumulation4, seed0). A=random/B=0 preserves the initial function; a zero A
gradient at the first step is expected. Training order is shuffled once by
seed0, exhausts all declared rows, then cycles in that same order. It is
eval/autograd with dropout disabled and existing FP32 saved-activation offload.

The input is the SHA-pinned public original TRAIN cohort, with exact ID/split,
reference-quality and tokenizer proofs. Held-out IDs are refused. Official ON
user+assistant-reference rendering targets the code and official turn terminator
[106,107]; global EOS1 is absent and is not manually appended. Prefix tokens are
masked, each example is exact length with no foil/padding, and lengths over1024
stop preflight rather than being truncated. The initial mistaken EOS1 check was
corrected before any model run after independent tokenizer evidence showed the
actual template boundary. The exact helper was then compared with all255
independent token streams and train204: all equal, train maximum336/all366.

Every completed step logs actual CE, examples/tokens/time/VRAM, each projection's
calls and A/B grad norms, and all per-matrix norm/cap reductions. Continuous
scale-BA norms use r×r FP64 Gram products and full-matrix denominators; only B is
shrunk when its ratio exceeds0.005. Step1/4/64 record detached aggregate
input/output activity, without a localization claim. All partial microbatch
results and failed attempts remain. CPU guarded receipts/stat and frozen
parameter version/storage checks are not relabeled as a fresh full-weight hash.

After4 optimizer steps, a fresh pilot export and SHA-bound pause allow root to
approve continuing to fixed64 with the same in-memory model/optimizer.
Timeout/stop is pilot_only with factors retained; no1GB optimizer checkpoint is
written. The training deadline includes loading and excludes only the explicitly
bounded operator wait. No parent baseline or WSP is rerun by this recipe.

[The existing dense merger](../../scripts/experiments/gemma4_lora/dense_lora_recipe.py)
now has one explicit local_training receipt alternative to public donor
provenance, restricted to factor_format v2. It binds the exact parent,
adapter/config, positive completed steps, frozen base and data/source hashes;
mixed or forged/mismatched provenance is refused. Existing FP32 matrix arithmetic,
one BF16 rounding, unchanged-span validation and full saved verification are
reused. Matrix edits cannot use native-only faststore. The separate dense
candidate needs a fresh disk/reserve check; no existing checkpoint is removed.

CPU validation:

- Existing host LoRA tests plus new tests:13 total,11 passed,2 Torch-only
  numerical tests skipped because that host environment lacks Torch.
  This includes a complete synthetic410 local-receipt create/verify roundtrip,
  tamper/mixed-parent/data binding, frame/holdout/continuation guards.
- Existing immutable af9 image, CPU-only/no-network/no-GPU/user1003, readonly
  repository and NVMe temporary directory: all5 new tests passed,1.378s
  unittest /3.914787s container. It verifies first-step Agrad0, later A/B
  gradients, frozen base, FP32 merged-function agreement, Gram cap affecting
  only B, and independent one-round BF16 storage words. It does not assert
  BF16 unmerged-output and merged-matmul bit equality.
- Final focused Ruff and AST parse passed. The final extra microbatch/attempt
  diagnostic fields were added after these numerical tests; kernels were
  unchanged. Initial Ruff5 semicolon errors, one intermediate unused
  diagnostic variable, and an initial CPU-container command's missing
  bare python entrypoint (exit127/zero tests) are preserved; the latter
  was corrected to the existing /usr/bin/python3. No install occurred.
- The CPU test container was removed, the exact27-container set was preserved,
  and no Gemma weights were loaded. GPU memory, throughput and real training
  compatibility remain unverified until the authorized pilot.

[Usage/schema](../../scripts/experiments/gemma4-purpose/README.md#bounded-local-lora-training-explicit-opt-in)
and [original design](2026-11-07-f04-lora-training-design.md) distinguish
training fit, actual saved BF16 deltas, normal-generation execution and official
WSP ability. This implementation makes no coding improvement or adoption claim.

The ordinary ON chat format does not prove that direct-code supervision preserves reasoning. A lower answer loss may reflect immediate-code/output-format fitting or suppressed thinking rather than better inference. Compare the candidate’s existing WSP reasoning-token/finish traces with the completed parent traces; no new parent run is required, and no training-loss gain is a capability claim.

[Exact source snapshots, all CPU logs/failures/commands, hashes and pending dense-merge preparation](records/2026-11-07-f04-lora-training-implementation/SHA256SUMS) are retained. Sources remain frozen during the separately owned GPU run; documentation updates do not alter the runtime pins.
