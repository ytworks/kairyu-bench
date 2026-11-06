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
| Joint edits from saved local gradients | [local-G solver](../scripts/experiments/gemma4-purpose/joint.py) | Explicit purposes, fold constraints, permitted losses and cost metric; BF16 predictions require finite measurement |
| Matched cost metrics on saved local gradients | [cost comparison](../scripts/experiments/gemma4-purpose/compare_costs.py) | Identical constraints, explicit near-zero floors, output-name collision checks; no new forward |
| Saved finite-response analysis and quantized joint-set proposals | [finite-response guide](wsp-finite-response.md) | Endpoint secants, active-cone projection and Pareto reports are predictions, not scores |
| Original-base composition | [composition](../src/kairyu_bench/wsp/weights/composition.py) | Report overlap, cancellation and actual BF16 deltas; no assumed additive benefit |
| Exact composition of verified saved checkpoints | [byte-splice composer](../scripts/experiments/compose-verified-checkpoints.py), [guide](wsp-checkpoint-composition.md) | Disjoint full-tensor bytes, full/partial virtual reversion, fresh output hashes; no GEMM or rerounding and no assumed functional additivity |
| Fixed native/sparse-matrix RAM replay | [finite replay](../src/kairyu_bench/wsp/weights/finite_replay.py) | Original → up to eight fixed native/matrix sets → restored original; no persistent weight writes or automatic adoption |
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

The [science recipe guide](../scripts/experiments/gemma4-purpose/README.md) includes executable author-owned train/transfer cases, preview-first public-answer generation, and restricted CPU-container execution. Generation records the declared checkpoint and decoding frame, persists public answers and numeric usage only, and does not establish serving identity by itself. The sandbox keeps expected values outside the generated-code process and reports infrastructure failures as unknown. Neither result is a WSP score.

The same guide documents current/legacy attention geometry, one-case gradient accumulation and native/matrix joint proposals. [Stored donor alignment](../scripts/experiments/gemma4_lora/README.md) compares saved BF16 bytes with sampled scientific derivatives; it cannot infer the full donor direction from those cells. See the [code90 protocol](experiments/2026-11-06-code90-plan.md), [GPU pilot and actual failures](experiments/2026-11-06-science-matrix-pilot.md), and [candidate-method review](experiments/anonymous-science-candidate-review-2026-11-06.md).

The [J final comparison](experiments/2026-11-06-weighted-reasoning-c1-review.md) separates saved calls from the source-defined startup probe and process counters. The [fixed B+H composition plan](experiments/2026-11-06-native-coherent-splice-k.md) records contemporaneous four-arm comparisons; its implementation proof does not establish a score gain.

The [lossless text adapter extractor](wsp-text-adapter-extraction.md) explicitly partitions text and vision factors, preserves raw F32/BF16 bytes and unchanged config, and distinguishes original repository payloads from derived text-only exports. It creates no checkpoint or benchmark call.

The [K1 exported-text trial record](experiments/2026-11-06-k1-exported-text-c1-review.md) records the actual lossless F32 selection, 499-test CPU validation and fixed-unit cross-base recipe; checkpoint quality and official scores remain separate outcomes.

The [paired score-only contrast tool](wsp-paired-score-contrast.md) adds explicitly weighted multi-arm contrasts with shared source-stratified row draws, preserving covariance. It reads SHA-pinned numeric exports only and does not alter official scoring or declare adoption. The subsequent full CPU check passed 503 collected tests (451 pass, 52 optional skips), Ruff and mypy73 with unchanged source hashes; the earlier 499-test extraction result remains its separate historical validation.

The focused BF16 footprint tool passed three behavior fixtures and an actual 1,188-tensor scan in 83.388 seconds (different scope from comprehensive statistics; official-run overlap disclosed). Its [guide](checkpoint-delta-footprint.md#actual-public-base-validation-2026-11-06) records the prior-hash/stat proof, imported-reader pin and nonfinite semantics. The subsequent complete CPU check passed 506 collected tests (454 pass, 52 optional skips), Ruff and mypy73 with unchanged source hashes; all earlier validation counts remain historical records.
