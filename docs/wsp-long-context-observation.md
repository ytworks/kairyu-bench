# Opt-in long-context authored observations

The native and sampled-matrix collectors, public-answer diagnostic, and finite
replay accept three optional settings:

- `--max-frame-tokens 1024` or `2048` bounds the complete rendered prefix plus
  each correct/foil answer. Oversized cases fail; tokens are never truncated.
- `--text-attention-backend sdpa` requests text-only SDPA. It requires an
  explicit token limit. Root, vision and audio dispatch remain eager.
- `--cuda-memory-cap-gib N` optionally limits the PyTorch CUDA caching allocator
  before loading weights. N must be finite, positive and no greater than current
  free or total device memory. This is not a limit on all GPU allocations.

For a collector, any opt-in policy requires chat-public-answer mode,
`--answer-batching serial --max-cases-per-microbatch 1`. The diagnostic is
intrinsically serial. Finite replay already makes serial answers one case at a
time and requires `--answer-batching serial`. Finite plans must include the
exact `observation_policy` matching their CLI; a policy-free historical plan
cannot silently execute under a new backend.

Without these options, native and finite replay retain their historical uncapped
frame behavior; matrix and diagnostic retain their 512-token checks. All loaders
retain eager dispatch. Default artifacts omit the new policy field. The matrix
sample budget remains 512 coordinates per tensor, unrelated to frame length.
Objectives, complete public-answer scoring, forced-empty-thought framing,
padding, optimizer, sampled coordinates and gradient-checkpointing behavior do
not change.

## Bound policy and runtime

New provenance and summaries carry the same `observation_policy` object:

```json
{
  "max_frame_tokens": 1024,
  "attention_dispatch": {
    "": "eager",
    "text_config": "sdpa",
    "vision_config": "eager",
    "audio_config": "eager"
  },
  "cuda_memory_cap_gib": 88.0,
  "answer_batching": "serial",
  "max_cases_per_microbatch": 1,
  "execution_order": "original"
}
```

88 GiB is an example, not an approved real-run budget. An absent memory cap is
represented by null. A cap-only eager policy may leave the explicit frame limit
null and retains each tool's legacy frame checks.

Before weight loading, the collector tokenizes and checks supplied complete
frames. The finite evaluator does the same before its model load. The resolved
model/subconfig attention dispatch must exactly match the request for present
subconfigs. Absent optional audio/vision configs are recorded as null with an
explicit absent list; they do not imply a fallback. There is no
silent eager fallback, automatic retry, or shortened-context fallback.
Runtime `observation_execution` records resolved dispatch and allocator
fraction/free/total memory when a cap is requested. Existing allocated/reserved
peak measurements remain available; allocator measurements are not asserted to
cover every CUDA allocation.

The coupled proposal adapter and completed-collection composer reject different
policy presence, backend, token limits or allocator caps, including eager/SDPA
mixtures. They require provenance/runtime agreement. Changing GPU free memory
between two runs is recorded but is not treated as a different numerical policy.

## Validation and limits

Existing CPU fixtures exercise complete correct/foil token preservation, both
1024 and 2048 boundaries, rejection before model loading, the exact dispatch
passed to the loader, allocator setup before loading, rejection of insufficient
memory and silent backend fallback, and matching/mismatched collection
composition through the existing proposal-and-saved-byte fixture. No new static
registration test or GPU/model experiment is required by this code change.

This path is CPU-contract validated only. It does not establish 31B long-context
capacity, fused attention, GQA memory savings, eager parity, or better generated
answers. In the pinned Gemma4 implementation, full-attention head dimension 512
can require repeated K/V under SDPA; requested SDPA is not evidence of a fused
kernel. Any future GPU trial must separately pin its accepted or explicitly
provisional saved parent, new authored cases, token limit, backend policy,
resource owner and measured peaks. Old eager gradients cannot be relabeled as
new SDPA observations.

The complete CPU suite finished with **520 collected: 464 passed and 56 optional skips**, whole-tree Ruff, mypy74, data and shell checks passing, and no source changes during checks. Evidence: `A/cpu-integration-long-context-v1/ci-results.json` SHA256 `a23ac1ac9c4605214f66c4c14f6fe7baa5054944e93ad27be69aa32650d24a6d`. The existing test count is unchanged. This excludes the subsequent materializer label-only AV authorization change, which received its own inert preview and Ruff checks.

The optional `--execution-order longest_frame_first` orders folds by their largest
complete prefix plus correct/foil answer, then cases within each fold by that
length. Serial collectors therefore process the largest actual frame first,
without a separate model probe. Saved purpose, fold, and case order stays
canonical. The explicit policy records this choice; adapters and composition
require an exact match. Omission retains the original execution order.
The observation owner can bind an explicit free-memory budget separately from
the Torch allocator cap. The prospective AV run uses 88 GiB for the allocator
plus 2 GiB outside it, requiring at least 90 GiB free after releasing O2.
This is a resource limit, not evidence of fused SDPA or reduced memory use.

Actual provisional AV observations used all 120 authored training cases
(19 purposes, 38 folds), with complete frames up to 1,268 tokens. The declared
2,048-token limit is a rejection boundary; this run does not establish that
every 2,048-token workload fits. The accepted checkpoint remained separately
bound to an immutable AD-state snapshot while AV was observed provisionally.

| Attempt | Collected blocks | Allocator cap | Observed result |
| --- | --- | ---: | --- |
| v1 | native + matrix | 88 GiB | Native OOM before a completed fold; matrix LL 31/38 partial folds |
| v2, explicitly authorized resource trial | native + matrix | 92 GiB | Native OOM before a completed fold; matrix LL 27/38 partial folds |
| v3, explicitly authorized matrix-only collection | matrix | 88 GiB | Both LL and margin completed all 38 folds / 120 cases |

The two native failures requested another 80 MiB and 40 MiB respectively.
At failure, PyTorch had 86.88/90.76 GiB allocated and 1.09/1.22 GiB reserved
but unallocated. The failed native case index inside the first longest-first
fold was not emitted. Partial matrix peaks were not recorded; no partial
arrays were resumed or combined. Each attempt restored the same O2 container
healthy and preserved AD1, AV0, AR3 and the embedding services.

The successful matrix-only run recorded peaks of 85.5930 GiB (correct LL)
and 85.6586 GiB (margin), with 88.0 GiB reserved. Collection took 39.687 and
69.745 seconds after model setup. Both objectives ran in parallel on O2's
two GPUs; concurrent ordinary full14 evaluation means these are not
uncontended timing measurements. Requested and resolved text dispatch was
SDPA, root/vision remained eager, and the absent audio config was recorded
as null. This does not identify which PyTorch SDPA kernel executed.

The observer's opt-in `observed_blocks: ["matrix"]` with
`native_parameters: "frozen_not_observed"` requires exactly two matrix
objectives, one on each declared GPU. Native gradients and cross-family
baseline parity are explicitly unmeasured. Full checkpoint identity still
binds every unchanged native parameter; no fake native arrays or historical
eager arrays are supplied. Legacy four-collector execution remains the default.

The source offers a plausible memory explanation, not an allocation-level
causal proof: native collection enables gradients on every native norm/scalar
parameter, requiring parameter-derivative saved activations; matrix collection
keeps parameters frozen, differentiates from a detached embedding leaf and
contracts sampled matrix factors in output hooks. Matrix collection still
retains an input-VJP graph.

Evidence is retained under `A/provisional-av-long-purpose-observation-v{1,2,3}`.
The v3 completion is `fda99e5539c586a06fbb37e338760faf124d7b770ae5eb574486e92697be2847`;
postchecks `76477b618dfdd49b67cb41ec68eba75ff96c7cac88b97768517b6b7dc5bcdcfa`
and the 88-file output inventory `dadcaa5e0cb51b482bd0193b00eb02c63dbae0616cfbbae39407732b38621729`
cover both complete matrix collections. This observation makes no candidate
edit, capability, adoption, eager-parity, or native-gradient claim.

## Native saved-activation CPU storage

The native collector alone accepts the optional
`--saved-activation-storage cpu-fp32`. It requires public-answer mode, serial
answers and no virtual gains. Omission (or `device`) keeps the old execution
and artifact fields. This option neither changes attention, objectives,
complete frames, weights, nor the declared allocator cap.

During each answer's forward and immediate VJP, PyTorch saved-tensor hooks copy
nonleaf CUDA FP32 tensors with at least two dimensions to CPU without changing
dtype. Backward restores each saved value to its original device on demand.
Other saved tensors remain on their device. Copies are blocking; no compression,
recomputation, automatic fallback or activation checkpointing is introduced.
The hooks are scoped to that answer and are removed even if forward or backward
fails. Native trainability and gradients retain the existing final cleanup.

Requested storage is recorded separately as
`native_saved_activation_storage` in provenance and summary runtime.
`saved-activation-storage.json` records completion or interruption, the same
policy, saved/offloaded/unpacked event counts, and copied byte totals grouped
by dtype and shape. Completed summaries also carry these metrics. Counts and
bytes include duplicate saves or unpacks: they are not unique allocations,
peak CPU usage, or measured GPU memory reclaimed.

Installed RMSNorm source computes FP32 normalization followed by a trainable
scale. The scale derivative needs the normalized activation; at the observed
1,268-token frame, four residual norms across 60 layers have about 6.09 GiB of
such logical FP32 payload. Including Q/K and final norms gives about 9.89 GiB
by shape arithmetic, not a measured unique saved-storage inventory or a proven
cause of the previous OOMs. Some constant normalized inputs are leaves and are
not offloaded; BF16 layer-scalar inputs are also outside this filter. Duplicate
copies, host memory use, transfer latency and backward restore peaks can remain
substantial. No claim that the 31B native workload now fits under 88 GiB is made.

The existing native norm/serial fixture was extended, with no new test method.
In the installed Torch CPU image it checks exact LL and saved native gradients,
success and reference-failure cleanup, and an exception inside the save-hook
context. The private fixture exercises CPU-to-CPU copies through the same hook
code; it does not measure CUDA transfers or memory reduction. The focused test
passed in 0.099 seconds, and focused Ruff/mypy checks passed. No GPU trial or
production model collection has run for this option. Any later observation
must use fresh provenance for the actually selected current parent; historical
AV arrays must not be relabeled after a parent change.
