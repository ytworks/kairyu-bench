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
  "max_cases_per_microbatch": 1
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
