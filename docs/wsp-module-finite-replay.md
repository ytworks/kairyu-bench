# Exact BF16 module-recipe RAM replay

The experimental [finite_module_replay.py](../scripts/experiments/gemma4_lora/finite_module_replay.py)
compares one through eight **fixed** signed per-projection LoRA recipes in one
Gemma model load. It calls the same `delta_block` and `merge_block` primitives as
the versioned persistent module editor. It does not approximate merged weights
with nonzero adapter hooks.

The model contains the actual BF16 result of `W + c * (2 B A)`: FP32 GEMM in
128-row blocks, multiply by 2, multiply by the exactly represented FP32 coefficient,
separate FP32 addition, then ties-to-even BF16 once. Coefficient zero preserves
the original bytes, including signed zero. Every uploaded block is read back and
compared byte for byte. Each candidate records every projection's source/target
SHA256, changed element/byte counts, and actual BF16 delta norm.

A plan pins the source index, a prior complete content-identity receipt, authored
public-answer cases, their exact tokenizer frame, and the fixed candidate specs:

```json
{
  "schema_version": 1,
  "architecture": "gemma4_module_lora_bf16",
  "checkpoint_index_sha256": "<sha256>",
  "checkpoint_attestation": {"path": "/source-identity.json", "sha256": "<sha256>"},
  "cases": {"path": "/authored-cases.json", "sha256": "<sha256>"},
  "frame_evidence": {"path": "/frames.json", "sha256": "<sha256>"},
  "candidates": [
    {"id": "fixed-candidate", "spec": "/module-spec.json", "sha256": "<sha256>"}
  ]
}
```

The prior source receipt must have `identity_level=content_sha256`, complete
recognized checkpoint/config/tokenizer inventory, per-file SHA256 and matching
current file stats. This is continuity from that full scan, **not** a new full
weight hash. Module recipe preparation also tracks the complete source tree's
file stats. Candidate specs, donor/config/costs, numerical implementation and
compiled-kernel fingerprints are checked again after restoration.

```sh
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export KAIRYU_BENCH_BF16_LORA_LIBRARY=/absolute/trusted/kernel.so
PYTHONPATH=src python scripts/experiments/gemma4_lora/finite_module_replay.py \
  --checkpoint /absolute/original-checkpoint --plan /absolute/fixed-plan.json \
  --output /absolute/fresh-diagnostics --maximum-host-cache-gib 192 \
  --maximum-seconds 1200
```

The command above is a metadata/recipe preview and does not load the model.
Execution additionally requires `--run --gpu-exclusive-approved` and exactly one
explicitly exposed GPU with at least 70 GiB free. Source checkpoint and scripts
should be mounted read-only; only the fresh diagnostics directory is writable.
For this 31B experiment the container needs an explicit `--memory 256g` budget
and eight assigned host CPUs with BLAS/OpenMP set to four threads. The earlier
160 GiB collector limit is insufficient: original/delta caches alone require
about 164 GiB, and file-backed checkpoint pages also count against the container.
The runner rejects an insufficient cgroup-v2 limit before loading the model.
The owner stops and restores its chosen serving replica and remains responsible
for the surrounding container lifecycle. No serving or Docker operations are
implemented in this recipe.

For the 410 text projections in this study, the CPU cache is approximately
54.55 GiB original BF16 plus 109.10 GiB FP32 delta, with a 192 GiB explicit bound.
It never stores six complete BF16 candidates. The first base sweep precedes
cache construction. GEMM is computed once per block and reused across candidates;
each candidate is merged anew from the original, never accumulated from a previous
candidate. GPU memory includes only one model and bounded copy buffers.

The shared `PublicAnswerEvaluator` is also used by native/sparse
[finite replay](wsp-finite-response.md). All public correct/foil answer tokens
contribute to each example's mean logprob margin; correct NLL covers all correct
answer tokens. Full-vocabulary KL is diagnostic only at each correct answer's
first/last token positions. Authored calibration must be supplied explicitly;
this tool neither proves dataset independence nor turns these values into
generation accuracy or a WSP score.

A failure in a later candidate, copying, cache construction, or result writing
enters the same final restoration path. Only fully initialized original buffers
are exposed to restoration. All cached original BF16 bytes are copied back and
read back, then the base forward sweep is repeated. Recovery may exceed the
20-minute execution limit; failure remains `incomplete`, never `completed`.
Uncatchable process/host failure still requires owner recovery.

For later publication/materialization, use the same pinned image, NumPy/BLAS
build, four CPU threads, kernel SHA and block width. Independently require all
410 persistent projection hashes to equal the RAM receipts, including zero-scope
unchanged projections. Stored-byte hashes/counts must match exactly; accumulated
floating norm statistics may differ only by explicitly checked reduction
roundoff. Neither a common formula nor a small norm alone establishes equivalence.

The recipe selects no candidate and imposes no all-purpose nonregression veto.
It reports gains, losses, NLL and diagnostic KL separately. Free-generation and
official benchmark validation remain subsequent owner-controlled steps.
