# Exact BF16 module-recipe RAM replay

The experimental [finite_module_replay.py](../scripts/experiments/gemma4_lora/finite_module_replay.py)
compares one through eight **fixed** signed per-projection LoRA recipes in one
Gemma model load. It calls the same `delta_block` and `merge_block` primitives as
the versioned persistent module editor. It does not approximate merged weights
with nonzero adapter hooks.

The model contains the actual BF16 result of `W + c * (2 B A)`: FP32 GEMM in
128-row blocks, multiply by 2, multiply by the exactly represented FP32 coefficient,
separate FP32 addition, then ties-to-even BF16 once. Coefficient zero preserves
the original bytes, including signed zero. With the default
`--device-proof roundtrip`, each complete BF16 matrix is uploaded once, read back
once and compared byte for byte. The CPU GEMM and rounding blocks remain exactly
128 rows under both supported proof modes. Each candidate records every projection's source/target
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
candidate. GPU memory includes only one model and bounded copy buffers. CPU transfer
workspace/readback temporaries are bounded below 1 GiB and included in the
explicit estimate. Moving device transfers to matrix granularity does not change
the arithmetic, candidate bytes, source proof or restoration comparison.

## Optional staged device proof

`--device-proof staged-int16` is opt-in; `roundtrip` remains the default.
During cache preparation it reads and hashes the attested original tensor bytes
into a private CPU buffer, then compares the complete loaded parameter's int16
view with those bytes uploaded into a staging tensor. This preparation comparison
does not write the model. Each candidate still uses the identical CPU 128-row
GEMM/merge arithmetic. The complete CPU target is uploaded into staging, copied
to the parameter, and checked with all-element device int16 equality before its
CPU target SHA is recorded. Restoration performs the same byte proof on every
cached original and still requires the base forward A/A check.

This is **CPU target SHA plus device storage-byte equality**, not a
device-to-host hash. Records explicitly label
`candidate_hash_origin=cpu_target_verified_by_device_int16_equality`,
`actual_ram_storage_bytes_verified=true`, and
`actual_ram_readback_verified=false`. Restoration uses
`storage_bytes_exact` plus `proof_method`; the legacy
`readback_exact` field is retained only for roundtrip mode. Every projection,
including zero-coefficient projections, is proved before a candidate is recorded.
The extra GPU staging tensor is at most one projection; the existing conservative
four-matrix transfer-workspace allowance remains below 1 GiB. No kernel,
coefficient, block width, metric, or model-forward calculation changes.

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
verified using the explicitly selected storage-proof method, then the base
forward sweep is repeated. Recovery may exceed the
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

## Bounded merge and transfer microprofile

`scripts/experiments/gemma4_lora/profile_merge.py` times two explicit source
projections without loading a model or writing a checkpoint. Its default is an
input preview. `--run` compares the existing 128-row `merge_block`,
`dense.merge1`, and preallocated ctypes kernel paths, checks their target
BF16 SHA/count/L2 agreement, and separately records source reading, cached
direction computation, hashing, and a host copy. The kernel already uses four
OpenMP workers; adding an outer pool of four would create sixteen workers and
is not justified by wrapper overhead alone.

```sh
python scripts/experiments/gemma4_lora/profile_merge.py \
  --checkpoint "$BASE" --spec "$FIXED_MODULE_SPEC" \
  --output "$FRESH_TIMING_JSON" --maximum-seconds 60 --run
```

Use the same pinned image, compiled kernel, NumPy/BLAS and thread settings as the
RAM/persistent comparison. Explicit `--tensor` may select one or two mapped
matrices; defaults are layer-0 attention O and MLP down. Source and donor remain
read-only. The report hashes the selected matrix bytes and checks input
continuity; it does not claim a new full-checkpoint hash.

The 2026-11-06 CPU measurement used NumPy 2.2.6, eight available CPUs,
BLAS/OpenMP four threads, and the same compiled kernel as the RAM run. It
completed in 3.736 seconds with exact target SHA/count/L2 agreement on every
path and unchanged source inputs:

| Matrix | Shape | Module merge | Wrapper merge | Direct kernel | Target SHA | Host copy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Layer-0 attention O | 5376 × 8192 | 0.0934 s | 0.0919 s | 0.0803 s | 0.0866 s | 0.0357 s |
| Layer-0 MLP down | 5376 × 21504 | 0.2336 s | 0.2408 s | 0.1933 s | 0.2269 s | 0.0635 s |

This small profile identifies comparable merge and SHA costs, not the complete
cause of a 319.6-second full-model application. GPU transfers, extra copies,
memory locality and the complete tensor distribution remain separate costs.
The exact measured source is archived beside
`module-merge-cpu-profile-v1/{launch,timings}.json`; later changes only extend
profiling, without reclassifying that CPU measurement.

After an owner releases one GPU, optional
`--probe-device cuda:0 --gpu-probe-approved` compares two scratch-tensor paths.
The original path uploads, reads back, compares every BF16 storage element and
hashes the returned bytes. The alternative uploads a staging tensor, copies it
into the scratch target, and compares int16 views on device. It records
**CPU target SHA plus all-element device byte equality**, not a D2H hash.
A final independent full CPU byte oracle must match both paths; its time is
reported separately from the alternative. Integer views preserve signed zero
and NaN payload distinctions. This probe does not load a model. Its independent oracle validates the optional
runner proof method before use; the two proof modes remain explicitly distinct.

The two small CPU contracts cover multi-block signed merges, fresh-output and
source protection, and staged equality with signed-zero and NaN storage bits.
GPU scratch testing is opt-in and restricted to one exposed GPU with 4 GiB
free; the operation has a 60-second budget. The parent experiment owner manages
GPU availability and the pinned container launch.


The owner-run scratch GPU measurement completed in 6.925 seconds with unchanged
source inputs and exact independent CPU oracle matches on both matrices:

| Matrix | Original roundtrip + hash | Staged upload/copy/equality | Final oracle, outside staged timing |
| --- | ---: | ---: | ---: |
| Layer-0 attention O | 0.22013 s | 0.06472 s | 0.18856 s |
| Layer-0 MLP down | 0.57852 s | 0.08223 s | 0.49477 s |

The staged column excludes the required CPU target SHA; the separate CPU profile
measured about 0.0866/0.2269 seconds for that work. Consequently the transfer-only
speedups must not be quoted as full application speedups. Full candidate timings
remain to be measured. Launches with a read-only container filesystem require
a writable NVMe `TMPDIR` for Torch import; the first attempt failed at import
before GPU tensor work, and the separately preserved second attempt supplied it.
The experiment artifacts are `module-merge-device-profile-v{1,2}`.

The existing exact six-candidate replay fixture runs both proof modes, including
signed-zero bits, a failure on the fifth forward, source-byte mismatch rejection,
and final original restoration. Staged mode makes zero D2H `raw()` calls within
the application path; its CPU fixture is separate from the actual GPU scratch
measurement above.

For collections made with serial public answers, pass
`--answer-batching serial` to the module replay. The default remains `paired`.
Serial replay splits the existing deterministic packing into individual cases,
uses the same correct/foil pair-derived padded width for each answer, and runs
one answer through the model and output head at a time. All answer-token
margin/NLL contributions, per-case and per-fold means, and the diagnostic KL
positions remain unchanged. This matches the serial collector layout; it does
not imply numeric equivalence to paired or differently padded execution.
The selected layout is recorded in preview, provenance, and summary.
