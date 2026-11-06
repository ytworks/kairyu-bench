# Focused BF16 checkpoint delta footprint

This read-only recipe measures all stored BF16 tensors between two checkpoints using the existing strict safetensors reader. It requires identical names/shapes/dtypes and reads each pair of payloads once in bounded chunks (default/max 8 MiB). No model/API is involved.

An explicit SHA-pinned JSON plan has `schema_version: 1`, `kind: bf16-delta-footprint-plan`, `expected_tensors`, optional `chunk_bytes`, `base`, `candidate`, and `prior_evidence`. Each identity contains `checkpoint_path`, complete `files` rows with `path/size/sha256`, `inventory_stat` keyed by relative path, and `tree_sha256` (SHA of path-sorted rows serialized with sorted keys and compact separators). Stat must include device/inode/size/mtime_ns/ctime_ns; uid/gid/mode/nlink are also accepted. Prior evidence is a nonempty list of `{path, sha256}` metadata references.

Build declarations from existing verified download/creator/freeze evidence. Provenance is caller-supplied: the tool pins evidence bytes and checks current full inventory/stat before and after, but does not authenticate upstream evidence or rehash weight payloads. Cache/attestation exclusions match existing redeployment inventory. Metadata inputs are bounded to 16 MiB.

```sh
python scripts/experiments/checkpoint-delta-footprint.py \
  --plan plan.json --plan-sha256 "$PLAN_SHA" --output preview.json
# After reviewing the plan and byte estimate:
python scripts/experiments/checkpoint-delta-footprint.py \
  --plan plan.json --plan-sha256 "$PLAN_SHA" --output footprint.json --run
```

Default preview reads only headers/metadata. Output must be fresh and outside both checkpoints. Execution publishes only after input continuity checks. Read order is logical tensor-name order, not physical file-offset order.

Output reports bit changes, numeric changes where **both operands are finite**, finite/nonfinite counts, and global/per-tensor/layer/projection-family delta L2. Signed zeros can differ in bits but remain numerically equal. Identical raw chunks skip float decoding but inspect exponent bits for NaN/infinity. Any nonfinite operand makes full `delta_l2` null; `finite_delta_l2` explicitly covers only finite pairs. FP64 sum of squared finite BF16 differences is range-safe at feasible file sizes; ordinary reduction rounding applies.

This focused path omits base-weight norms/moments, relative norms, quantiles, top coordinates and cosine. Estimated working memory is at most 384 MiB at 8 MiB chunks. Magnitude overlap is not directional agreement, cancellation, or capability evidence. Use `wsp weights diff` for comprehensive moments and sampled distributions.

## Actual public-base validation (2026-11-06)

The fixed script `1f306d44dfbad9a907473f0f437d807622286f2f32b60014d49ff6b9a5e5b7da`
completed one full 1,188-tensor BF16 observation in **83.388 seconds wall time**
(83.239 seconds inside the tool), reading exactly 125,092,355,504 paired payload
bytes. It found 58 changed tensors, 1,130 bit-identical tensors, and
1,015,123,842 changed elements under both the stored-bit and finite-numeric
definitions. There were no nonfinite operands; delta L2 was
25.575145366575246. All changes were in text attention output projections
`o_proj`, layers 1–58; layers 0/59 and all other tensor families were unchanged.

Original identity `cab440f871a32b5ea1c12364db7dd35ac8768a83b413ca24bbc18d6478eccbd9`
and public protocol-view identity
`3dfb3bd1db0ab2ba4fff843616bb0d3109e0b97a6c9349658620908f2a24bebc`
came from prior full-file verification. All 12 original and 9 public-view file
stats matched before and after. The launch/completion receipt additionally pinned
the imported reader SHA
`898297c05e7179cdc4568d6f290b978073bcebafea804ac043b54030e8205bf9`
and checked its stat/hash continuity. No additional payload SHA pass was performed.

Retained local evidence is under
`analysis/code90-resume-20261106/heretic-public-base-official-protocol-v1/`:
`fast-delta-plan.json` (SHA `4b321ed38a2f2617ef658b4931b744940f50deb854f941ccd94019254de353f1`),
`fast-delta-footprint.json` (SHA `a15a986cd376d738410e5ff6a1d9170e3e7ae0f75d0fbdcee74d8fc531cb5b7f`),
and `fast-delta-launch.json` / `fast-delta-completion.json`.
The numeric result is separate from any generated-answer score.

The earlier comprehensive `wsp weights diff` run was deliberately interrupted
after 715.305 seconds and 27.145 GB read, with **no completed numeric result**.
That existing tool remains valid and available: it also calculates source and
candidate moments, norms, cosine, top coordinates and sampled quantiles. The
focused recipe omits those metrics, so these timings are different workloads,
not a like-for-like speed comparison. The successful scan ran on CPUs 8–15 with
BLAS/OMP set to one; its 12:14:02–12:15:25 UTC interval overlapped the official
four-arm run that began at 12:13:20 UTC. It is not an isolated inference-speed
measurement.

Three focused CPU behavior tests cover chunked direct-reference deltas, signed
zeros/nonfinite values, strict layout and missing tensors, metadata-only preview,
mutation rejection and fresh output. The complete subsequent CPU suite passed
**506 collected: 454 passed, 52 optional skips**, whole-tree Ruff and mypy over
73 package source files, with no source change during checks. The scoped script
mypy check also passed. The full receipt
`analysis/code90-resume-20261106/cpu-integration-fast-delta-footprint-v1/ci-results.json`
has SHA `7fea5402f4019526411f32059901726a1381eebd097e19099fb51935a692d891`.
