# AV / F02 saved-weight and live-activation comparison preparation

Prepared 2026-11-07. This is an independent comparison-preparation record, not a
candidate ranking, an official WSP score, or an instruction to rerun either
benchmark. No model request, checkpoint write, service change, or GPU work is
performed by this preparation. The observation owner retains runtime ownership.

## Inputs and provenance

All operations run on the authorized SSH host. These abbreviations are used:

```bash
R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
B=$R/worktrees/kairyu-bench-weight-tools
P=$R/analysis/fresh-purpose-20261107-f02/materialization
C=$R/analysis/fresh-purpose-20261107-av-f02-comparison-prep-v1
AV=$R/checkpoints/code90-current-ad-helper-av-v1
F02=$R/checkpoints/fresh-code-joint-f02
N=$R/analysis/fresh-purpose-20261107-av-live10-v1
N2=$R/analysis/fresh-purpose-20261107-f02-live10-v1
export KAIRYU_BENCH_PYTHON=$R/infra/weight-tools-ci/bin/python
```

| Input | SHA256 |
|---|---|
| AV checkpoint tree | `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf` |
| F02 checkpoint tree | `46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c` |
| `P/full-verification-freeze-input.json` | `f897d77bf5e4bf219f6c8391cbcb738c780912d1a666ff6ac5cb4ae150fa8715` |
| `P/edit-manifest.json` | `1cb562c4d9fbd1fe3bc6a2f2c489b5b482406a5d31ff086ca0f2b48a49d47880` |
| `P/verify.json` | `3d5b609ef17bc3a1ba1ad064af55e1beac36174e722a3e72339179f7a734a8c5` |
| `C/plan.json` | `a377adb73772b60f7ad02feb6a1083855f0d6c9c38efc771b4bb560d60f3e6f7` |
| `C/launch.json` | `dd9a80aedb16fa8e71da3274d4cbd621f5acffb6ccfc864879ba83a67a21e276` |
| `C/manifest-derived-summary.json` | `42ba7ec1419e539d183063d6cea0c0e279fbe7dec24204df547d28e3a5d5d648` |

The saved editor verification completed successfully: `verified=true`,
`legacy=false`, all nine file hashes verified, seven nonweight files unchanged
from AV. Create took 215.173925 seconds; verify took 174.630913 seconds. The
subsequent full-diff plan saved complete nine-file stat snapshots for each
checkpoint. Stat continuity preserves the earlier verification evidence; it is
not a claim of a new full payload hash. The running diff will add its own full
identity hash pass only if it completes successfully.

## Why perform one full saved-weight diff

The manifest proves actual BF16 stored changes and exact delta norms, but its
per-edit source norm describes one changed coordinate. Summing those source
norms covers only the coordinates that changed after BF16 rounding. It does not
cover the full sampled proposal support, the full edited matrices, or the full
checkpoint. Therefore a full diff adds materially new denominators and a direct
independent check of full tensor coverage.

The existing implementation is used without a new tool. The only launched
full diff is this command (already running; do not launch it again):

```bash
cd "$B"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
/usr/bin/time -v -o "$C/weights-diff-time.txt" \
  ionice -c 3 nice -n 15 taskset -c 32-35 \
  ./kairyu-bench wsp weights diff "$AV" "$F02" \
  --group-pattern 'model\.language_model\.layers\.([0-9]+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj))\.' \
  --chunk-mib 4 --sample-size 4096 --top-k 8 --seed 0 \
  --output "$C/weights-full-diff.json"
```

There is no selection pattern, partial-coverage option, axis expansion, or
hash-disabling option. The group pattern labels layer/projection combinations;
it does not select or exclude tensors. Both inputs contain 1,188 stored tensor
entries and 31,273,088,876 elements. Stored entries are counted as represented
in the checkpoint, rather than claiming semantic deduplication of tied weights.

The planned work is one complete numeric pass over each checkpoint and one
complete file-identity hash pass over each checkpoint. Estimated logical bytes
read are 125,125,114,929 per checkpoint, or 250,250,229,858 total, including
metadata, numeric data, and hashes. This is logical I/O, not physical disk I/O.

The job started at epoch 1794030964.1380677 (owner PID 262967, timing wrapper
262968, CPU worker 262969). At 63.5 seconds, process logical reads were
3,276,774,580 bytes, physical read_bytes was zero (page cache), and RSS was
188,736 KiB. That rate suggested approximately 40 minutes for the numeric
pass plus the final hash pass; this is a projection, not elapsed completion
time. At document preparation the execution remained running; no full-diff
norm or successful hash count is claimed yet. The owner writes actual elapsed
time and exit status to `C/execution.json`; `weights-diff-time.txt` contains
resource use. The output is published only at completion. This low-priority
single-thread CPU work does not block official score comparison.

## Norm and count definitions

Let D be stored F02 minus stored AV after BF16 rounding. The full diff reports:

- Per tensor: norm(D) / norm(full AV tensor).
- Global: norm(D over all compared floating tensor entries) /
  norm(all compared floating AV entries).
- Edited-tensor relative norm: the same delta numerator divided by the
  full AV norm of every tensor with at least one stored-bit change.
- Layer/projection groups: norms combined by sum of squares, never an average
  of tensor-level relative norms.
- Stored-bit changes and finite numeric changes separately; nonfloating exact
  differences are reported separately and excluded from floating norms.
- Full-data moments, L2 norms, cosine, and top-delta coordinates. Only delta
  quantiles use up to 4,096 deterministic sampled coordinates per tensor
  (seed 0); small tensors can be fully sampled. Quantiles are not full norms.

The manifest-only summary was computed once from the SHA-checked manifest and
verify outputs: require all edits to be one BF16 element of an exactly matched
q/k/v/o/gate/up/down matrix, group by tensor/layer/family, sum changed counts,
and combine each stored delta_l2 and source_l2 with math.hypot. The nine
manifest/verify file hashes were cross-checked. No checkpoint payload was
rescanned for this summary.

| Family | Matrices | Changed BF16 cells | Stored delta L2 |
|---|---:|---:|---:|
| attention q | 60 | 29,130 | 0.3656089733 |
| attention k | 60 | 29,780 | 0.5095248653 |
| attention v | 50 | 25,112 | 0.4386809008 |
| attention o | 60 | 29,544 | 0.4460823329 |
| MLP gate | 60 | 29,911 | 0.3915411227 |
| MLP up | 60 | 29,745 | 0.3742014066 |
| MLP down | 60 | 30,197 | 0.3895058159 |
| Total | 410 | 203,419 | 1.1089462630318312 |

The changed coordinates occupy 406,838 stored bytes. Their AV norm alone is
7.209352966810595, giving a changed-coordinate relative norm of
0.15382049791944463. This ratio must not be called the full-matrix or full-model
relative change. No native scalar/vector changes were present. Detailed
per-layer and per-tensor values are retained in the manifest summary. Magnitude
and projection name do not prove responsibility for an ability.

After full-diff completion, require complete coverage, all 1,188 compared
tensors, no incompatible/missing/extra tensors, all file hashes matching the
verified AV/F02 identities, no input stat changes, 203,419 stored-bit changes,
and delta L2 agreement with the saved manifest. Record actual bytes, elapsed
time, completed numeric/hash pass counts, configuration equality, and output
SHA256. A failure or discrepancy remains a failed trial record and must not
be hidden by a replacement scan.

## Existing activation comparison: faithful limits

The original AV merge is `N/merged.json` (2,890,586,309 bytes). The F02 owner
will produce `N2/merged.json` after the complete 778-item canonical run and
writer closure. Reuse these merges and the owner's per-call audits; do not
repeat large capture scans or change the canonical item set.

The capture sampling is the first eight unique prefill positions and sixteen
specified decode offsets, with token bins of 64. It is a sample of execution,
not full-token activation coverage. The exact offset set is
0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384. A bin can contain
several requested positions; bin start/end/count alone cannot establish the
exact sampled position set.

Existing merge keys retain item, request, site, phase, token kind, accepted
state, replica, rank, and token bin. Call/trial/request scope and metric sample
counts are retained. Accepted true, false, and unknown are distinct. MTP
acceptance uses the matching replica/step/request acceptance record; it is
sampler acceptance, not proof that the token survived public-response EOS
truncation. Missing records are not zero-filled.

`observe compare` only combines the two run arrays and rejects duplicate run
IDs. It does not establish cross-model call or token correspondence.
`observe view` supports baseline/candidate views with separate request,
rank, site, and acceptance scopes. Its strict paired-delta mode additionally
requires an explicit identical pair_key, accepted=true, compatible task/site/
phase/token-position/sample counts/runtime, and other matching metadata.

The existing live merges do not provide pair_key, and the saved runtime
metadata lacks the separate vllm_version field required by that strict
pairing guard. Therefore the unmodified HTML will correctly refuse paired
numeric activation differences. Do not synthesize a pair_key, infer one from
call order, copy a missing version into the artifact, or treat an empty diff
as zero activation change. Preserve that limitation alongside the side-by-side
counts and metrics. The HTML excludes rejected records from its plotted views;
accepted-only is an additional filter. Unknown acceptance remains explicitly
unknown when shown. The owner's audit retains rejected and unknown counts.

For a meaningful prefill comparison, first establish the same task/source,
unambiguous corresponding request/attempt, identical full request-payload
SHA256 (not just visible prompt text), matching tokenizer/template/config and
runtime conditions, exact sampled prefill positions, TP rank/site, acceptance
state/mask, and sample counts. Request IDs and call IDs differ between runs and
must remain visible; they are not fabricated into matching IDs. Prefix-cache
execution may omit positions even for the same request. A descriptive
same-prompt comparison can then be discussed for those verified scopes, while
the current HTML remains in separate-run mode.

Matching a decode ordinal is insufficient: generated prefixes, draft paths,
rejections, and effective sequence lengths can differ between models. Label
such profiles as conditional observations with different-prefix uncertainty,
not a paired causal weight-effect measurement. Do not read or reproduce hidden
reasoning to establish alignment.

AV's merge status remains partial_or_running because two pre-run requests
were unlinked (5,760 records), despite both writers closing with zero
queue/drop/error/quota counters and activation for all 778 official items.
Retain this distinction, excluded-other-run counts, unmatched requests, and
per-request missing offsets. F02's closure and counts must be reported
independently; no AV counts are imputed into F02.

## Smallest existing offline command set after closure

Wait for the observation owner's full canonical result, successful merge,
writer/rank closure and exact sampling audit. In a new output path, preserve
the complete merged data; do not select only favorable tasks. The two commands
below are preparation only and were not run while F02 capture was active:

```bash
cd "$B"
test ! -e "$C/activation-comparison.json"
test ! -e "$C/activation-comparison.html"
./kairyu-bench wsp observe compare "$N/merged.json" "$N2/merged.json" \
  --output "$C/activation-comparison.json"
./kairyu-bench wsp observe view "$C/activation-comparison.json" \
  --weights "$C/weights-full-diff.json" \
  --output "$C/activation-comparison.html"
```

The weight association checks the full checkpoint identities against the
capture identities. The ongoing full diff keeps hashing enabled for this
association. Store input/output SHA256, command options, actual times and
source hashes with the execution receipt. A large JSON/HTML build is CPU/RAM
work; it is not an additional benchmark. The HTML stores compressed task
payloads and lazily loads them, but the Python compare/build still loads full
input structures. Budget memory and disk before execution.

Use the existing `docs/experiments/scripts/fresh-capture-code-audit.py` outputs
for per-code-item request/call IDs, full-input hashes, seed/usage, exact
prefill/decode position sets, rank and acceptance counts. The original audit
and its v2 correction remain separate preserved files. Its numerical profile
is only the predeclared decode offset 64 across 60 layers and three roles;
the exact prefill position sets are identity evidence, not a newly extracted
per-position prefill numerical profile. Prefill bin metrics can be inspected
in the canonical merged report after confirming the exact constituent
positions. No new comparison tool, raw prefill-value scan, or artificial
alignment is introduced in this preparation.

## Source evidence

| Source | SHA256 |
|---|---|
| wrapper `kairyu-bench` | `56b53b7d72d75efcb2bef24643322f73625eba5c08a645c2f847768e558b6176` |
| weights/cli.py | `4193ee3863ac8d96ead9ba167ac572619d280ed9277f5729df970361cb882b26` |
| weights/statistics.py | `7c021ac29e0e6bdbcb5a0bc150a598eeef7c9d1a39baf3931525d849ca0370f0` |
| weights/checkpoint.py | `898297c05e7179cdc4568d6f290b978073bcebafea804ac043b54030e8205bf9` |
| observe/cli.py | `3bfbb7e6dd2fe0e89fbd2b2211e1688a9e56daa704bde6575f8e229e51042b9f` |
| observe/core.py | `363fc7e33bcb22b20d846171892366fa067bdb7f3062cd35e4b8e616dba2849d` |
| observe/report.py | `eb3f44c74034a2087d78a8941823ef5e7f3a82ec1be5d8f566dc01d462258258` |
| observe/tracing.py | `02b21d991f2d3be7a4f5bd648857e9a887bf1fd7dd7984245b5244ed96ee27b4` |
| fresh-capture-code-audit.py | `4f73da8d944875617c3bbeda0588e19c0e63e48c44f6b7ad2e884ac470885d5b` |

Implementation locations: statistics.py:311–459 (full diff, coverage, norms,
hash identities); core.py:149–348 (merge and acceptance) and :350–357
(compare concatenation); report.py:280–400 (request scopes, strict pairing,
acceptance filtering, weight association); tracing.py:91 (full payload hash).
These source hashes, rather than documentation-only Git HEAD changes, define
the inspected implementations.


## Completed F02 capture and metadata-only comparison

The canonical F02 dev10 subsequently completed all 778 items. Code source
average is 87.777778, tied with AV; all 26 code item scores are unchanged.
The frozen advancement rule leaves AV as the parent. Capture limitations are
separate from the complete official scoring result.

The F02 owner restored ordinary e275... healthy and completed its run owner.
Both F02 writer closed flags remain false (0/2 acknowledgements), although the
queues/drops/errors were zero and the owner's single audit confirmed gzip EOF
and activation-plus-acceptance record counts against each status file. All
778 official items and all 1,590 client traces had both TP ranks represented.
That coverage does not prove every scheduled activation or an inferred count
of backend calls in multi-call trial scopes. Two unlinked requests remain.
Shutdown logs document abort timeout and forced engine/worker termination;
zero queue counters are not a substitute for the missing close acknowledgement.
AV's two writer acknowledgements are true. Both captures retain their partial
status and all anomalies.

| Existing capture count | AV | F02 |
|---|---:|---:|
| Activation records, all scopes | 7,001,280 | 6,958,800 |
| Official-run activation records | 6,977,160 | 6,934,680 |
| Excluded other-run records | 18,360 | 18,360 |
| Unlinked activation records | 5,760 | 5,760 |
| Official items with activation | 778 | 778 |
| Code items retained | 26 | 26 |
| Writer closed acknowledgements | 2/2 | 0/2 |

F02 merged.json SHA256 (owner-computed):
a56a03136c7da8b1bcd0b25720885a82ac030802397477c4637c8ae2e83ebdf4.
AV code-capture-audit-v2.json SHA256:
3c5ce3756ffb966ee13ad58721af6fd498f3c317db629c9376cd2e7e65b28360.
F02 code-capture-audit.json SHA256:
b47de1e83b4451fd4beed2219e7a4f899b97c6431e212ec5b922211c6a651188.
The F02 audit recipe SHA is
83e4ba37a34d47cacd7ad2421e2e3eb10e9f0c75ac4d5a0b9bb8715624ded2e8;
it adds writer/trace accounting to the retained code audit. The earlier source
hash table is the initially inspected version, not a claim that this follow-up
used an identical whole script.

One metadata-only follow-up loaded these two SHA-checked audit JSON files and
required exactly the same 26 code item IDs. Every item has exactly one call
per run; both original request and call IDs are retained. All 26 full payload
input hashes match. All 52 item/rank combinations have identical exact prefill
position sets and identical prefill acceptance-state positions. These are
eligible same-input metadata scopes for later descriptive prefill inspection,
not newly computed numeric activation deltas or fabricated pair keys.

Decode offset sets match in 48/52 item/rank combinations. The four differences
are retained below without imputing missing offsets. Even the matching sets
do not prove an identical generated prefix. The follow-up retains accepted,
rejected and unknown states separately, and separately records the state counts
for the existing offset-64 profile. It does not reread raw activation streams,
extract new prefill numerical profiles, or read generated reasoning.


~~~json
[
  {
    "item_id": "livecodebench:abc393_d",
    "rank": "0",
    "av": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048
    ],
    "f02": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048,
      4096
    ]
  },
  {
    "item_id": "livecodebench:abc393_d",
    "rank": "1",
    "av": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048
    ],
    "f02": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048,
      4096
    ]
  },
  {
    "item_id": "livecodebench:abc394_b",
    "rank": "0",
    "av": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512
    ],
    "f02": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024
    ]
  },
  {
    "item_id": "livecodebench:abc394_b",
    "rank": "1",
    "av": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512
    ],
    "f02": [
      0,
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024
    ]
  }
]
~~~


The result is C/activation-audit-side-by-side.json, SHA256
c01522d28b645aac7e0c406fd4a945dbf747f592bb2d0c2761dc524a8ffbbfb7.
Its method is a direct dictionary join on the fixed item ID set, requiring
unique single calls; compare the saved full input hashes, seeds and prompt
token counts, then compare exact per-rank prefill positions and their
phase-filtered acceptance-state positions. Preserve each side's decode set
and offset-64 acceptance counts; compute no metric subtraction. Nonunique calls
would be reported as ambiguous rather than matched by order. All 26 items,
including the three official failures, remain in the artifact.

Copies of the manifest-only summary, exact full-diff launch plan and all-code
metadata comparison are included under records/2026-11-07-av-f02-comparison/.
The full weight-diff result is still pending; no projected time is reported as
an actual successful scan time.


## Full saved-weight diff completed and independently closed

The single low-priority CPU diff completed with exit 0 in 3,045.507109 seconds
(50 minutes 45.507 seconds outer elapsed; the diff implementation reports
3,045.035457 seconds). This exceeded the initial approximate 40-minute numeric
projection. Later sampled read-rate projection was approximately 48.5 minutes
for numeric work before the final hash pass. Projections remain labeled as
estimates; the completed measured time is the number above. CPU utilization
was 99% of one thread, maximum RSS 255,488 KiB, swaps 0. There were no model
calls, GPU contexts, service changes, repeated numeric scans, or substitute
sampled denominators.

Complete numeric passes: two (one AV and one F02). Complete identity hash
passes: two (one per checkpoint), covering 18/18 file hashes. Actual logical
I/O exactly matched the estimate: 62,546,177,752 numeric payload bytes,
62,578,656,419 identity-hash bytes and 280,758 metadata bytes per checkpoint;
250,250,229,858 bytes total. Logical I/O is not the physical disk-read amount.
The final time receipt records filesystem inputs separately.

All 1,188 stored tensors and 31,273,088,876 stored floating elements were
compared; no missing, extra, incompatible or nonfloating tensor existed.
All elements in both checkpoints and the delta were finite. Both checkpoint
trees and all 18 file hashes matched the earlier independent saved verification.
Every field of the fresh nine-file stat inventory on each side matched the
pre-run snapshot. All seven pinned source files were unchanged; every
recognized configuration/tokenizer file compared equal.

The full comparison found exactly 203,419 stored-bit and finite numeric changes
in the same 410 text projection matrices. Delta L2
1.1089462630318312 equals the saved BF16 manifest value exactly. There are
31,272,885,457 unchanged numeric elements. Native scalar/vector and other
stored tensors have zero changes.

| Denominator scope | Exact full AV L2 | Delta / AV L2 |
|---|---:|---:|
| All stored floating checkpoint entries | 97,779.12796904877 | 0.00001134133926192162 |
| Full 410 edited matrix tensors | 2,651.4095893171348 | 0.00041824781335177944 |
| Changed coordinates only (manifest summary) | 7.209352966810595 | 0.15382049791944463 |

The whole-checkpoint denominator is strongly affected by unchanged tensors:
vision_tower.std_bias alone contributes 48.399341% of its squared norm, and
several unchanged native layernorm vectors also have large norms. Therefore
the tiny whole-checkpoint ratio alone is not a useful claim that every local
change is tiny or that functional effect must be tiny. The full edited-matrix
ratio and changed-coordinate ratio describe different, explicitly stated
scopes. No tensor's magnitude or label assigns it a coding capability.

| Family | Full matrices | Changed cells | Full family AV L2 | Delta / full family AV L2 |
|---|---:|---:|---:|---:|
| mlp.down_proj | 60 | 30,197 | 1043.242376096 | 0.000373360807 |
| mlp.gate_proj | 60 | 29,911 | 1402.311083038 | 0.000279211316 |
| mlp.up_proj | 60 | 29,745 | 1402.010379424 | 0.000266903450 |
| self_attn.k_proj | 60 | 29,780 | 567.590240056 | 0.000897698426 |
| self_attn.o_proj | 60 | 29,544 | 762.275295075 | 0.000585198465 |
| self_attn.q_proj | 60 | 29,130 | 893.018390495 | 0.000409408112 |
| self_attn.v_proj | 50 | 25,112 | 555.701831530 | 0.000789417770 |


full-weight-summary.json additionally retains all 60 layer summaries with two
distinct denominator scopes (all stored layer floating parameters, and full
projection matrices), every changed matrix's exact shape/count/norm, and the
largest whole-model denominator contributors. These are reductions of the
completed full report; no checkpoint was rescanned to calculate them.
Only the original delta quantiles are sampled (up to 4,096 coordinates per
tensor, seed 0). Sparse sampled quantiles that happen to be zero do not
override the exact changed-cell count.

| Completed artifact | SHA256 |
|---|---|
| weights-full-diff.json (4,837,069 bytes) | cdc3ea38826640a25c680f05aa4df0f5e1e128e6e2c8c0ec39e1838607007ec1 |
| full-weight-summary.json | fd5d258b9ff3abc74ab10d4564e62f9937dc82ca74d0b00799775870c69448c9 |
| full-diff-closure.json | 5fd3a36c5087e89eb3fe4c7092f688747cc014fe63020ab67b224cd6d4f8a99a |

The full report, derived summaries, closure, execution and timing receipts are
included under records/2026-11-07-av-f02-comparison/. The closure cross-check
uses the prior SHA-checked full-verification-freeze-input.json file/hash table,
the original plan's full stats and source hashes, and the full output coverage,
finite counts and exact manifest delta. Per-family and per-layer summaries use
math.hypot across full per-tensor norms; they never average relative ratios.

The observation owner created a separate bounded four-task viewer while this
full diff was pending. Its v1 explicitly omits static weights and remains
preserved. The newly completed full report can be supplied to the existing
observe view --weights entry point for a distinct v2 artifact if useful.
That weight association does not repair the missing live pair_key/runtime
version or establish equal decode prefixes. No extra raw capture scan or
synthetic activation pairing was performed by this preparation.
