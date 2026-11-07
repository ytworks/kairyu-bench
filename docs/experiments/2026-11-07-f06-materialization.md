# F06 saved BF16 materialization from completed LoRA training

The first CPU materialization failed on a private-copy permission boundary before editing tensors. The original attempt, source, traceback, elapsed time, and cleanup remain in [records](records/2026-11-07-f06-materialization/SHA256SUMS). The second explicitly authorized create and independent saved-model verify both completed. No capability gain is claimed.

The fixed inputs are F04, final step 64 (256 complete training microbatches), the FP32 rank-8/alpha-8 adapter, coefficient 1, and the existing dense merge arithmetic. This work makes no model requests or serving changes. The CPU job reads F04 and writes a separate private F06 directory, so root explicitly allowed it to overlap the training owner's independent same-container restoration; it does not acquire or alter that GPU owner's locks.

| Attempt | Result | Time |
|---|---|---:|
| v1 preview | Pass | 0.464 s |
| v1 create | PermissionError opening private copied shared.safetensors with inherited read-only mode; existing failure cleanup removed the owned candidate directory | 120.143 s |
| v1 verify | Not run; no manifest published | — |
| v2 preview | Pass; identical preview output and 3 command argument lists | 0.520 s |
| v2 create | Pass; manifest published | 444.002 s |
| v2 verify | Pass; exact BF16 regeneration and all source/candidate hashes match | 297.805 s |

The original file was copied with cp --reflink=auto --sparse=always. Its read-only mode was inherited. The minimal fix verifies the new touched shard is an independent inode with link count 1, then adds only its owner-write bit. It never chmods the parent or a shared link. Mathematical merge code, factors, training, and coefficient remain unchanged. All 9 existing recipe tests passed, including a new read-only-parent create/verify regression that checks parent device/inode/mode/size/mtime/ctime/link count/owner/group and bytes remain unchanged. Ruff and whitespace checks passed. The focused suite took 3.577 s (3.761 s subprocess wall time); no extra full suite was run.

Initial capacity was 69,026,975,744 bytes against a required 68,014,454,366 bytes: the full private copy plus 5 GiB reserve and 64 MiB workspace. The second attempt recomputes capacity and pins the empty owned bundle directory left by v1, while requiring the checkpoint and manifest paths remain absent. It preserves the failed attempt rather than overwriting it.

The artifact directory contains the exact operations and prepared plans, original and fixed source, regression source/output, all v1 stdout/stderr, source/stat evidence, and compressed diff. The final training factor SHA is f51ac4a9f95babe0e5651f6f3bbfdfdbb35cc5f1a4bca54d9d63f21f48571644; the merge spec SHA is 44e1928332dc347bfe19048b93ddb837e79f79797d2db807e6230bb52ce78d8d. Large factors and checkpoint payloads remain on NVMe, referenced by identity, not duplicated into Git.

Direct-code supervised loss can improve answer formatting or suppress reasoning without improving task-solving ability. Neither the training loss nor the saved-weight difference is a WSP score. The completed F04 baseline remains the comparison; no baseline repeat was performed.


The saved manifest reports 8,193,921,391 changed BF16 values across all 410 matrices. This is 27.9783% of the exact 29,286,727,680 text-projection elements. Saved difference L2 is 3.7080033175; its ratio to the same-scope parent L2 2651.4095893 is 0.001398502643. This denominator covers the 410 projections, not the entire checkpoint.

The [record-only norm comparison](records/2026-11-07-f06-materialization/saved-retention-v1/report.json) joins every saved edit with the corresponding step-64 factor Gram norm. Intended continuous L2 is 3.8483755593, giving saved/intended norm ratio 0.9635242872. None of the 410 matrices vanished. Per-matrix ratios range from 0.688950 (layer 4 down) to 1.027301 (layer 58 output projection). The intended norm is the FP64 Gram norm of the real product of final FP32 factors; this is not a new exact reduction of every blockwise FP32 GEMM value. No additional checkpoint or factor payload was read for this comparison.

Direction cosine and rounding error are unavailable in the existing manifest: merge1 returns source square norm, saved difference square norm, maximum difference, and counts, but no cross product. The norm ratio is therefore not directional retention. If separately authorized, a minimal diagnostic could reuse validated dynamic F06 mapping, scale and coefficient with audit5 on parent blocks and the fixed coefficient in slot 0; other slots would be zero and unused. It would require another complete text-projection parent read plus factor products, approximately 58.57 GB, even if the completed verify and stat continuity avoid candidate rereading. No such diagnostic was executed, and no arbitrary cosine threshold is treated as a capability criterion.


The complete second operation took 742.881 s, including preview and metadata guards; all stage stderr was empty and the owner exited 0. Parent file stat continuity passed, candidate full tree is 0fc0ef29346c4d8a06bea8570dee96d8e0ba7038fee0d5aded8db57852474ec5, and the external manifest SHA is 22b0603cc2dc9a2060c76f8baf99c28eceac08b56116727173bc61a719b5c0d4. The verifier confirms every touched BF16 block by regeneration, every nonrecipe span by comparison, and all 9 source plus 9 candidate file hashes. Its internal before/after metadata snapshots matched. The separate deployment identity stores fresh post-verify POSIX stat with every file mtime/ctime no later than verify output; it does not claim that the internal snapshot had been separately persisted. Candidate files are all independent nlink-1 files.

The current complete record is [final closure](records/2026-11-07-f06-materialization/final-closure.json), with [full candidate identity](records/2026-11-07-f06-materialization/materialization-f06-v2/fullcandidateidentity.json), [verify receipt](records/2026-11-07-f06-materialization/materialization-f06-v2/verify-execution.json), and the [archive index](records/2026-11-07-f06-materialization/SHA256SUMS). No additional cosine scan, holdout NLL, parent repeat, or model call was performed. Root prioritized the authorized complete canonical WSP evaluation after this byte verification and norm audit; deployment and capability evaluation are separate from this record.
