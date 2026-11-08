# F03 lossless dependent archive: payloads unchanged

A new archive is prepared and validated at analysis/fresh-purpose-20261107-f03-native-v1/lossless-archive-v1. It contains **132 checksum-listed files plus SHA256SUMS**, **37,752,648 bytes**, and took **2.196185 seconds**, including **1.406839 seconds** of selected-coordinate re-planning using the existing archived editor. Attempts 1, failures 0. No checkpoint fullhash/diff, physical reconstruction, model call, GPU use, lock operation or service change occurred.

**F03 remains a complete nine-file checkpoint and is currently served. No payload release, move, rewrite or unlink was authorized or performed.** Its two private safetensors files total **62,546,338,264 bytes**. The archive remains valid provenance if F03 is adopted or rejected; this operation makes no selection or capability claim. Existing trials and all benchmark results remain unchanged. The running WSP logs were not copied as completed records.

## Exact reconstruction dependency

The mandatory immutable base is checkpoints/code90-current-ad-helper-av-v1, tree SHA256 **5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf**. F03 tree SHA256 is **504d0914e8f7d2a4e51a0888e5692e2d1a158b82d3a77b600810a14bf7e33b63**.

The saved full editor verify passed all nine file hashes with verified=true and legacy_full_hashes_missing=false. Its source at editor.py:513–585 verifies each entire candidate file, exact declared after_hex bytes, and the full file with those spans replaced by before_hex. The latter hash must equal the full source hash. Thus the complement equals AV and the changed bytes equal the frozen spec. This is existing full-verify evidence, not a fresh 62.5GB payload scan.

Preparation matched all **18 complete stat records** (device, inode, size, mtime, ctime, mode, uid, gid, nlink) to the original fullcandidateidentity before and after archival. Both shard files remained private nlink1. The archived existing editor validated the manifest and re-planned all **112 spans / 60,473 changed values**, matching tensor, shard, offset, size, dtype, shape, original/changed bytes, selected/changed index lists, counts and L2 **0.0043893681725189855**. This is selected-coordinate checking, not a model forward or full-file diff.

The original spec and full manifest are gzip-compressed without JSON reserialization; both streams were inflated and their byte SHA verified. All seven nonweight files were copied and freshly hash checked. All nine AV/F03 saved file hashes and original/current full metadata remain in the archive. It contains all 25 other materialization records, proposal report/plan/parent state, the frozen 89-file existing package, and the exact one-off operation source. Five archived core editor/package source hashes match the materializer's source pins. No new general archive tool was introduced.

The archive is not a standalone model: AV must be retained. A future reconstruction into fresh NVMe paths must verify archive/base hashes, use the archived editor and spec, run full editor verify, and match all nine CANDIDATE-SHA256SUMS entries. Content bytes and relative filenames are reconstructable; new filesystem inode/ctime/device values are not promised. The [archived README](records/2026-11-07-f03-lossless-archive/archive-README.md) supplies commands, which were not executed here.

## Identifiers and operation record

| Artifact | SHA256 |
|---|---|
| Original spec | e10018748de2795a8af03b98abb6e18362393a152c74f46b7a7892fe4975e878 |
| Saved full verify | 616806dc095ebecb04ac10d4514ec6c73f9bbd53d3f572d1ef704452320062c4 |
| Original fullcandidateidentity | b9e5dbd64520196507d1d53fae64b2609d5ea45559c78aee4a60cf701d6876b5 |
| archive.json | 00ef50f95b783c25da078e4274e142471d9bfceefb34d78042fba5d83eb9a4e0 |
| archive-validation.json | 419f5477838de3db89d8b6b4d4e758c48e9bb935b9c76e03cb498e132ede3e52 |
| Archive SHA256SUMS | 374f40c074c4ea8c893d802217d31d1ef99987cb6d146016972790128aa3b520 |
| Exact executed one-off source | 2abde870d9d06075ed836536b449c8c5162cba921a0578d291e92687d38bdc0e |

The [source](records/2026-11-07-f03-lossless-archive/prepare.py.txt), [command](records/2026-11-07-f03-lossless-archive/command.json), [receipt](records/2026-11-07-f03-lossless-archive/preparation-receipt.json) and [checksum inventory](records/2026-11-07-f03-lossless-archive/SHA256SUMS) are retained in the PR. It used the existing CPU Python, nice15/ionice idle, no visible GPUs, one BLAS/OpenMP/MKL thread, disabled bytecode and a fresh NVMe tmp directory. Source, metadata and compressed records only were copied. Free NVMe bytes were 69,807,161,344 before and 69,769,011,200 after; concurrent WSP writes mean the difference is not attributed entirely to archival. Space released by this task: zero.
