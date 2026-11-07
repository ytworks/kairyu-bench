# F03 saved checkpoint and fresh full verification (2026-11-07)

The explicitly authorized checkpoint `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/checkpoints/fresh-native-contract-f03` was created once from AV. CPU preview, one apply and one separate fresh editor verify completed; GPU/API/service/finite calls were **0**. This records materialization, not capability improvement or adoption.

- Exact source tree: `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
- Exact spec SHA: `e10018748de2795a8af03b98abb6e18362393a152c74f46b7a7892fe4975e878`.
- Candidate tree: `504d0914e8f7d2a4e51a0888e5692e2d1a158b82d3a77b600810a14bf7e33b63`.
- Actual differences: **60,473 elements / 112 tensors**, L2 **0.0043893681725189855**, matching the frozen F03 spec and independent audit.
- Apply: **205.122 s**; fresh full verify: **165.188 s**; operation total **371.949 s**.
- Reserve: **50 GiB**; final NVMe free **70,977,908,736 bytes**.
- All **9 files** verified; `verified=true` and `legacy_full_hashes_missing=false`. No failure or automatic retry was recorded.

The exact [operation recipe](scripts/fresh-f03-materialize-20261107.py) was retained in this PR and copied with its five required package files to a hashed, read-only NVMe source snapshot. It held the global `candidate-evaluation.lock` exclusively through preview, copy, verification and identity binding; normal release is recorded. The four running model replicas remained mounted read-only on AV; no serving action was made.

The editor checked all source and candidate regular files, including undeclared bytes and nonweight files. Candidate edited bytes reconstruct their original source hashes when the manifest's before-values are restored. The source tree recomputed from those fresh verification hashes equals the pinned AV tree.

## Candidate identity without an extra 63 GB hash pass

Complete `Checkpoint.identity(hash_files=False)` and full POSIX inventory were captured before and after fresh verify and required to match exactly. The identity uses the verify-produced candidate file SHA values with the matching post-verify sizes/stats, following Checkpoint's canonical tree serialization. Thus the content identity derives from a fresh complete verification under stat continuity; it does not claim a second independent payload-hash pass.

Canonical evidence: `/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-f03-native-v1/materialization/fullcandidateidentity.json` (SHA `b9e5dbd64520196507d1d53fae64b2609d5ea45559c78aee4a60cf701d6876b5`). It contains the nine file hashes, exact stat identity and derivation. Source and candidate all-file hashes are retained below.

| File | Source SHA256 | Candidate SHA256 |
|---|---|---|
| chat_template.jinja | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` |
| config.json | `e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1` | `e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1` |
| generation_config.json | `d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de` | `d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de` |
| model-00001-of-00002.safetensors | `0589dc8a7a3afba79fe2d8444221dae410db57ce1be33eb72d9be637d70fcf10` | `48abfc21afab59b95aa9504d8848fd2dd6965ab3a04c9748af4a337bfe70112f` |
| model-00002-of-00002.safetensors | `9be24bc9910dbbe54f3f46154189c729071fd56dc41331f4a97a6ad3e02b3b56` | `8110662f1c1c7386a23478e1e1d33259bc4f99b5d6f115780641dea04eb3a700` |
| model.safetensors.index.json | `390b7f9a0a502e4526cf20862d22954c911cc318d8aba2cdb745c6fbd98264f6` | `390b7f9a0a502e4526cf20862d22954c911cc318d8aba2cdb745c6fbd98264f6` |
| processor_config.json | `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c` | `32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c` |
| tokenizer.json | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| tokenizer_config.json | `9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633` | `9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633` |

[Complete materialization records](records/2026-11-07-f03-materialization/) contain preview, plan, apply/verify receipts and stdout/stderr, all source snapshots, before/after stat proofs, full candidate identity, normal lock release and checksum manifest. The editor manifest is gzip-compressed losslessly, with original/compressed SHA and round-trip proof. Its exact values were neither rescaled nor recomputed.

The native proxy's two unmet design targets and box-exceed counts remain in the [independent F03 audit](2026-11-07-f03-native-spec-audit.md); successful file verification does not turn those into passed behavioral requirements. No finite or WSP result is claimed here.
