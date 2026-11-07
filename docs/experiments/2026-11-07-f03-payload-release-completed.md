# F03 lossless representation change completed

Root executed the approved operation once after all four ordinary replicas served F04. Only F03's two private safetensors files were unlinked; their complete contents remain reconstructable from immutable AV, the preserved exact spec, full edit manifest, original full verification, metadata and source archive. No trial, answer, result, spec, archive or nonweight file was removed. F03 remains a partial, unadopted trial: 2,338 records, 2,337 scored and one unknown.

| Step | Result | Elapsed |
|---|---|---:|
| Preview v1 | Stopped before marker/unlink: five existing nondumpable OS processes inaccessible | 0.551723 s |
| Preview v2 | Passed; zero checkpoint changes | 0.954138 s |
| Root apply v1 | Both private payloads represented by archive; postclosure true | 9.845617 s |

The first wrapper's all-process-visibility requirement exceeded the existing F02 boundary. Root authorized retaining the known limitation: four fixed OS process identities plus at most one current operator ancestor sshd can remain inaccessible. Unknown inaccessible identities or increases still stop the operation. Visible F03 references were zero, but inaccessible and other-UID references are explicitly **unknown**, never imputed to zero. Old source, failed receipt, diagnostic, source diff and final source are retained.

The unchanged mutation engine SHA is 435e01109887123db12519c100e304fa7bd4f5a71b772e6c96eedce1b130f894; final wrapper is c181b7291d542e961d7f6421ce5ae0661f3c0a122a7eb0a4cc046c4690ddd3ae. Plan v2 is 7473e36c8948981fbb020c4cbf3917f7b98f4474b9d4fca25a6cea4f04fba964; preview receipt addb6bf51642dd9a8e5d0df5483ca20351799da92844b571c5992895d0ed374a; apply receipt 601464cd0002841809d96890db1a20010964a5adf72542e1b60735fd2a1b07cf.

Closure protected 132 archive checksums, AV's nine-file stat continuity, F03's seven retained metadata files, all 27 container identities and 23 unrelated services. All four F04 replicas were healthy and idle; every Docker mount was disjoint from F03. Both targets were owned regular private nlink=1 files with exact saved inode/stat and no-follow directory-FD checks. Durable ARCHIVED.json preceded each per-file intent, unlink, directory fsync and completion journal. Its final marker prevents loading the archived directory as a saved checkpoint.

| Released F03 file | Logical bytes | Historical full SHA256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 49,923,154,850 | 48abfc21afab59b95aa9504d8848fd2dd6965ab3a04c9748af4a337bfe70112f |
| model-00002-of-00002.safetensors | 12,623,183,414 | 8110662f1c1c7386a23478e1e1d33259bc4f99b5d6f115780641dea04eb3a700 |

Logical release was **62,546,338,264 bytes**. Observed free bytes increased from **7,263,371,264 to 69,809,655,808**; filesystem delta differs slightly from logical size. No payload rehash, physical reconstruction, model call or service action occurred. Four focused guard tests passed in 0.038 s; no new full suite ran.

[Historical preparation](2026-11-07-f03-payload-release-preparation.md), [archive/reconstruction evidence](2026-11-07-f03-lossless-archive-preparation.md), [complete records and packaging source](records/2026-11-07-f03-payload-release-completed-v1/), and [final wrapper](scripts/fresh-f03-payload-release-20261107.py) remain. Large JSON and the source diff use deterministic gzip; original bytes remain on NVMe and packaging records raw/stored SHA256. This is not a claim that physical reconstruction was attempted.

The operation used the final wrapper with --plan plan-ready-v2.json, the exact plan/wrapper pins above, --output lossless-release-apply-v1, --apply, --approved-plan-sha256 with the same plan SHA, and --approved-operation-sha256 with the unchanged engine SHA. Absolute paths are fixed in the archived plan. Do not repeat this completed operation.

Minor documentation-only failures: one optional source-search filename was absent; one pre-SSH Markdown write had JavaScript quoting syntax failure and made no remote changes before correction. Neither was a model trial or additional release attempt.
