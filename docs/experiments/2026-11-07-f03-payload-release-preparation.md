# F03 private-payload release preparation

Historical preparation status (superseded by the [completed release record](2026-11-07-f03-payload-release-completed.md)): **prepared, unbound, not executed at this stage**. F03's complete partial WSP record remains retained: all 2,338 records, 2,337 scored and one unknown. Its 90-item code score does not rank the incomplete all-domain run as complete. AV remains the completed comparison parent by root decision. This preparation neither changes that decision nor removes any trial, metadata or model file.

The exact targets are only the two private files under checkpoints/fresh-native-contract-f03:

| File | Bytes | Prior full SHA256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 49,923,154,850 | 48abfc21afab59b95aa9504d8848fd2dd6965ab3a04c9748af4a337bfe70112f |
| model-00002-of-00002.safetensors | 12,623,183,414 | 8110662f1c1c7386a23478e1e1d33259bc4f99b5d6f115780641dea04eb3a700 |

Potential logical release: **62,546,338,264 bytes**. Observed free bytes at preparation were 7,265,431,552; adding the logical target size is a projection, not observed released space. Current AV9/F03 9 full stat records match the archive; both targets are owned regular nlink1 files with inodes distinct from AV. No huge rehash/diff or physical reconstruction was performed.

The [wrapper](scripts/fresh-f03-payload-release-20261107.py) reuses the unchanged F02 engine SHA435e01109887123db12519c100e304fa7bd4f5a71b772e6c96eedce1b130f894. The wrapper adds fixed F03 paths, the archive's 132 entries and saved whole-file restoration proof, in-memory dev→device stat-key normalization, and mandatory all-four F04 identity/healthy/idle gates. All candidate archives, manifests, specs, seven metadata files, AV/original/other candidates and unrelated services remain protected. The [dependent archive](2026-11-07-f03-lossless-archive-preparation.md) remains mandatory, as does immutable AV.

Before every relevant engine checkpoint, the wrapper rechecks its own/helper/plan SHA, successful F04 manifest/verify and closure pins, F04's exact nine-file full stats (including shared inode/link count), exact 27-container membership, 23 protected CIDs, all four healthy ordinary F04 read-only mounts and zero active/waiting requests. The original engine rejects any Docker mount equal to, below or above F03, including stopped containers. Same-UID references are checked by open FD and mapped inode; inaccessible still-live same-UID processes abort. Other-UID process internals are not inspected, and this visibility limit is retained.

The engine obtains the global evaluation lock nonblocking, rechecks all 132 archive file hashes and 18 checkpoint stats, uses exact directory FD/no-follow paths, writes and fsyncs ARCHIVED.json before release, then writes a durable per-file intent before each exact fstat/stat/nlink checked unlink. It retains per-file completion journals and partial-failure receipts without automatic retry. Final closure verifies archive integrity, AV continuity, metadata hashes, target absence, marker and all protected services. The marker prevents treating an archived directory as a saved model. Some inherited diagnostic text says F02; the actual fixed target in this wrapper is always F03.

The [unbound plan](records/2026-11-07-f03-payload-release-preparation/plan-unbound.json) cannot execute. After F04 endpoint/normal-release closure, a fresh all-27 sanitized snapshot and all evidence are bound into a new plan-ready.json, status ready_for_exact_preview. The existing engine then previews with no approval flags. Root must separately authorize the final plan and both source identities before apply.

Future preview only, after binding:

    python docs/experiments/scripts/fresh-f03-payload-release-20261107.py --plan PLAN_READY --plan-sha256 FINAL_PLAN_SHA --expected-wrapper-sha256 72e5bdb4566652d5baeae5fb4d9ef71a02f3b5920951891a3f5167f63a39881c --output NEW_PREVIEW_DIR

A later root-approved apply adds --apply, --approved-plan-sha256 FINAL_PLAN_SHA and --approved-operation-sha256 435e01109887123db12519c100e304fa7bd4f5a71b772e6c96eedce1b130f894, preserving the wrapper pin. The latter identifies the reused mutation engine; the plan binds the wrapper separately. All output paths must be fresh under NVMe analysis. No N4 launch may overlap this locked operation.

Focused CPU verification: the existing fault-test module now has four passing tests (0.037s), covering unapproved engine apply, durable evidence non-overwrite, unbound/duplicate F04 CIDs and wrapper mismatch before delegate/lock. Ruff passed. [Validation](records/2026-11-07-f03-payload-release-preparation/cpu-validation.json). No new full suite, GPU/model call, service action, global-lock acquisition, live preview or payload mutation occurred during preparation.
