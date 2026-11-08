# F02 private-payload archive recipe (2026-11-07)

**Prepared only: apply has not run.** Both private shards remain, ARCHIVED.json is absent and released bytes are zero. F02 remains unadopted; no score or candidate status changes.

The [dedicated recipe](scripts/fresh-f02-payload-release-20261107.py), SHA 435e01109887123db12519c100e304fa7bd4f5a71b772e6c96eedce1b130f894, reuses [F01's executed directory-fd/archive/stat approach](2026-11-07-f01-representation-change.md). It adds global flock, all27-container protection, per-file fresh guards and durable intent/released journals. It only accepts [F02 plan](2026-11-07-f02-payload-release-plan.md) SHA 9f758511d865d9444858bcf645746efced8f567246dcd3a4472b6c9b055f6a3a.

| Fixed target under checkpoints/fresh-code-joint-f02/ | Bytes | Historical complete SHA256 |
|---|---:|---|
| model-00001-of-00002.safetensors | 49,923,154,850 | e6144c1c2322cd2a05ced629c754c229ec530747470b3ba423a3c905b8d6e97c |
| model-00002-of-00002.safetensors | 12,623,183,414 | f472653dfcfe3218f33111b44e936e097a8ba01014b9b5376f33c3196980181b |

The sum is 62,546,338,264 logical / 62,546,374,656 allocated bytes. Preview free space was 71,026,036,736 bytes; projected post-release free is 133,572,411,392 bytes, subject to concurrent writes and open descriptors. No capacity was released by this preparation.

## Preview and guard evidence

The real preview passed in **0.780823s**: global flock, archive135 hashes/exact inventory, prior full verification/reconstruction proof, AV9/F029 full stat continuity, owned ordinary nlink1 targets, all27 running/stopped container identities/states/mounts, and zero visible same-UID payload references. It rejected equal/ancestor/descendant mounts. Other-UID and inaccessible/disappeared processes remain an explicit visibility limit. The preview PID exited and its lock rows disappeared. No fresh payload fullhash or physical reconstruction occurred.

[Preview receipt](records/2026-11-07-f02-payload-release-recipe/preview-receipt.json) SHA b260a0744ebd12b2ec1d5c2729f61b709f815e9e7860d1a549f09e7bdd7e2a7c. [Independent read-only review](records/2026-11-07-f02-payload-release-recipe/independent-review.json) found no blocking issue and made no further preview/model/service operation.

The [two focused CPU tests](../../tests/test_f02_payload_release_recipe.py) protect refusal before any operation without exact approval, and exclusive durable journals against overwrite. Initial 2/2 tests passed but a nonexistent TMPDIR silently fell back to remote /tmp for a tiny automatically-cleaned fixture. That environment deviation is retained in [validation](records/2026-11-07-f02-payload-release-recipe/cpu-validation.json), not described as NVMe validation. Tests now pass TMPDIR explicitly (invalid paths fail); corrected NVMe-only 2/2 tests and Ruff passed. Original source/test versions, receipts and the documentation-only parse failure are retained. No real unlink or new full CI was executed.

## Exact commands

Completed preview, from the worktree:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ionice -c3 nice -n15 /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python \
      docs/experiments/scripts/fresh-f02-payload-release-20261107.py \
      --output /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-f02/lossless-release-preview-v1

Only after separate root authorization, the concrete apply command would be:

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ionice -c3 nice -n15 /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python \
      docs/experiments/scripts/fresh-f02-payload-release-20261107.py \
      --output /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-20261107-f02/lossless-release-apply-v1 \
      --apply \
      --approved-plan-sha256 9f758511d865d9444858bcf645746efced8f567246dcd3a4472b6c9b055f6a3a \
      --approved-operation-sha256 435e01109887123db12519c100e304fa7bd4f5a71b772e6c96eedce1b130f894

Under nonblocking global flock, all guards rerun. Before unlink, ARCHIVED.json and its directory are fsynced with not_a_saved_model=true, mandatory AV/archive, all9 historical hashes and reconstruction instructions. Each file uses fresh checkpoint/archive/marker/source/all27 mount/reference guards → durable intent → nofollow open and exact fstat/stat → directory-fd unlink → directory fsync → durable released journal.

Postclosure checks AV9/F02 nonweight7 continuity, archive135 hashes and original stat, seven nonweight content hashes and all27 containers. The owned marker is finalized atomically; free-space delta is distinct from logical released bytes. AV, original model, all other checkpoints/results, all trials and every service remain outside the mutation scope.

An error stops without automatic retry, reconstruction or service action. The marker, durable intents, returned-unlink list and observed absent targets remain. An abrupt process/power loss can interrupt the final receipt; durable marker/intents are the recovery boundary, never permission to retry blindly. Actual apply still requires fresh guards and has not occurred.
