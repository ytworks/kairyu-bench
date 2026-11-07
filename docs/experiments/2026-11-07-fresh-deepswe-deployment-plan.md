# Planned homogeneous four-replica final DeepSWE run

Status: **final DeepSWE plan only**, updated 2026-11-07 after the separate F02 single-replica incident and recovery. No final DeepSWE run, homogeneous four-replica switch, Wiki edit/push or HF access has been executed under this plan. A read-only HTTPS Wiki fetch and ancestry follow-up is recorded below. The current observation owner holds gemma-0; no final deployment may begin before that owner completes, drains writers, restores ordinary serving and releases its lock. The final checkpoint and run name remain unbound. Earlier inspected states and source hashes below are historical snapshots, supplemented by the incident amendment.

The actual chat/gateway Compose labels identify the owner as
`/home/y-takagi/kairyu/examples/gemma-4-31b-8gpu`. That repository's AGENTS,
CLAUDE, PROGRESS and imported rules were read. Its current control is reused;
this plan adds no competing service implementation or framework change.

## Why not call the full example lifecycle

`CHAT_CHECKPOINT_DIR=... ./run.sh up` is the documented full-example entry point,
but it is not a chat-only operation. `control.py:1475` calls preflight, model and
reranker checks, then `docker compose up --build --detach --wait` with no service
filter, validates serving, updates the global checkpoint record, and provisions
the Chat UI. It requires WebUI admin configuration. That may rebuild/reconcile
the gateway and other services and updates UI functions/tools. It is broader than
the necessary final-model switch.

The old experiment `runtime/switch_checkpoint.py` reads admin environment values
from the UI container and calls that full entry point. Do not use it for this
limited deployment. No credentials are needed for the existing chat-only helper.

The PR already includes
[`redeploy-replica.py`](scripts/redeploy-replica.py), whose current CLI was checked
with `--help`. It imports the actual owner `control.py`, renders that owner's
Compose, and changes one declared `gemma-N` service using:

```text
docker compose ... up --no-deps --force-recreate --detach --wait \
  --wait-timeout TIMEOUT gemma-N
```

It pins the already-present image by content ID, preserves the existing command,
environment, CPU/GPU allocation, healthcheck, IPC and non-checkpoint mounts, and
allows only the checkpoint mount and tree label to differ. Selected chat services
have no build stanza; the helper does not request a build. It verifies an idle
target before recreation and attempts target-only rollback if recreation or
verification fails. Full environment values are kept in memory, not reports.
`_compose_env()` alone is insufficient: its fallback CPUSET is `0`; the helper
explicitly restores each current chat CPUSET from its baseline snapshot.

The helper is strict about preserving observer flags. Therefore it correctly
refuses to render the currently instrumented gemma-0 as ordinary serving. First
let the observer owner use its ownership-bound restore; do not circumvent that
check or operate its backup container. The helper's locks are scoped to its own
baseline paths, so using a new path does not supersede another owner's lock.

## Current inspected state and immutable source references

At the initial inspection before F02, the chat replicas were healthy but served **different** checkpoints (historical snapshot):

| Replica | GPU pair | CPU set | Current role |
| --- | --- | --- | --- |
| gemma-0 | 0,1 | 0-15,64-79 | AV, active observation deployment |
| gemma-1 | 2,3 | 16-31,80-95 | AD |
| gemma-2 | 4,5 | 32-47,96-111 | original |
| gemma-3 | 6,7 | 48-63,112-127 | AR |

All four use image `sha256:af9e746b913088cbf6c4464a31d6904a6b7e1d3ba62f0895362ddf13595f7abe`.
Their ordinary flags are TP2, BF16, context 262144, memory utilization 0.86,
prefix caching, max sequences 64, batched tokens 8192, async scheduling,
custom-all-reduce disabled, original MTP assistant with four speculative tokens,
Gemma tool/reasoning parsers, and native thinking enabled. gemma-0 additionally has
the owner's capture worker/middleware/request-ID flags until restoration.

The gateway is the existing four-replica pool, with current network address
`172.26.0.9` on `gemma-4-31b-8gpu_default`; derive and revalidate its address at
launch rather than hardcoding this observation. The embedding, Docling and UI
containers have older Compose owner-path labels; do not reconcile them through
the new owner during this switch. Record their container IDs, image IDs and start
times before and after to prove they were not recreated.

The global `model-volumes/gemma-4-31b-8gpu/chat-checkpoint.json` still names old
h0397, despite the actual mixed mounts. It is stale and must not attest the final
model. The single-replica helper intentionally does not update it. After all four
replicas pass a homogeneous closure, archive the old record and atomically write
the verified final identity using the owner's record schema, with separate
before/after evidence. Do not label a mixed intermediate state as globally served.

| Source | SHA-256 |
| --- | --- |
| Owner control.py | fc9a78bc232f9cc331926a0737e64f4b4fe14946574f4a02bc1aacb9a800b851 |
| Owner compose.yaml | fe49b81c153f68335ef46e07dc9f9910ee49ee066902e703d11d152c8809c5cb |
| Owner example.json | 99b1863206205abc92f5bde917c607cced03a220bd77323cf264eb781f4aedb7 |
| Owner kairyu.yaml | a28e232f27b89d1acc35e3f5871ccffafd8145211e4ffcb33cc0c6b514fa640d |
| redeploy-replica.py | da3dc93dcd1f7c44bbd64c9a0b65403424d436bef984d7964e0f50aee9aecc02 |
| Benchmark wrapper | 1fa56a0f80dd77872d4a9cd43cf023afa00c35d33f0e6e2e452329967fcb6220 |

These are inspected-source identifiers, not assertions that the future execution
will use unchanged files. Recheck at binding and retain the execution identifiers.

## Minimal switch after final selection

1. Bind one final saved checkpoint, its editor spec/manifest, successful independent
   verification, complete file inventory and canonical tree. Freeze all final
   choices before DeepSWE. Confirm full model/config/tokenizer/template integrity,
   NVMe headroom and no subsequent writes. The helper's freeze path requires the
   exact checkpoint path, all file hashes, complete inventory stats and tree;
   distinguish reused hashes plus stat continuity from a new full payload hash.
2. Wait for the observer owner to restore gemma-0 to its ordinary state. Confirm
   all four command/environment/mount sets are clean, with no capture or sparse
   overlay active. Coordinate exclusive evaluation ownership. Check gateway
   outstanding counters and all vLLM running/waiting counters; inspect active
   benchmark processes. The helper's instantaneous idle check is not a global
   traffic drain or protection against an uncoordinated new request.
3. Create a fresh all-chat baseline with the existing helper, retain rollback
   identities and snapshots of the protected gateway, two embeddings, UI, Docling
   and other active services. Preview each service. Use the same baseline for the
   four previously unchanged targets, with distinct output directories. Apply
   sequentially and verify each receipt before proceeding. Do not dispatch a
   benchmark through the mixed gateway between replacements.
4. After each apply, compare protected-service IDs/start times and untouched chat
   identities. The helper validates its selected service, not every unrelated
   container on the host. If a step fails, keep its failure/rollback evidence and
   do not continue to benchmark. Earlier successful replacements remain changed;
   they require deliberate owner-coordinated rollback with fresh baselines if
   abandoning the switch. Never run broad `down` or stop embeddings for this.
5. Require all four read-only checkpoint mounts, canonical tree labels and full
   checkpoint proof to match the frozen final model; ensure normal flags and
   image IDs match the intended common configuration. Verify health and model
   identity directly on every replica and through the retained gateway. Only then
   update the global checkpoint record and start the final full benchmark.

Invocation shape, to bind later after the above requirements (not executed):

```bash
PYTHONPATH="$BENCH_REPO/src" "$CPU_PY" \
  "$BENCH_REPO/docs/experiments/scripts/redeploy-replica.py" \
  --control "$EXAMPLE_DIR/control.py" --service gemma-0 \
  --candidate "$FINAL_CHECKPOINT" --freeze "$FINAL_FREEZE" \
  --baseline "$FINAL_EVIDENCE/all-chat-before.json" \
  --output "$FINAL_EVIDENCE/gemma-0-preview"
```

The corresponding apply uses a new output directory and `--apply`. Repeat for
`gemma-1`, `gemma-2`, `gemma-3`, always with that same final checkpoint/freeze.
Preview is not output-free: it creates baseline/receipt evidence but never
recreates a container. No preview/apply was run for this unbound plan.

## Full 32-worker, 113-task, one-round command

The existing dedicated DeepSWE daemon answers on the NVMe socket and currently
has no task containers, 119 images, and ID
`0b409923-4529-4688-aefe-501fd8d5ed34`. About 132 GiB was free on NVMe at the later
inspection. These are snapshots, not launch clearance. Retain the dedicated
cache/results same-path mounts and check image digests against the original run.
Do not prune either the host daemon or another evaluation's images.

Bind `FINAL_RUN_ID`, `FINAL_BENCH_IMAGE` and the attested gateway endpoint before
running. This command chooses the default new-build path and explicitly unsets
immutable runner reuse; the alternative opt-in is described in the amendment below.
Record the resulting image ID and source hashes; an old tag's name does not pin
the built payload.

```bash
set -euo pipefail
BENCH_REPO=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/worktrees/kairyu-bench-weight-tools
RESULT_ROOT=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/results
: "${FINAL_RUN_ID:?bind the final unique run ID}"
: "${FINAL_BENCH_IMAGE:?bind a new runner image tag}"
: "${FINAL_CHAT_URL:?bind the verified homogeneous gateway endpoint}"
test ! -e "$RESULT_ROOT/$FINAL_RUN_ID"
test ! -L "$RESULT_ROOT/$FINAL_RUN_ID"
cd "$RESULT_ROOT"
unset DOCKER_HOST DOCKER_CONTEXT HF_TOKEN
unset KAIRYU_BENCH_RUNNER_IMAGE_ID
unset KAIRYU_BENCH_DEEPSWE_REASONING_EFFORT KAIRYU_BENCH_DEEPSWE_TASKS
unset KAIRYU_BENCH_DEEPSWE_CONTEXT_WINDOW KAIRYU_BENCH_DEEPSWE_MAX_TOKENS
export KAIRYU_BENCH_IMAGE="$FINAL_BENCH_IMAGE"
export KAIRYU_BENCH_CACHE_DIR=/mnt/nvme/kairyu/bench-cache/gemma-wsp-coding-20261104
export KAIRYU_BENCH_DOCKER_SOCKET=/mnt/nvme/kairyu/docker/gemma-wsp-coding-20261104/run/docker.sock
export KAIRYU_BENCH_DOCKER_NETWORK=gemma-4-31b-8gpu_default
export KAIRYU_BENCH_CLEAN_TASK_IMAGES=1
export KAIRYU_BENCH_DEEPSWE_WORKERS=32
export KAIRYU_BENCH_DEEPSWE_ATTEMPTS=1
export KAIRYU_BENCH_DEEPSWE_RETRIES=3
export KAIRYU_BENCH_DEEPSWE_API_TIMEOUT_SECONDS=3600
export KAIRYU_BENCH_DEEPSWE_AGENT=mini-swe-agent
exec "$BENCH_REPO/kairyu-bench" run "$FINAL_CHAT_URL" \
  --only deepswe --run-id "$FINAL_RUN_ID" --results-dir "$RESULT_ROOT"
```

The requested dedicated privileged arrangement is already within the user's
explicit benchmark authorization. Host root/sudo are not used. Do not invoke the
old stateful final-run launcher. The cleanup setting above follows the current
repository instructions; a historical candidate used cleanup=0 after prefetch.
Record this operational difference, image provenance and cleanup results rather
than concealing it. It does not authorize removing unrelated cached images.

## Original R3 parity and completion evidence

Original R3 is `gemma4-base-c32-thinking-20261104-r3`, completed 1/113. Its historical
thinking guard proves four original-tree labels, mounts, readiness, native
thinking on and no additional thinking-token cap; it does not prove a numerical
reasoning effort or an independent full rehash at launch. Its serving image ID is
not recorded there. Exact historical binary parity must not be claimed without
additional evidence. Prefer a fresh original comparison when material conditions
cannot be established, using the same homogeneous four-replica settings and full
113 tasks/32 workers/one round. That baseline measurement is not a weight-editing
restart: the selected cumulative edited checkpoint remains unchanged.

At launch, compare `selected.json`'s unique `problem_ids` and `trial-plan.json`'s
`slots` against R3: 113 IDs, 113 slots, every repeat index 0, one result path per
ID. The SHA-256 of R3's sorted ID list, JSON-encoded with separators `(',', ':')`,
is `856a54fff453bd4b2daac1e6dd4f8b4943ac401a340846b5ccdf7bfaca2b0c4a`.
Match dataset revision `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`, mini version
2.4.6/revision `a83fcae82d2a08f0ee0c688f9d137b3566c097f8`, source revision, official
scoring, package/agent lock, all agent configs and all actual task image digests.
Current adapter SHA `a3ebcf772fba4c2aabe365c5346120df3c57367714129dc81f2092c15c868ca3`
and lock SHA `4cb499a6388e1cc6f1b14eb0124725014ac4b8e0a10db5390accae990a44566e`
match R3's recorded values.

Also match workers32, attempts1, retries3, HTTP timeout3600, task timeout1,
setup multiplier4, mini agent/Chat Completions, server-default effort, and no
new mini token/context/step limits. Keep all 113 outcomes even when unscored.
Every minute report scored/planned, passed, active tasks, steps, elapsed time,
retries/failures, API/daemon health and capacity. On completion cross-check all
canonical UUIDs/slots/jobs, official rewards, run/normalized/report statuses,
cleanup and final checkpoint/serving continuity. Save paired problem-level gains
and losses; do not infer independent HTTP counts from agent steps.

## Fresh Wiki ancestry and append-only publication plan

The existing Wiki checkout has no unstaged changes and HEAD
`9a5335927336e339ebaccc272110e52ca61d1369`, but its tracking branch is stale and
reports ahead4. A fresh read-only probe through its SSH origin failed with
`Permission denied (publickey)`. An HTTPS `ls-remote` succeeded and independently
returned that same remote master HEAD. No credentials were extracted, and no
fetch/push was attempted in that initial probe. The later fetch-only follow-up below supersedes the stale tracking comparison.

After the final full result exists, obtain a fresh remote snapshot over the
working HTTPS transport in a new NVMe checkout, or fetch to a distinct tracking
ref. Check both ancestry directions against the intended parent and inspect
changes in `DeepSWE-Comparison.md`; do not push the apparent ahead4 blindly or
force-push. Append a new uniquely named final-run section with checkpoint tree,
conditions, 113 problem outcomes, aggregate score, steps/time, failures, baseline
parity limitations and the exact HF decision. Preserve previous records.

Immediately before publication compare the remote HEAD again. If it advanced,
base the addition on that new HEAD and repeat the content/ancestry check. Use a
standard existing authorized Git authentication flow without printing credentials;
SSH failure is not evidence that HTTPS push authentication is available. Push
normally, then verify remote commit and page hash. Record this receipt in the
experiment and PR. If Git authentication is unavailable, preserve the complete
ready-to-publish section and report the specific publication blocker without
claiming the Wiki was updated.

## F02 incident amendment: enforce unambiguous Compose ownership

This amendment follows the completed single-replica F02 deployment incident and recovery, recorded in [the deployment audit](2026-11-07-f02-serving-preview.md). It is a plan for the later final evaluation, not permission to touch the current observation deployment. The active F02 observer owns the global lock; no current observer/retired cleanup or further model switch is performed here. The finally selected checkpoint remains unbound.

The first helper version inspected the canonical container name but Compose selected by project/service labels. A stopped observer-retired object also retained those labels. The first F02 apply and its rollback failed; the ordinary AV container stopped, the stopped retired Docker object disappeared, and a never-started rollback object remained. Original AV was restored with the exact CID/configuration, the owned new object was attributed and removed after evidence preservation, and all host observation evidence remained. The precise Compose error was suppressed by that helper and is unavailable. A later Docker events query returned zero records and is not evidence of no events. Duplicate label ownership is supported by the preserved original inspect, pinned observer creation/restore source and successful CID receipts; an independent retired-object inspect was not saved.

The repaired existing helper SHA is 2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae. It enumerates **all** containers, including stopped/renamed objects, by exact Compose project and service labels. Before preview and immediately before apply it requires the sole matching CID to equal the selected canonical target. Rollback allows zero or one remaining member and rejects multiple members. It forces COMPOSE_REMOVE_ORPHANS=0 for apply and rollback. The second F02 preview and application passed with this guard, but that does not pre-clear a future final switch.

Add the following gates before the minimal four-replica switch above:

1. Let the observation owner complete evaluation, drain/close writers, retain its failures and unjoined records, restore ordinary serving and release the real global evaluation flock. Do not clean a container while its current owner is active.
2. Under exclusive evaluation ownership, enumerate project/service labels for **each of gemma-0 through gemma-3 before changing any one of them**. Each service must have exactly its intended ordinary CID. Canonical names alone are insufficient. Record all selected IDs, states, label sets and all other service snapshots.
3. If a stopped retained observer or other duplicate is present, stop the deployment plan before any recreation. Do not weaken the guard, hide it with a new baseline path, or use broad Compose down/remove-orphans/prune. Establish which completed owned trial created that exact object, preserve the owner receipts and its safe configuration/mount/label/log evidence, and verify all trace/results bind paths remain. Preserve meaningful container-only artifacts before any disposal. Names or matching checkpoint paths alone do not establish ownership.
4. A separate explicit cleanup decision may remove only an attributed, unused stopped object from a completed owned trial after evidence preservation. Keep the original ordinary serving CID, other original services, all checkpoints and every trial record. Use an exact CID and non-force operation after rechecking state/ownership; do not generalize a one-object authorization to older or unrelated objects. The F02 recovery only authorized its own never-started rollback object; that is not blanket authorization to clean future retained observers.
5. After any authorized cleanup, record the exact object loss, before/after snapshots and retained host evidence. Create a **new** protected-container snapshot and all-chat baseline/source plan. Keep old failure receipts and source hashes. Weight hashes with verified stat continuity may remain reusable, while changed deployment source must have a new explicit execution pin.
6. Re-run the helper preview for all four services with the repaired guard and orphan removal disabled. Recheck global ownership, exact single-CID selection, idle requests and actual source hashes immediately before each sequential apply. Recheck untouched services after each step. If failure occurs, preserve automatic rollback results; if that rollback fails, first recover the specifically affected ordinary service and audit impact before any new attempt.

At the final closure, independently require homogeneous read-only checkpoint mounts, exact canonical tree, serving image/flags and successful direct model/chat/logprob reachability on all four replicas, plus the unchanged intended gateway pool. No benchmark may use the mixed gateway during replacement. The original model comparison must use the same final evaluation conditions; historical parity limitations above still apply. No final DeepSWE or HF access was performed in this amendment.

## Immutable runner update

The old statement that the wrapper always builds described its initial inspected version. It now supports the explicit KAIRYU_BENCH_RUNNER_IMAGE_ID opt-in: a full lowercase sha256 image ID must resolve to exactly that local image, build is skipped, and run uses --pull=never. Mutable tags, short IDs, missing images and mismatches fail before runner startup. Normal invocation still builds from the current checkout. The command above deliberately unsets this opt-in to choose its documented new-build route; alternatively, select and attest one exact measured immutable runner for both final comparison arms. Do not accidentally inherit a prior WSP runner pin.

An immutable image still does not freeze bind-mounted cache environments, official source checkouts, dataset/task images or serving. Record those separately, including the 113-task identity/condition evidence above. Do not call an old tag, package lock alone or a progress/config digest a full binary-equivalence proof.

## Fresh HTTPS Wiki fetch and ancestry evidence

A later read-only follow-up actually fetched Wiki master using HTTPS with the existing gh credential helper configured, without displaying or extracting any credential:

    GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
      -c 'credential.helper=!gh auth git-credential' fetch --no-tags \
      https://github.com/ytworks/kairyu-bench.wiki.git \
      master:refs/remotes/fresh-https-20261107/master

The existing origin URL and working branch were not changed. An independent HTTPS ls-remote matched the fetched ref. Both local and remote HEAD were 9a5335927336e339ebaccc272110e52ca61d1369; both ancestry checks passed and left/right commit counts were 0/0. DeepSWE-Comparison.md had SHA61ab2abc8299c98a03636d79cd78d5451e0e41796089c6ddbf3fe03d7864be11 on both sides with no diff. Working tree, local HEAD and page bytes remained unchanged. The old origin tracking ref still described historical ahead4; it must not supersede this fresh comparison.

Receipt: analysis/fresh-final-wiki-ancestry-20261107-v1/receipt.json. It records the command, transport configuration, remote identities, ancestry/count results, page SHA and no-page-change/no-push status. Fetch/read success does not prove push authorization or privilege. No Wiki page was changed or pushed. After the full final result exists, fetch again into a fresh audit/ref, repeat ancestry and page comparison, append the complete result against the then-current parent, and use the existing HTTPS authentication flow for a normal push only at that result-publication stage.

Actual source recheck for this amendment: repaired helper SHA 2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae; current official wrapper SHA 56b53b7d72d75efcb2bef24643322f73625eba5c08a645c2f847768e558b6176. Future execution must recheck the selected source/image identities instead of relying on this document date.


## Later state update: complete AV switch and active dev30

The earlier mixed-service and stale-record descriptions above are historical.
After the user's explicit four-inference-container approval, all four ordinary
TP2 replicas were sequentially recreated onto AV, tree
`5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
All 23 protected unrelated containers retained identity; the combined closure
and direct/gateway endpoint proof are in the
[AV deployment record](2026-11-07-av-dev30-homogeneous-serving-plan.md).
The global served-model record was subsequently atomically reconciled with AV;
[before/after bytes and receipt](2026-11-07-av-served-record-update.md) retain
the former h0397 record. No global identity is inferred from the old record.

The separate [complete canonical dev30 run](2026-11-07-av-dev30-dp4-run.md)
is active under owner PID 397808 and the global evaluation lock. It retains
all 2,338 items and the immutable f91 runner. No CaptureWorker is deployed;
its tracing is client HTTP only. Do not switch services or launch overlapping
GPU work while this owner runs. Final DeepSWE selection, matched-original
comparison, run, Wiki append and conditional Hugging Face publication remain
unperformed. This state update does not bind AV as the final model or claim
the coding target has been reached. Recheck all live state at the final handoff.
