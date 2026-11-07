# Saved finalist materialization and fresh dev90

These two host-bound recipes reuse the successful AL saved-checkpoint path:

- [Materialize one selected saved-parent increment](experiments/scripts/materialize-ad-finalist.py):
  derived from executed `materialize-and-deploy-current-ad-library-al-v1.py`,
  SHA `a20fc5efb905c2ea1a4c8d0ee0ac44c5378f9ace6bfc0933f98cb6a90ed2fe87`.
- [Launch fresh AD plus one or two saved finalists](experiments/scripts/launch-ad-finalists-dev90.py):
  derived from executed `launch-current-ad-al-dev90.py`,
  SHA `fc3de9942500b0ef8a5cd498ffe0a56a12ef14b95e7551cc095d062787ebca38`.

The AN and AR materialization bindings have now executed once each. Both saved
checkpoints passed independent full-file verification and the measured BF16
selected-word comparison, then deployed healthy to slots0 and3. The fresh
ordinary AD/AN/AR dev90 started on 2026-11-07 at 00:00:50 UTC; both finalists
were rejected after scoring, as recorded below.
The unchanged absolute workspace and existing operational helpers are part of
this machine-specific recipe; this is not a general service orchestration SDK.

The recipe accepts the recorded AN–AS, AV and BA labels, chooses target slot0
or slot3 through the materialization plan, and accepts the official plan via
`--plan`. The materializer checks the incumbent name and checkpoint against
`plan.parent` and the retained control against `plan.parent_service`; the
default control remains `gemma-1` for the historical AD bindings. A single official comparison can contain one or two distinct selected
slots plus fresh AD1. Source correspondence, syntax and Ruff are recorded in
`A/current-ad-functional-finalists-preparation-v1/source-validation.json`.
No model or full CI rerun is required to prepare these bindings.

## Bind after selection and restoration

The inert metadata templates are
`slot0-materialization-template.json`,
`slot3-materialization-template.json` and `dev90-template.json` under the
preparation directory. They contain unresolved fields and are not executable
plans. The root decision selects at most two actual candidate IDs/spec SHAs.
It must also bind the completed overlay owner, all four original services
restored, and fresh serving snapshots. Do not reuse the stopped-service
snapshots captured before the overlay screen.

Every candidate applies its exact BF16 spec to its plan-bound accepted saved
checkpoint. It never uses original weights or another finalist as its base.
One existing editor create is followed by one independent editor verify.
The existing full-file hash evidence, private-inode checks and measured selected
BF16 parity produce the freeze without a third payload hash pass.
Deployment replaces only the selected slot with rollback available; the retained
parent control, other chat services and embeddings are protected by fresh checks.

The materialization recipe supports `--save-only`, and otherwise includes the
existing deployment helper. No automatic cleanup or retry is introduced.

```text
CPU_PY materialize-ad-finalist.py --plan MATERIALIZATION_PLAN --plan-sha256 SHA
CPU_PY materialize-ad-finalist.py --plan MATERIALIZATION_PLAN --plan-sha256 SHA \
  --authorization AUTH --authorization-sha256 AUTH_SHA --apply
CPU_PY launch-ad-finalists-dev90.py --plan OFFICIAL_PLAN --plan-sha256 SHA
CPU_PY launch-ad-finalists-dev90.py --plan OFFICIAL_PLAN --plan-sha256 SHA --apply
```

These are invocation shapes, not execution authorization. The materializer
requires the exact root authorization and source/finite/screen proof bindings;
the official launcher requires the exact plan-bound root decision.

The ordinary dev90 protocol remains the archived `5e3f2df` source, tier30/dev,
HumanEval+30, MBPP+30, LiveCodeBench20 and SciCode10, with c32, logprob c8,
agent c1 and the same model-serving configuration. Each arm has an independent
numeric-only observer. Actual OCI/source82 capture and serving counters remain
part of the existing path. Final numerical closure and paired evidence are
performed by the existing benchmark owner, not an additional scorer here.
An exploratory overlay score does not replace a fresh saved-weight score.

## Storage budget

The historical AN/AR per-create reserve was **80 GiB**. A private candidate requires
62,578,656,419 bytes plus the existing 67,108,864-byte editor headroom:
**148,545,111,203 free bytes** before each create. Two copies are sequential;
there is no implicit relaxation of the reserve or automatic deletion. The later
AV and prepared BA bindings explicitly declare **75 GiB**; the unchanged editor
also requires 64 MiB of headroom, for **143,176,402,083 bytes** total before
creating a checkpoint with the same nine-file layout.

A read-only metadata survey found the completed previous authored screen's
private cache used 1,649,119,232 allocated file bytes (7,910 files, UID1003,
no symlinks or multiply linked files). Root removed that exact cache after its owner closed and fresh all-three-daemon
running/stopped mount and inventory checks passed (receipt SHA
`622e04883d384cd42d48f8002ed7a6811516a7c58e049382773be2b8f2968d74`).
After AN deployed healthy, root separately removed only the two unmounted
retired AL payload shards, 62,546,338,264 bytes (receipt SHA
`d183370da20917493af43ec990982e2016479f5199433f268aec1f654854ce38`).
All AL metadata/results and accepted ancestry remained. The active official
screen cache was excluded. Each create rechecked free space; no recipe performs
automatic deletion.

## Executed saved-weight confirmation

Both candidates apply their original fixed specs directly to accepted AD tree
`27629fd3cd4c7573ca535832ce4e97cb8bc10b28ddd3f6f12ddf1b194633b1e8`
and incumbent state `7d65d218…`. AR does not use AN as its parent.

| Saved candidate | Changed BF16 values | Tensors | Raw delta L2 | Create / verify / deploy seconds |
|---|---:|---:|---:|---|
| AN | 103,562 | 240 | 0.37208415696387465 | 239.725 / 183.252 / 111.849 |
| AR | 98,769 | 240 | 0.2865434048114147 | 226.559 / 181.254 / 111.773 |

The saved edited subsets match the measured candidate words within the
115,498-coordinate finite union. Source/candidate full-file hashes come from
the existing create/verify operations, with no third payload hash pass.

- AN closure: `225697374dfc0fd59adcd180c1c599cb348c31197d886e5739d204554b7e6905`;
  freeze: `bd1d74608ca10b1edb39b7ac2dfe93fc8cc81ba115cad7a0609de9b68649f39d`.
- AR closure: `e1e2e65ddeae257c6b32126a2c1b8dc15dc415552da84d8f0c2a275bb7ba5d6e`;
  freeze: `f1ef80185a375f926a5d6630c59d988d18cf687701ce8a709a817b64e31fbcfa`.
- Ordinary comparison: `A/current-ad-an-ar-dev90-v1/plan.json`,
  SHA `309c6bffdcf1136bce816700f2bac4a07be69302579d1725a2c7056ae20a2b97`;
  actual launch `00:00:50.556234 UTC`, all three runner captures completed.

The actual official child working directory is explicitly `R/results`, matching
the absolute results argument and its host bind. This preserves results with
the unchanged archived wrapper. An overlay score remains exploratory; these
fresh ordinary saved-weight scores are the confirmation evidence. The completed
AD/AN/AR scores were 74.742063/72.436508/73.075397: both candidates lost
against fresh AD and were rejected, with no full14 run. All 90 items per arm
were scored with zero unknown/truncated/empty outcomes. See the
[experiment result](experiments/2026-11-07-ad-functional-targets.md#ordinary-saved-weight-confirmation-an-and-ar-rejected)
for source tradeoffs, usage and closure hashes.


## Prepared BA binding: held before execution

The BA materializer binding applies the complete-unit spec
`ecc1d988…` to saved AV tree `5d056eb0…`, preserving AV0 as control and
targeting rejected AR3. Its 154,318 edited BF16 coordinates in 242 matrices
are bound to the fresh D-v2 sampled source words, the completed finite union,
and the independently verified actual post-TP overlay words (154,451 physical
destinations including tied K/V copies). This establishes selected-word
application evidence, not functional equivalence between overlay and saved runs.

Preparation is recorded under `A/current-av-full-unit-ba-materialization-v1`:
plan `58d77285002b7a8940adbdf3ad5807a657097fea22520702d2a24eb99fec1cc9`,
selected expectations `5ab1e796…`, and readiness `469c9077…`.
Ruff check/format and the inert metadata preview passed. Formatting preserved
the normalized AST of the reviewed owner. **No BA create, verify or deployment
has run**; execution was held while the working-parent choice was reconsidered.
A later execution still requires the exact authorization and unchanged
incumbent, source, closure and serving bindings.

The separately authorized obsolete closed QK overlay cache was removed after
fresh all-three-daemon running/stopped mount checks and an exact full inventory
match: 4,520 files and 988 directories, 933,350,947 logical bytes and
943,149,056 allocated file bytes. Receipt
`A/obsolete-qk-overlay-cache-audit-v1/cleanup.json` has SHA
`6f39b9fc231dbcd3ed85627f85f464ee1609afcaafce378f996473573789b7ce`.
No model, result, proof or current-run cache was removed; there were no service
actions. Recorded free space afterward was 146,189,119,488 bytes.

Here `R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104`,
`A=R/analysis/code90-resume-20261106`, and `CPU_PY=R/infra/weight-tools-ci/bin/python`.


### Compose membership before replacement

The single-replica helper inspects **all** containers with the target Compose
project/service labels, including stopped or renamed containers. Preview and the
last pre-apply check require exactly the expected original CID. Rollback refuses
more than one remaining member; zero members permits recreating the original
service after a failed create. These guards do not replace exclusive ownership
against an external actor between the check and Compose execution.

This protects a demonstrated failure: an observation restore deliberately kept a
stopped retired observer under another name, but with the same Compose labels.
Compose selected that object as well as the canonical service, removed the
retired object and stopped the original before failing; the old blind rollback
also failed. The original AV was restored with its exact CID and other production
services were unchanged. Benchmark/capture host artifacts survived, while the
retired container's unexported stdout/stderr may be unavailable. Do not treat a
healthy canonical name as proof that Compose will touch only that container.
Resolve an extra member's ownership and retain its evidence before any deliberate
cleanup; this helper never deletes an unexpected member to make its guard pass.

The helper also forces `COMPOSE_REMOVE_ORPHANS=0` for apply and rollback, so an
inherited shell flag cannot extend deletion to other project services. Rollback
restores the original weights/configuration and health; Compose recreation does
not promise restoration of the original container ID.
