# Saved finalist materialization and fresh dev90

These two host-bound recipes reuse the successful AL saved-checkpoint path:

- [Materialize one selected AD increment](experiments/scripts/materialize-ad-finalist.py):
  derived from executed `materialize-and-deploy-current-ad-library-al-v1.py`,
  SHA `a20fc5efb905c2ea1a4c8d0ee0ac44c5378f9ace6bfc0933f98cb6a90ed2fe87`.
- [Launch fresh AD plus one or two saved finalists](experiments/scripts/launch-ad-finalists-dev90.py):
  derived from executed `launch-current-ad-al-dev90.py`,
  SHA `fc3de9942500b0ef8a5cd498ffe0a56a12ef14b95e7551cc095d062787ebca38`.

The new parameter bindings are **prepared, not executed**. The source lineage
does not claim that the new finalists have been saved, served or scored.
The unchanged absolute workspace and existing operational helpers are part of
this machine-specific recipe; this is not a general service orchestration SDK.

The only behavioral changes are selecting an AN–AS label, choosing target slot0
or slot3 through the materialization plan, and supplying the official plan via
`--plan`. A single official comparison can contain one or two distinct selected
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

Every candidate applies its exact BF16 spec to the same accepted AD saved
checkpoint. It never uses original weights or another finalist as its base.
One existing editor create is followed by one independent editor verify.
The existing full-file hash evidence, private-inode checks and measured selected
BF16 parity produce the freeze without a third payload hash pass.
Deployment replaces only the selected slot with rollback available; AD1, O2,
the other slot and embeddings are protected by fresh checks.

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

The explicit per-create reserve is **80 GiB**. A private candidate requires
62,578,656,419 bytes plus the existing 67,108,864-byte editor headroom:
**148,545,111,203 free bytes** before each create. Two copies are sequential;
there is no implicit relaxation of the reserve or automatic deletion.

A read-only metadata survey found the completed previous authored screen's
private cache used 1,649,119,232 allocated file bytes (7,910 files, UID1003,
no symlinks or multiply linked files). It is only a prospective cleanup option:
the exact owner must be closed and all current/running/stopped mount references
must be checked before separately authorized deletion. The active official
screen cache is excluded. Retired AL weights may only become a cleanup option
after a successful replacement; its metadata, results and all accepted ancestry
must remain. Recheck free space immediately before each actual copy.

Here `R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104`,
`A=R/analysis/code90-resume-20261106`, and `CPU_PY=R/infra/weight-tools-ci/bin/python`.
