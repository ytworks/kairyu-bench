# Bounded authored finite replay owner

[The executed owner recipe](experiments/scripts/replay-ad-authored-finite.py) runs a
single authored finite-replay batch, then restores the exact serving container it
temporarily stopped. It complements the existing
[finite replay tool](wsp-native-coupled-objectives.md); it does not generate answers
or run an official benchmark.

This is a **host-bound execution recipe**, not an independently validated portable
service controller. Its absolute workspace, evidence directory and pinned
operational helper are intentionally retained from the actual run. Read the
source and provide those exact dependencies before use. An inert preview checks
the bound plan; `--apply` additionally requires its exact authorization SHA.
The published recipe does not authorize a new run. The current optional plan-directory
parameter and declared fold-count support are described below.

## Executed relationship

The original runtime remained unchanged:

- Runtime: `D/run-current-ad-functional-six-finite-v1.py`
- Original SHA256: `2686e317783973c8c606495644598acd30a89473c590ca35c32c017d10646ef3`
- Plan SHA256: `b03b57ef2f0a84dcc2707e4c8774f08354b1012b9da2b434e3d77a385c43da14`
- Finite core SHA256: `9a5fb90f262459d142f63889e3a5807cb231bf797c302247379ed24d0be57631`

The initial publication changed formatting, split the import statement, and removed the
unused local alias `W_COUNTER` while preserving `counter`. After normalizing
those two syntax differences, that initial parsed AST was identical to the executed file.
Syntax compilation and Ruff check/format passed; publication did not repeat the
GPU run or full CPU suite. The correspondence receipt is
`A/functional-finite-owner-publication-v1/validation.json`.

Here `R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104`,
`A=R/analysis/code90-resume-20261106`, and
`D=R/runtime/code90-resume-20261106`. The owner evidence directory is
`A/current-ad-functional-six-finite-v1`.

## Lifecycle and proof scope

The owner binds the accepted AD checkpoint and prior full-file-hash attestation,
then checks current inventory/stat continuity, source hashes, all candidate
specifications, the authored cases and frame. It does not rehash the full model.
It checks that the task-owned AL service is idle before stopping only gemma-0,
uses GPU0 with CPU16–23 as UID1003, and preserves AD1, O2, AG3 and embeddings.
The fixed collector runs serial answer evaluation with four threads and seed0;
the collection window is 900 seconds and same-CID restoration allows 600 seconds.
Cleanup failure cannot skip the independent restoration attempt.

All candidates begin from AD; they are alternatives, not a sequence of cumulative
edits. The finite core restores the selected tensor union before each candidate
and after the batch. Its exact selected-word readback is checked against expected
BF16 hashes derived from the previously collected source samples. This proves
those selected words, not a fresh full-model GPU hash.

## Actual AN–AS result

The single run began at **2026-11-06 23:01:53.948588 UTC**. All six candidates
completed in **188.9853 seconds** of finite measurement. The original AL container
was healthy and restored at **23:07:10.219893 UTC**; total owner time was
**316.1145 seconds**. No retries, physical checkpoint copies or official scores
were produced.

Eight phases (baseline, six alternatives, restored baseline), **240 tensors** and
**115,498 union coordinates** matched the independently derived selected-word
hashes. Baseline fold correct-logprob difference was zero; the maximum baseline
case-margin difference was `9.5367431640625e-7`. Final A/A restoration was exact.
Parent AD state, checkpoint stat and protected-service checks passed.

Purpose means below equally weight the purposes and their two folds. Code3 is
code, algorithm and implementation; science6 is the six science purposes.
Positive correct-logprob or margin is favorable. These are teacher-forced
diagnostics, not ordinary code execution accuracy.

| Candidate | Science6 correct-LL Δ | Code3 correct-LL Δ | Code3 margin Δ | Actual unmet bounds / 84 |
|---|---:|---:|---:|---:|
| AN | +0.010088 | +0.096713 | +0.092459 | 34 |
| AO | +0.011993 | −0.105734 | −0.225936 | 33 |
| AP | +0.009380 | +0.016508 | −0.004276 | 33 |
| AQ | +0.009823 | −0.009081 | +0.143638 | 36 |
| AR | +0.009913 | −0.009109 | −0.156604 | 31 |
| AS | +0.007608 | −0.005663 | −0.075293 | 40 |

AQ's code margin improvement coincides with lower correct-logprob and larger
foil-logprob reduction. Linear predictions and raw-cap eligibility did not
guarantee measured bound satisfaction. No candidate adoption or functional
parity follows from these results.

Evidence in the owner directory:

| Artifact | SHA256 |
|---|---|
| `completion.json` | `782f3e653c2318068412aa741ab0b2cac75abce7de896fda36148fa249156e94` |
| `postchecks.json` | `ecd4385e654b14185400168d21bca923c7ef08c5c7867ab9fc446ca3980cfc01` |
| `numeric-final-summary.json` | `3fab7ab83a3453bea77106b2f99ea52450ceb9b7dbf9d5315f82065d512776c8` |
| `selected-byte-expectations.json` | `7b4a4880fba12399f5857322369bc53f6b15cc28c86268491e2b3a24cba6d7ed` |
| `independent-numeric-review.json` | `98d533192257f8c9f04b7d9b798560e65524728b21e26da49e0256dc871c4442` |


## Plan-bound cohort reuse

The owner now accepts optional --owner-plan /absolute/owner-plan.json. Omitting
it retains the historical directory above. The finite-completion fold count is
read from the pinned plan instead of assuming 42, and the exact temporary
container cleanup label is plan-bound with the historical value as its default.
Source, checkpoint, selected-word checks, timeouts and independent same-CID
restoration remain unchanged. This small parameterization passed syntax,
Ruff check/format, inert preview and an independent limited-diff review; it does
not change the finite numerical core or imply another executed batch.

The prepared AT–AY reuse observes all 160 authored rows / 23 purposes / 46 folds,
while its fitting subset is separately recorded as 112 rows / 17 purposes /
34 folds. The 48 omitted fitting rows remain diagnostics. Its explicit matrix
scope can include Q/K/O and coherent MLP, with native parameters and V frozen;
there is no inherited all-QKV-zero claim. Rejected AN in slot0 is the planned
scratch service, and AD1/O2/AR3 plus embeddings remain protected. Candidate
eligibility, source proofs and actual container snapshots are bound at dispatch.
The unbound template is A/current-ad-helper-six-finite-v1/owner-plan-template.json.
