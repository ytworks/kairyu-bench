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

The completed AT–AY reuse observed all 160 authored rows / 23 purposes / 46 folds,
while its fitting subset is separately recorded as 112 rows / 17 purposes /
34 folds. The 48 omitted fitting rows remain diagnostics. Its explicit matrix
scope can include Q/K/O and coherent MLP, without native or V tensor writes.
In Gemma4 full-attention layers with K_eq_V, K-source edits can change both key
and value functions; the value function is not claimed frozen. There is no
inherited all-QKV-zero claim. Rejected AN in slot0 was the
scratch service, and AD1/O2/AR3 plus embeddings remained protected. Candidate
eligibility, source proofs and actual container snapshots were bound at dispatch.
The unbound template is A/current-ad-helper-six-finite-v1/owner-plan-template.json.


## Actual AT–AY result

The parameterized recipe SHA256
`ed12a9df2f3ef442060cb39224f3bfee6bb111c1ee5e6b54eb6d5876b3aa3f95`
was executed once at **2026-11-07 00:43:26.185758 UTC**. Finite measurement
took **201.3247 seconds**; the same AN0 container was healthy again at
**00:48:55.003408 UTC**, with total owner time **328.6602 seconds**.
All eight phases matched **360 tensors / 179,842 union coordinates** against
the independently derived BF16 expectations. Baseline fold correct-LL difference
was zero, maximum case-margin difference was `9.5367431640625e-7`, and final
A/A margin, NLL, KL and selected-word restoration differences were zero.
Source pins, current AD state, all nine checkpoint file stats and protected
serving identities remained unchanged.

The primary code group for this batch is implementation, algorithm and workflow
(core3); plain code is separately retained. Science6 excludes the two newly
authored helper/composition purposes. Each reported group equally weights its
purposes and folds. The 68 fitting conditions cover the selected 17 purposes;
the six omitted purposes remain separately reported diagnostics.

| Candidate | Core3 correct-LL Δ | Plain code Δ | Science6 Δ | Helper Δ | Composition Δ | Actual unmet fit bounds / 68 |
|---|---:|---:|---:|---:|---:|---:|
| AT | +0.046729 | −0.173047 | +0.044504 | +0.097285 | +0.072177 | 29 |
| AU | +0.050484 | −0.052019 | +0.046859 | +0.093518 | +0.071005 | 26 |
| AV | +0.040201 | +0.118661 | +0.046601 | +0.093360 | +0.071512 | 36 |
| AW | +0.049410 | +0.129617 | +0.012132 | +0.060786 | +0.060627 | 13 |
| AX | +0.042476 | +0.131789 | +0.045506 | +0.094267 | +0.077114 | 31 |
| AY | +0.036213 | +0.029985 | +0.042132 | +0.075003 | +0.074617 | 36 |

AX, AV and AW were selected for a subsequent ordinary-generation comparison.
These finite diagnostics establish neither ordinary benchmark improvement nor
adoption. The accepted physical parent remains AD.

Evidence under `A/current-ad-helper-six-finite-v1`:

| Artifact | SHA256 |
|---|---|
| `owner-plan.json` | `127294963b3274678db7cece686f604455807a47a6cdfb3013144d3d9f17c2a9` |
| `completion.json` | `a289535209686206cc2024337960eca568fadfff76ec54b24fdbc310fe69f3dc` |
| `postchecks.json` | `4ce0c08cb7034269299df6809e0381be101ffb7469995555c79c640b67b6f191` |
| `numeric-final-summary.json` | `a96e9850c2d1ba6d0d668420805162fb751af2c8d4bd331f48fa54bec83e665e` |
| `selected-byte-expectations.json` | `9ff04076149370194f435fe915215b63866521ff433555702105c7575a39333b` |
| `replay/selected-byte-proofs.json` | `808af06c7b20517f7f51bf2bcc332d10d75db66ee6bff177c33b4406e6f329a4` |
| `independent-proof-tail-review.json` | `19dabb3a5b87d96e2a74c839f6ce67d6979b17eeb3fd0ad4a90b6d36eaf1c9b7` |
