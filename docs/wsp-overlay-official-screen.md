# Official code90 screening with sparse overlays

[screen-matrix-overlays-official.py](experiments/scripts/screen-matrix-overlays-official.py)
reuses the exercised four-slot overlay loading and restoration owner. It replaces
the authored eight-task generator and sandbox dispatch with the unchanged,
SHA-pinned official WSP runner. This is an exploratory comparison of **logical
overlays**, not an evaluation of six newly saved physical checkpoints.

The owner accepts one empty-parent control and one to six independent O/MLP
matrix increments from the same accepted physical parent. It loads four TP2,
static-MTP4 slots and evaluates up to three candidates with the control in each
of at most two rounds. The same empty control **server process** remains loaded;
each round gets a fresh control **official run**, unique run ID, observer and
counter interval. Thus each candidate is compared with its own round's control,
not a historical score or the first round's control result.

## Preserved evaluation contract

The official source is immutable 5e3f2df0c7f455013af3fa2d7abb3719858f808f.
The plan pins its 76 source/resource files and the 82 installed-package files
checked inside every actual runner before automatic container removal.
Each run uses dev/tier30, HumanEval+30, MBPP+30, LiveCodeBench20 and SciCode10;
chat concurrency32, logprob concurrency8 and agent concurrency1. Official
native-thinking behavior, sampler, token caps, selection and scoring are not
modified. The temporary serving image and load path differ explicitly from
ordinary saved-weight serving; all compared overlay arms use the same loader.

Each report label is the SHA-derived logical overlay identity. Receipts also
retain the unchanged physical parent tree, exact selected target words on both
TP ranks and the pinned unmodified draft bypass. An empty overlay is not a full
GPU-state hash. Loader-word agreement does not guarantee generated-output
equivalence.

The existing RequestRecorder is enabled identically for all runs with separate
capture directories and a 256 MiB quota. It records completed HTTP attempts;
logical calls, retries, errors and token totals are summarized separately.
These are not agent steps. Process counter intervals are separate from formal
item traces and retain startup traffic. No new raw-response parser is added.

After each completed round the owner reads numeric run/report metadata and calls
the existing wsp-run-evidence helper for source-equal scores, paired contrasts
and changed item IDs. SciCode partial credit remains numeric. Unknown outcomes
remain visible. It never treats a small norm, a finite-proxy improvement or an
overlay score as adoption of a physical checkpoint.

## Plan and lifecycle

The plan extends the preceding overlay-screen fields with official source and
installed-file pins, a fresh results directory/run prefix, the fixed protocol,
an environment mapping, and the existing numeric evidence helper. Transfer-case
and sandbox fields are absent. It requires exact incumbent, parent/draft freeze,
finite-closure and selection pins, distinct overlay plans/receipt paths,
fresh all-four serving snapshots and embeddings, and exact root authorization.

The default CLI only checks the plan SHA and prints an inert preview. It does
not import a Docker helper, inspect services, load weights or create results:

```sh
python screen-matrix-overlays-official.py --plan PLAN.json --plan-sha PLAN_SHA
```

The explicit --apply mode additionally requires --authorization and
--authorization-sha. Private HF_TOKEN is inherited, never recorded. A unique
owner reservation precedes mutation. All four exact original containers must
be idle before stopping. Candidate servers are removed between rounds; the
control is reused. Bounded child cleanup and independent restoration attempts
run even after a different child or temporary-container cleanup fails.
If capture failed before recording a runner, cleanup discovers only exact full
commands containing that fresh run ID, under a 20-second metadata deadline.
It rechecks the exact container ID and command before removal; discovery failure
is recorded and cannot skip original-container restoration.
The same original container IDs and protected configuration must be restored.

The plan supplies startup, per-round official, active-window and restoration
budgets. A failed run is recorded; there is no retry or amplitude search.
No candidate checkpoint is written and no existing parent is deleted.

A selected finalist still requires a saved physical checkpoint, byte
verification and a fresh ordinary-serving comparison before working adoption.
The repeatedly observed dev90 cohort is selection evidence, not final >90
confirmation or an unseen holdout.

## Validation status

The new owner has passed syntax, Ruff and a default unbound preview with zero
Docker, model or service operations. The first authorized execution started on 2026-11-06 at 23:11:17 UTC for
AN–AS, after the completed six-candidate finite replay. All eight original reports and same-CID restoration subsequently completed;
the result and recovery limits are recorded in the
[functional experiment](experiments/2026-11-07-ad-functional-targets.md). Prior transfer-screen evidence
does not substitute for this official-owner closure. Its known numeric report fields were checked against the completed AD/AL pair.
The archived runner and scoring sources are unchanged.

## Output persistence correction

The first executing owner (1d8977f3) omitted the official subprocess working
directory. The immutable wrapper mounted its caller directory, leaving the
requested result path inside the runner. Round-one original reports were
preserved by explicitly authorized opaque copies into the mounted observer
storage; no answer bodies were inspected and no reports were reconstructed.
For round two, before finalization, root authorized exact output-path routing:
items/Markdown and the atomic JSON temporary paths point into the mounted
recovery directory. Its separate response mirror excludes all routed files.
This affects artifact storage, not generation or scoring. The active source and
plan pins remain unchanged; all-four restoration completed at 2026-11-06T23:35:37.827730Z.

Future published dispatch uses the results directory itself as its working
directory. Actual OCI capture also requires a writable bind that covers the
requested result path and maps it to that same host path. Equal mounts across
arms alone are insufficient. The source change is for future execution; it
is not represented as the bytes used by the active batch.

The Q/K-capable revision accepts explicit Q/K/O/MLP specs only with the separately reviewed loader mapping and all selected post-TP destination proofs. Full-attention tied K/V loading proves both copies; no direct V or native edits are added. The former AN–AS runtime and its output evidence remain frozen. A new plugin requires a new pinned image and fresh overlay plans.
