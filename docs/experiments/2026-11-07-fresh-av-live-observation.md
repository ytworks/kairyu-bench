# Fresh AV live observation, 2026-11-07

Status: all 778 official items completed and scored; exact service restoration
and merge succeeded. Capture remains `partial_or_running` because two pre-run
requests lack client traces. This is an observed AV baseline, not a new weight
candidate or an adoption conclusion.

The user requested a fresh assessment from the coding objective, retaining tools
and evidence without inheriting earlier hypotheses or candidate decisions.
This operation measures one explicit saved AV checkpoint on the complete frozen
WSP tier-10 dev manifest: **778 items, 29 sources, all 14 areas**, no source or item
exclusions. Historical selection and numerical evidence are audited separately.

The operational recipe is
[`scripts/fresh-av-live10-20261107.py`](scripts/fresh-av-live10-20261107.py).
It uses existing observation, serving restoration and official WSP tools. It
holds `candidate-evaluation.lock` throughout its operation. Host UID is 1003;
there is no host-root or sudo execution. Only the named Gemma replica is replaced;
the original container ID is retained for exact restoration.

## Identities and conditions

Artifact root: `analysis/fresh-purpose-20261107-av-live10-v1` beneath the supplied
remote experiment root. Private Docker inspect backups remain private and must
not be committed. Numeric receipts, source hashes and this explanation are the
reviewable evidence.

- Saved AV identity, freshly hashed before attachment:
  `5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf`.
- Original checkpoint identity:
  `a876ff11e5a4ad6c97d95da1a5bc1da632b271b748de9c601bed071baa5e2b8b`.
  Its prior full hashes plus all nine file-stat identities were checked; its
  full payload was not freshly rehashed by this operation.
- Both chat and logprob use the single `gemma-0:8000/v1` endpoint. AgentAdapter
  derives its endpoint from that same chat client; no mixed-model gateway is used.
- TP2, DP1, MTP4, native BF16, the original compiled/CUDA-graph serving settings.
- Chat concurrency 32, logprob concurrency 8, agent concurrency 1.
- Frozen manifest SHA256:
  `a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`.
- Run ID: `gemma4-fresh-av-observed-dev10-20261107-v1`.
- Unique runner tag: `kairyu-bench:fresh-av-live10-20261107-v1`;
  image config digest `sha256:9ab4fbffb5a957cb96f4a8afd4afd31d4cb5649c25251dbc1d5cf3b83e954ac7`.
- Actual Docker image ID / OCI index digest:
  `sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8`.
  The `9ab4…` value above is the image configuration digest, not this engine’s
  runnable Docker image ID. `runner-image-source.json` hashes all 224 actual
  `/app/{src,scripts,adapters,pyproject.toml}` files copied from the running
  container to `runtime/fresh-purpose-20261107-av-live10-v1/runner-app`.
  `runner-cache-environment.json` records the actual PID-1 interpreter, installed
  versions in the WSP/EvalPlus/SciCode cache virtualenvs and 11 source revisions.
  Later comparisons must reuse this image or the archived source and verify
  those dependencies; rebuilding from a changing checkout is insufficient.
- Observer source snapshot and dependency/configuration hashes are retained in
  `source-dependencies.json`; capture identity/configuration hashes are in
  `checkpoint-identities.json`. The observer has a fresh dedicated NVMe compile cache.

The tier name is a calibration target, not this topology's measured runtime.
Earlier complete ordinary TP2/DP1 AV and AD runs took 1,559.1 and 1,696.6 seconds
of generation respectively. A new observed run is not interchangeable with those
ordinary runs for candidate ranking. A candidate comparison must use matched
source, observation and serving conditions.

## Capture scope and quota

Both TP ranks expose exactly 180 sites: attention, MLP and residual for all
60 layers. We retain each backend request and rank separately. Prefill captures
at most eight executed positions. Requested prompt-relative decode offsets are
`0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384`.
There is no semantic thinking/code boundary annotation, full attention matrix or
full activation tensor. These measurements are not causal capability labels.

The total storage allocation is 2 GiB. The current implementation gives each of
two workers 939,524,096 event bytes, with separate metadata/client/server budgets.
`CaptureWriter.bytes_written` counts actual gzip observation payload bytes plus
uncompressed acceptance-event bytes; it does not count expanded JSON size.
Writer queues separately cap pending estimated payload at 64 MiB per worker.
Quota flags, event drops, queue drops, error counters and `.quota-exhausted`
markers must be retained. Missing or rejected positions are not zero activations.

The preflight used a rounded conservative historical capture allowance of
806 MiB, scaled by 24/64 requested positions and multiplied by four (about
1.21 GiB). This is an estimate, not a promise: request mix, prefix caching,
compression and MTP acceptance can differ. Capture settings are not changed
mid-run to hide missingness. The complete benchmark remains unchanged if the
capture becomes partial.

## Minimal overhead/output check

Four fixed short synthetic requests were sent sequentially before and after
attachment, with temperature 0, top-p 1, seed `1700 + index`, maximum 64 tokens
and **thinking disabled**. They are not WSP items and do not estimate coding
accuracy. Only public-answer hashes, numeric usage, finish status and elapsed
time were retained; no hidden reasoning was inspected or stored by this probe.

| Measurement | OFF | ON |
| --- | ---: | ---: |
| Request 1 seconds | 0.086918 | 0.094703 |
| Request 2 seconds | 0.071712 | 0.076546 |
| Request 3 seconds | 0.097448 | 0.104531 |
| Request 4 seconds | 0.068080 | 0.073825 |
| Sum seconds | 0.324158 | 0.349605 |

All four public-answer hashes matched and all responses finished with `stop`.
The total elapsed increase was 7.85%. This is one small sequential measurement,
not a c32 WSP overhead estimate or proof of output invariance on longer thinking
runs. Both worker streams produced real records with no errors/drops before WSP
started. The synthetic requests have separate item/run IDs in capture metadata.

## Retained operational issues

- Calling `./kairyu-bench wsp run --help` is not inert: the wrapper builds an image
  and its entrypoint prepares dependencies before CLI help. The observation
  owner unintentionally started one help container, identified its exact ID
  `f30d51e7f7c64ecef6b9868def8eca5e1d2c6f323744d15446b284e89c927472`, and stopped only
  that owned container. No benchmark/model request occurred. Subsequent help
  inspection used source/CPU parser inspection. This issue is recorded rather
  than expanding scope with an unrelated wrapper refactor.
- An intermediate progress message used `0/1` for the currently active Terminal
  task. The actual manifest contains two Terminal tasks. This was corrected to
  the full denominator immediately after checking the frozen selection.
- The root agent authorized a separately recorded **planned** finite-probe
  sublease for the exact Gemma-3 container and GPU6 computation. The sublease
  requires its own GPU lock, idle checks and exact-container restoration and
  protects Gemma-0/1/2 plus both embedding services. It does not release the
  primary lock or authorize gateway benchmark calls. Shared CPU/NVMe contention
  must be disclosed for overlapping timings; finite proxy scores remain separate.
  The root agent subsequently reported AV→F01→AV, all 120 cases per pass,
  zero restoration difference, and the exact Gemma-3 CID restored healthy.
  This operation did not inspect or operate Gemma-3 itself; the delegate owns
  its numeric receipts. This notification is retained in
  `finite-gpu6-sublease-completion-notice.json`.

## Completion requirements

Require all 778 items to retain their official scored/unscored outcomes. Preserve
all per-problem failures and source/area losses. After requests complete, check
both writer queues and gzip integrity, restore the exact original Gemma-0 CID,
and merge against the full WSP results. Incomplete writer close acknowledgement
must remain incomplete even when its queue is empty. The final result and
restoration status will be appended after those steps complete.

## Official completed benchmark

All **778/778** items are scored, with zero unscored items and zero request
errors. Code is **87.7778/100**, the equal mean of HumanEval+ **10/10**,
LiveCodeBench **5/6**, and MBPP+ **8/10**. The three failing code items are
`livecodebench:arc191_c`, `mbpp-plus:Mbpp/410`, and `mbpp-plus:Mbpp/576`; both
MBPP failures pass base tests and fail plus tests. The 23 other code items pass.
All per-item numeric outcomes and all source statistics are retained in
[`artifacts/fresh-av-live10-20261107-result.json`](artifacts/fresh-av-live10-20261107-result.json).

| Area | Score / 100 |
| --- | ---: |
| knowledge | 88.3333 |
| expert | 87.5000 |
| facts | 75.0000 |
| math | 98.7500 |
| code | 87.7778 |
| instructions | 85.4167 |
| tools | 82.5000 |
| agents | 50.0000 |
| long-context | 66.7705 |
| reasoning | 76.3043 |
| commonsense | 63.6667 |
| multilingual | 82.0960 |
| vision | 81.2500 |
| safety | 55.5556 |

Overall is 77.2086/100. Official loading time was 102.8 seconds and generation time 1800.7 seconds (30 minutes 0.7 seconds), with 558 chat and 3,062 logprob API calls. These counts include the official sampling plan and are not per-item step counts. TerminalBench completed both tasks: `install-windows-3.11` reward 0, 12 agent steps / 26 tool calls; `portfolio-optimization` reward 1, 7 agent steps / 13 tool calls. There were no Terminal trial exceptions.

## Final restoration and capture audit

The original Gemma-0 container was restored with the exact original CID
`4e545c35bc0e277344b302058eb9ffa758747f3012bc8e395e417f28d97aa56e`
and healthy state. The owner completed normally in 2,350.32 seconds; CPU merge
took 196.12 seconds and produced 2,890,586,309 bytes. The global evaluation lock
was verified available after the owner exited.

Both TP writers closed successfully: zero queue/pending bytes, zero dropped
records/batches, no errors, no close timeout and no quota markers. TP0 event
bytes were 169,003,868 / 939,524,096 (17.99%); TP1 were
128,634,191 / 939,524,096 (13.69%). All gzip members were decoded by both the
standard merge and the independent compact-batch audit.

The final merge retains **7,001,280 activation records**: 6,977,160 join to this
official run; 18,360 belong to the four separately labelled synthetic requests
and are excluded; **5,760 remain unlinked**. The latter are exactly two
HTTP-200 prefill-only requests, each 2 ranks × 180 sites × positions 0–7,
appearing after the four synthetic requests and before the first official
item in server-record order. Untraced model-discovery probes in WSP/agent startup
are consistent with these records, but their individual origin is an inference.
They are not silently relabelled or deleted; the standard status therefore
remains `partial_or_running` despite closed writers.

The reproducible CPU audit is
[`scripts/fresh-capture-code-audit.py`](scripts/fresh-capture-code-audit.py),
using the existing request parser and gzip reader. Its compact numeric receipt is
[`artifacts/fresh-av-live10-20261107-capture-audit.json`](artifacts/fresh-av-live10-20261107-capture-audit.json).
The audit found activation records for **778/778** official items; missing item IDs: `[]`.

All 26 code items have exactly one traced call and both ranks expose all 60
layers × attention/MLP/residual. Each has eight sampled prefill positions and
10–15 executed requested decode offsets, identical as sets across the two ranks.
The missing later offsets are outside the sampled execution of these calls;
shorter completions cannot supply the later measurements. Completion lengths
range from 482 to 11,372 tokens. MBPP/410 is 94 prompt + 797 completion tokens,
MBPP/576 is 94 + 1,392, and arc191_c is 613 + 8,895. There is no code truncation
or empty response in this run.

The 64-token merged bins may combine several requested early positions. The
following table instead reads the exact **prompt-relative decode offset 64**,
**layer 25**, separately by request and rank, from original compact batches.
Attention is before post-attention norm; MLP is before post-feedforward norm;
residual is the block output after the layer scalar. Their different numerical
scales are not directly comparable contribution weights. Both ranks happened
to agree numerically at this site/position, but remain separate records.

First-execution sampling includes rejected speculative positions. At offset
64, 18 questions have accepted observations (16 passing, 2 failing); 8 have
rejected observations (7 passing, arc191_c failing). This is not an accepted-only
trajectory and does not establish a failure mechanism. Later comparisons require
matching problem/input identity, call/sample, layer/role, rank, absolute and
relative position, phase and acceptance state; generated prefixes may still
differ and require interpretation as different trajectories. Token content and
a thinking/code semantic boundary are not measured. No channel/head attribution
is available in this capture.

| Item | Score | Rank | Accepted | Attention RMS | MLP RMS | Residual RMS |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| humaneval-plus:HumanEval/116 | 1 | 0 | true | 0.941119 | 0.175792 | 4.610483 |
| humaneval-plus:HumanEval/116 | 1 | 1 | true | 0.941119 | 0.175792 | 4.610483 |
| humaneval-plus:HumanEval/144 | 1 | 0 | true | 0.699860 | 0.181055 | 4.407145 |
| humaneval-plus:HumanEval/144 | 1 | 1 | true | 0.699860 | 0.181055 | 4.407145 |
| humaneval-plus:HumanEval/148 | 1 | 0 | true | 0.945289 | 0.185209 | 4.086910 |
| humaneval-plus:HumanEval/148 | 1 | 1 | true | 0.945289 | 0.185209 | 4.086910 |
| humaneval-plus:HumanEval/155 | 1 | 0 | true | 1.150832 | 0.199632 | 4.788787 |
| humaneval-plus:HumanEval/155 | 1 | 1 | true | 1.150832 | 0.199632 | 4.788787 |
| humaneval-plus:HumanEval/21 | 1 | 0 | false | 1.048464 | 0.217120 | 4.705369 |
| humaneval-plus:HumanEval/21 | 1 | 1 | false | 1.048464 | 0.217120 | 4.705369 |
| humaneval-plus:HumanEval/22 | 1 | 0 | false | 0.975483 | 0.207112 | 4.888043 |
| humaneval-plus:HumanEval/22 | 1 | 1 | false | 0.975483 | 0.207112 | 4.888043 |
| humaneval-plus:HumanEval/30 | 1 | 0 | true | 1.046065 | 0.190525 | 4.493063 |
| humaneval-plus:HumanEval/30 | 1 | 1 | true | 1.046065 | 0.190525 | 4.493063 |
| humaneval-plus:HumanEval/49 | 1 | 0 | true | 0.823943 | 0.169817 | 4.619727 |
| humaneval-plus:HumanEval/49 | 1 | 1 | true | 0.823943 | 0.169817 | 4.619727 |
| humaneval-plus:HumanEval/57 | 1 | 0 | true | 0.723154 | 0.261854 | 4.607836 |
| humaneval-plus:HumanEval/57 | 1 | 1 | true | 0.723154 | 0.261854 | 4.607836 |
| humaneval-plus:HumanEval/90 | 1 | 0 | false | 0.981683 | 0.169821 | 4.988757 |
| humaneval-plus:HumanEval/90 | 1 | 1 | false | 0.981683 | 0.169821 | 4.988757 |
| livecodebench:abc393_d | 1 | 0 | true | 1.013324 | 0.167566 | 4.949711 |
| livecodebench:abc393_d | 1 | 1 | true | 1.013324 | 0.167566 | 4.949711 |
| livecodebench:abc394_b | 1 | 0 | true | 1.121866 | 0.182338 | 4.362260 |
| livecodebench:abc394_b | 1 | 1 | true | 1.121866 | 0.182338 | 4.362260 |
| livecodebench:abc395_f | 1 | 0 | false | 0.855383 | 0.178065 | 4.566522 |
| livecodebench:abc395_f | 1 | 1 | false | 0.855383 | 0.178065 | 4.566522 |
| livecodebench:abc397_b | 1 | 0 | false | 0.864365 | 0.181684 | 4.803696 |
| livecodebench:abc397_b | 1 | 1 | false | 0.864365 | 0.181684 | 4.803696 |
| livecodebench:abc399_c | 1 | 0 | true | 0.927112 | 0.194868 | 4.636884 |
| livecodebench:abc399_c | 1 | 1 | true | 0.927112 | 0.194868 | 4.636884 |
| livecodebench:arc191_c | 0 | 0 | false | 0.746142 | 0.153766 | 4.546649 |
| livecodebench:arc191_c | 0 | 1 | false | 0.746142 | 0.153766 | 4.546649 |
| mbpp-plus:Mbpp/256 | 1 | 0 | false | 1.051414 | 0.194779 | 4.816031 |
| mbpp-plus:Mbpp/256 | 1 | 1 | false | 1.051414 | 0.194779 | 4.816031 |
| mbpp-plus:Mbpp/271 | 1 | 0 | true | 1.040993 | 0.186567 | 5.070911 |
| mbpp-plus:Mbpp/271 | 1 | 1 | true | 1.040993 | 0.186567 | 5.070911 |
| mbpp-plus:Mbpp/410 | 0 | 0 | true | 1.030764 | 0.191087 | 5.022425 |
| mbpp-plus:Mbpp/410 | 0 | 1 | true | 1.030764 | 0.191087 | 5.022425 |
| mbpp-plus:Mbpp/567 | 1 | 0 | true | 0.912405 | 0.206068 | 4.416720 |
| mbpp-plus:Mbpp/567 | 1 | 1 | true | 0.912405 | 0.206068 | 4.416720 |
| mbpp-plus:Mbpp/576 | 0 | 0 | true | 0.902118 | 0.169462 | 4.518276 |
| mbpp-plus:Mbpp/576 | 0 | 1 | true | 0.902118 | 0.169462 | 4.518276 |
| mbpp-plus:Mbpp/612 | 1 | 0 | true | 0.975593 | 0.172739 | 4.529507 |
| mbpp-plus:Mbpp/612 | 1 | 1 | true | 0.975593 | 0.172739 | 4.529507 |
| mbpp-plus:Mbpp/63 | 1 | 0 | true | 0.903108 | 0.193072 | 5.181970 |
| mbpp-plus:Mbpp/63 | 1 | 1 | true | 0.903108 | 0.193072 | 5.181970 |
| mbpp-plus:Mbpp/791 | 1 | 0 | true | 0.942521 | 0.212317 | 4.646260 |
| mbpp-plus:Mbpp/791 | 1 | 1 | true | 0.942521 | 0.212317 | 4.646260 |
| mbpp-plus:Mbpp/792 | 1 | 0 | false | 0.848114 | 0.197197 | 4.923497 |
| mbpp-plus:Mbpp/792 | 1 | 1 | false | 0.848114 | 0.197197 | 4.923497 |
| mbpp-plus:Mbpp/801 | 1 | 0 | true | 0.905084 | 0.186711 | 4.971838 |
| mbpp-plus:Mbpp/801 | 1 | 1 | true | 0.905084 | 0.186711 | 4.971838 |

These are descriptive amplitudes, not an optimized score, selected peak or
causal localization. All 26 questions are retained at the same specified site
and position; no cross-question/rank/token aggregate is used to assign a skill
to layer 25. The residual/normalization path means a raw MLP or attention output
change alone cannot predict the final retained state. A matched saved-weight
intervention and full benchmark comparison remain necessary.

## Reusing the owner safely

The owner now takes `--artifact-dir` and an explicit `--preview`. The original
executed recipe was preserved with its original SHA256
`59582d16e91fa2d4f31fd3b67fbd02cdbd1a68f8bbca4c2460688c94d6d52d24`
in the remote baseline evidence. A planned F02 preview verifies all 89 frozen
observer source files and the exact baseline runner image, without hashing a
candidate checkpoint or touching serving. CPU rejection previews confirm that
a different runner ID, source subset and TP topology are rejected. The planned
checkpoint and CID are placeholders, with `approved_for_execution` plus a real
64-hex CID required for execution. This recipe attaches to an already-served
checkpoint; it does not deploy candidate weights. Its finally block returns to
that exact pre-attachment CID. Weight deployment/adoption remains a separate
explicit operation.

Audit implementation issues retained: the first lint check found an import-order
error and initial draft used an incorrect request-ID pattern; both were corrected
to the existing parser before any data audit. One expanded-audit invocation used
a mistyped artifact directory and failed before opening capture input; the
corrected run completed in 16.61 seconds and retained the first audit output.
An intermediate message counted eight passing rejected-offset questions; the
receipt showed seven passing plus one failing, and the message was corrected.

To validate a prepared plan, run the recipe with the remote CPU environment:

```sh
/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python \
  docs/experiments/scripts/fresh-av-live10-20261107.py \
  --artifact-dir /mnt/nvme/kairyu/wsp/gemma4-coding-20261104/analysis/fresh-purpose-f02-live10-plan-20261107-v1 \
  --preview
```

That F02 directory is **planned only**: its checkpoint and serving CID must be
replaced with verified saved/deployed identities before any execution. Its
`preview.json` and `preview-validation.json` retain the successful matched-source
preview and three intentional rejection checks. The script never changes the
parent checkpoint or deploys weights. The baseline's full 5.77 MB numeric code
audit remains on NVMe, with its SHA in the compact PR artifact. No additional
GPU calculation was used for the RMS table.

## Later Compose collision: retained evidence and limits

A separate F02 deployment operation later failed while Compose processed the
stopped retired observer together with the ordinary Gemma-0 service. The
observer owner had followed existing `observe-docker.py restore`: stop the
observer, rename it `observer-retired-*`, restore the exact original CID, and
retain the stopped observer as recovery evidence. The observer clone preserves
original Compose labels, so this left two containers associated with the same
service. The retired observer was subsequently absent. This observation owner
did not perform the failed deployment or cleanup and left recovery to its owner.

A read-only [post-incident check](artifacts/av-observer-retired-host-evidence-20261107.json)
confirmed the capture configuration SHA, both worker-status objects against the
closed-writer receipt, official run/report SHAs, and private original-backup SHA
against their preexisting receipts. Host capture streams, site/status/acceptance
metadata, all results, merged output and audits remain present; no large trace
payload was rehashed in this check. All original missingness remains explicit.

The original saved inspect contains Compose project/service/container-number/oneoff
values `gemma-4-31b-8gpu` / `gemma-0` / `1` / `False`. The pre-attach source pin,
actual runner archive and current observer tool share SHA256
`c25c4ef70d1825686c0c9d93066432c5ffff15abcb813b86b5aeca6d80b4d97b`.
That implementation copies original labels, adding only observation labels,
and restore does not change labels. Together with successful creation/restoration
receipts this reconstructs the retired observer's Compose identity. A separately
saved inspect of the observer itself was not found; this distinction is retained.

WSP/merge/init/owner operation logs are on the host. The owner did **not** export
the observer container's Docker stdout/stderr separately before retirement;
those logs may therefore be unavailable after deletion. This limitation does not
remove the captured activation streams, but matters for later runtime diagnosis.
Any future owned-retired cleanup needs exact-CID ownership, stopped state,
healthy original restoration and preserved host evidence checks. No cleanup was
performed as part of this audit.
