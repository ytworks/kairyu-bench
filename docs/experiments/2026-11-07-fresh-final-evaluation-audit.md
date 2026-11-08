# Fresh final-evaluation audit, 2026-11-07

This is an independent operational and result audit for the fresh weight-editing
cycle. Prior hypotheses, candidate rankings and adoption decisions are not
adopted. Historical results below are evidence about completed runs, not evidence
that a historical candidate should be selected. All inspections and calculations
ran over SSH on the designated eight-GPU host. No benchmark, model deployment,
review request, Wiki push or Hugging Face upload was performed by this audit.

Repository inspected: `ba98dc17e3fac687faeab5072246e6601c26b6a1`.
Paths below are relative to the experiment root unless stated otherwise.

## Independently checked DeepSWE evidence

The original model's completed run is
`gemma4-base-c32-thinking-20261104-r3`, located in the existing benchmark checkout's
results directory. The historical edited-model run is
`results/gemma4-h0397-c32-thinking-20261106`. This audit read `run.json`, normalized
results, conditions and every canonical `raw/deepswe/trials/repeat-0/*/result.json`.
It did not inspect private reasoning or use benchmark answers for editing.

| Directly checked result | Original R3 | Historical h0397 |
| --- | ---: | ---: |
| Run and normalized status | completed | completed |
| Canonical task / unique trial UUID count | 113 / 113 | 113 / 113 |
| Official reward 1 / reward 0 | 1 / 112 | 1 / 112 |
| Official pass@1 | 0.8849557522% | 0.8849557522% |
| Unscored / canonical exceptions | 0 / 0 | 0 / 0 |
| Archive result files | 0 | 0 |
| Recorded agent steps | 8,274 | 8,071 |
| Recorded input tokens | 379,689,393 | 346,460,961 |
| Recorded output tokens | 3,488,483 | 3,143,746 |
| Wall time from run timestamps | 7,608.619834 s | 6,485.156925 s |

The 113 task-ID sets and all task checksums match. Each run has one canonical
agent-config variant across all 113 tasks. Original R3's success is
`claude-code-by-agents-recursive-delegation`; h0397's is
`true-myth-iterable-collection-combinators`. Thus the historical pair has one gain
and one loss, not an increase in successful tasks. This is not the new cycle's
final evaluation and does not satisfy the user's request to evaluate its final
edited checkpoint.

Earlier incomplete records remain evidence, not completed comparison scores:

| Run | Normalized status | Scored / planned | Passed | Unscored |
| --- | --- | ---: | ---: | ---: |
| gemma4-base-c32-20261104 | failed | 0 / 113 | 0 | 113 |
| gemma4-base-c32-20261104-r2 | partial | 16 / 113 | 0 | 97 |
| gemma4-wsp-c32-20261104 | partial | 109 / 113 | 0 | 4 |

## Conditions and new final-run requirements

Current `src/kairyu_bench/deepswe.py` accepts workers 1–32 and attempts 1 or 4;
`deepswe_runtime.py` requires the locked 113 tasks. The requested final evaluation
is workers **32**, attempts **1**, with no `--limit` or task filter. The repository
wrapper remains the execution entry point. Historical R3 conditions are:

- mini-swe-agent 2.4.6, revision `a83fcae82d2a08f0ee0c688f9d137b3566c097f8`;
  Chat Completions transport and server-default reasoning effort.
- workers 32, attempts 1, max retries 3, HTTP response timeout 3,600 seconds;
  task timeout multiplier 1 and agent setup multiplier 4.
- Agent cost limit 0, no explicit additional max-token setting in the recorded
  agent kwargs; the two historical runs have the same resolved agent config.
- Runtime adapter SHA-256
  `a3ebcf772fba4c2aabe365c5346120df3c57367714129dc81f2092c15c868ca3` and agent-lock
  SHA-256 `4cb499a6388e1cc6f1b14eb0124725014ac4b8e0a10db5390accae990a44566e`.
  Both hashes also match the current repository files.

Historical serving evidence describes TP2 × four replicas, MTP4, native thinking
on, without an added thinking-token cap. The current gateway lists the model ID
`gemma-4-31b`, but a model ID alone does not attest a checkpoint or identical
weights across replicas. Before the final run, verify all reachable replicas,
weight identity, serving image/flags, sampling, context/output settings, and
capture state. In particular, a single-replica edited endpoint cannot be treated
as the same serving condition as original R3's four-replica run. If the final
configuration uses one replica, obtain a new original-model run with that same
configuration for a valid comparison, or restore and attest a homogeneous
four-replica configuration. Keep the final run at 32 problem workers either way.

The existing `run-deepswe.sh` dispatches a legacy stateful
`runtime/final_deepswe.py`; it is not a generic new-run launcher. Construct the new
concrete command from the current wrapper and verified environment. Use the
existing dedicated daemon only after rechecking its identity, mounts and absence
of unrelated tasks. A new run ID and result directory must not overwrite an old
run. Retain native task and verifier limits and every selected task.

After launch, check 113 selected IDs and 113 planned slots; monitor official
scored/passed counts, active tasks, agent steps, elapsed time, retries, API and
Docker health, and free space. Completion requires 113 official scores, completed
run and normalized status, canonical-ID agreement, and no cleanup failure.
Missing scores stay null. The existing
[`deepswe-recorded-usage.py`](scripts/deepswe-recorded-usage.py) can extract
canonical usage and optional agent-recorded API counters without reading message
bodies; distinguish those counters from independent HTTP telemetry.

The user's explicit current request authorizes the final DeepSWE execution and
its existing dedicated privileged runner/daemon/socket arrangement. No additional
permission flow is introduced by this audit. Host root and sudo remain prohibited.

## Dedicated daemon and storage snapshot

The dedicated `gemma-wsp-deepswe-dind-20261104` container is running and its Unix
socket answers `docker info`. Its inner daemon has **0 containers, 0 running
containers and 119 images** at inspection. The Docker state retains
`OOMKilled=true`; that historical flag does not prove a current outage, but it
must not be silently described as a clean lifetime history.

Its socket/data root are under the existing NVMe `docker/gemma-wsp-coding-20261104`
directory; the benchmark cache is under NVMe
`bench-cache/gemma-wsp-coding-20261104`. The experiment results and that cache are
mounted at identical paths. NVMe has approximately **137 GiB free (97% used)**;
root has approximately 341 GiB free. Keep outputs/caches on NVMe and reassess
headroom before model materialization or task-image pulls. No pruning or cleanup
was performed. Do not remove the 119 images merely because no tasks are active.

## Actual Claude Opus 5.5 review path

The noninteractive shell does not find `claude` on PATH. The explicit binary
`/home/y-takagi/.local/bin/claude` exists and reports Claude Code 2.1.306. Its help
supports the safe/restricted/system-prompt options used by
[`claude-method-review.py`](../../scripts/claude-method-review.py).

Existing recent receipts show that this binary successfully received responses
whose assistant model, usage model and canonical model were all
`claude-opus-5-5`. That is availability evidence only; the new candidate needs its
own second opinion. No previous review conclusion is inherited.

For a concrete candidate, prepare an anonymous prompt covering the edits,
objective, actual measurements, losses and uncertainties. Check every byte,
run preview with `--model claude-opus-5-5`, an explicit binary and labels, then
compute the prompt SHA-256 and use `--send --approved-prompt-sha256` with a fresh
output directory. Do not include credentials, private paths or unnecessary raw
data. The script requires matching model identity, no tool use, successful CLI
exit, and matching review labels (or its explicitly requested single-label text
mode). Record failure/timeout receipts as trials; never call another model Opus
5.5. The prompt, receipt, review and adoption/rejection rationale belong in the PR.

## Wiki and conditional Hugging Face publication

The existing Wiki checkout is `/mnt/nvme/kairyu/bench-1017/wiki`, and the page is
[`DeepSWE-Comparison`](https://github.com/ytworks/kairyu-bench/wiki/DeepSWE-Comparison).
At inspection it has no worktree edits but reports `master...origin/master
[ahead 4]`. HEAD is `9a5335927336e339ebaccc272110e52ca61d1369`; that commit is also
recorded as remotely verified by the previous publication receipt. The stale
tracking relation and prior receipt do not establish today's remote state. Fetch
and inspect ancestry before appending a new result; preserve all existing history.

The previous `runtime/publish_final_wiki.py` imports `prepare_hf_release` from a
reporting location where it is currently absent. A backup exists under
`runtime/nvme-results-stage/reporting/`, but it encodes the old experiment state
and is not a generic publication tool. Do not blindly run it. Any reusable new
publication or evidence helper must be committed to PR #33.

The CPU environment now has `huggingface_hub`. A standard `get_token()` probe
returned **no available credential**; no token value was printed. An older
preflight records automatic approval review rejecting direct extraction from
shell configuration. That old rejected extraction was not repeated. If the final
same-condition edited-model result exceeds the original, prepare the exact
checkpoint inventory, model card, license, editing provenance and full evaluation
record first, then use a standard authorized authentication path for publication.
Do not infer usable credentials from documentation or claim an upload occurred.
No repository was created and no model was uploaded in this audit.

The historical backup release preparer is only a local inventory builder, not an
upload implementation. Its assumptions about the upstream license and the old
candidate must be revalidated against the selected checkpoint before use. Upload
eligibility depends on the complete final DeepSWE comparison, not WSP gains,
partial-test totals, reduced tokens or a historical candidate's result.
