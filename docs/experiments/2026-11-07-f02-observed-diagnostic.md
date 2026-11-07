# F02 matched observed diagnostic, 2026-11-07

Status: all 778 official items completed and scored. Code remains 87.7778,
with the same three failures as AV, so the frozen strict-code-gain condition
failed. Exact ordinary F02 restoration and merge completed; capture remains
partial because both writer close acknowledgements are missing and two pre-run
requests are unlinked. No flags were patched and no candidate run was repeated.

This is the one complete diagnostic authorized by the frozen
[post-data amendment](2026-11-07-f02-transfer-amendment.md), not a passed proxy
gate or an adopted improvement. Amendment body SHA256 is
`1b6df2b79abb7d4db023651df632509a49d0b35e816c40c6d342d3884691a1ae`.
All 778 canonical dev10 questions and all 14 areas remain, with the unchanged
F02 spec, one pass, no rescue candidate or amplitude search. The complete
[observed AV baseline](2026-11-07-fresh-av-live-observation.md) is the comparison.

Preparation uses a fresh artifact directory
`analysis/fresh-purpose-20261107-f02-live10-v1` and run ID
`gemma4-fresh-f02-observed-dev10-20261107-v1`. The earlier `-plan-` directory and
its placeholder preview remain as history. Candidate is
`checkpoints/fresh-code-joint-f02`, tree
`46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c`.
Full verification freeze input SHA256 is
`f897d77bf5e4bf219f6c8391cbcb738c780912d1a666ff6ac5cb4ae150fa8715`.

The existing observe init has no receipt-reuse CLI: it calls complete checkpoint
identity hashing directly. This operation used its ordinary fresh
full hash and required equality with the verified tree before attachment. No
manual skipped-hash identity or new caching feature is introduced.

The owner verifies all 89 frozen observer source files and uses that source for
its own observation imports, CPU init/merge and serving capture. It pins actual
runner image `sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8`.
TP2/DP1/MTP4, native BF16/thinking, chat32/logprob8/agent1, single gemma0 chat and
logprob endpoint, eight prefill positions, the same 16 decode offsets and 2 GiB
capture cap match the AV run. It performs the same four short OFF/ON synthetic
checks, keeps them separate from WSP, and returns to the exact pre-attachment
F02 CID. Restoring AV weights is a separate parent decision.

A CPU-only, read-only, network-isolated invocation of the same immutable runner
confirmed unchanged installed versions in all three existing cache environments
(142 WSP, 88 EvalPlus, 47 SciCode packages) and all 11 official source revision
marker SHA values. The check took 0.82 seconds. These are package-version and
revision-marker identities, not full package-content attestations. Host access
to one official-source cache directory was denied, so the intended metadata was
read through the same container context, without host root, GPU access or cache
changes. The exact read-only check program is included in the
[preparation receipt](artifacts/f02-observed-diagnostic-preparation-20261107.json).

The initial prepared plan required a resolved CID and explicit root handoff.
Those dependencies were subsequently satisfied as recorded below; the earlier
planned preview remains preserved.

## Authorized launch and matched runtime

After the deployment owner recovered from its separately recorded Compose
collision and completed apply-v2, root authorized this exact full diagnostic.
The final plan SHA is
`8ff5619bd55b93b9f967ae06cb8188ab6a015f690f9c1791f557bdea0a84e925`,
and executed owner SHA is
`ed51d091c05924a09ae3a3fa27c549d50286f0d9ed9fbc3912d1b31c0ac0a3a9`.
Ordinary F02 CID and final restoration target is
`e275b7bcc331d6bcdd82a1729c0ec0cc7a80e581a008bd273d0b1aa953e82c02`.
Owner PID 211879 acquired the exclusive global lock at start; launch epoch is
1794030031.376365. Before launch, all nine saved file stats matched full verify,
the served model was `gemma-4-31b`, active/waiting queues were zero and NVMe free
space was 74,751,864,832 bytes. No other model service was changed by this owner.

The ordinary init freshly hashed all checkpoint files and matched tree
`46680f4066f57c1596a1f4c66b67d23929c96b325943d981c6d74260cdbb3f5c`.
The attached observer CID is
`584418d20c2701e2339a6ce964332e905819c4c9454f0417d489452d9e988ee9`;
capture ID is `11228e268bd24e31b8bc3b2b82098c35`. The actual running WSP
container is `e6dcc4eb030c8d3c788ffd459fae0aca35fe09788cac748157a681abd20bd827`,
with immutable f91 image verified via Docker inspect. Its actual command has
tier10/dev, the full frozen manifest, single gemma0 endpoint for both APIs,
chat32/logprob8/agent1 and this capture. The selected count is 778.

All capture runtime, limits, sampling, decode offsets and 89 observer-source
hashes match AV exactly. Changed capture keys are only checkpoint, capture ID,
creation time and fresh output/source/cache paths. CPU observation init/merge
uses the existing CLI directly with frozen-source PYTHONPATH because the host
wrapper prepends its mutable checkout; the official WSP still uses the required
wrapper and immutable image.

The same four synthetic OFF/ON requests produced matching public-answer hashes
4/4. Total elapsed was 0.312782 versus 0.344448 seconds (+10.12%). As for AV, this
is a small sequential overhead check, not coding accuracy or a c32 overhead
estimate. They have separate capture run/item IDs and are excluded from WSP
scoring.

The [launch receipt bundle](artifacts/f02-observed-diagnostic-launch-20261107.json)
retains authorization, verified actual runner and matched observer settings.
Final per-item scores, cross-domain changes, writer completeness and restoration
are recorded below; partial counts were never ranked.

## Complete official result

All **778/778** items are scored, with zero unscored items, response errors or
observed client-trace errors/retry attempts. Code is **87.7778/100**: HumanEval+
10/10, LiveCodeBench 5/6 and MBPP+ 8/10. The unchanged failures are
`livecodebench:arc191_c`, `mbpp-plus:Mbpp/410`, and `mbpp-plus:Mbpp/576`
(the latter two pass base tests and fail plus tests). Overall is 77.86585/100.
The frozen strict-code-gain requirement is not met; this diagnostic is not an
advance or evidence of >90 coding capability.

[All 778 numeric outcomes and complete source statistics](artifacts/f02-observed-diagnostic-result-20261107.json)
are retained. No per-problem loss, unscored outcome or failed attempt is removed.
The independent paired official comparison is a separate record; these are the
F02 scores themselves.

| Area | Score / 100 |
| --- | ---: |
| knowledge | 91.6667 |
| expert | 87.5000 |
| facts | 75.0000 |
| math | 95.6250 |
| code | 87.7778 |
| instructions | 95.4167 |
| tools | 77.5000 |
| agents | 50.0000 |
| long-context | 73.0205 |
| reasoning | 74.1304 |
| commonsense | 64.0000 |
| multilingual | 85.1515 |
| vision | 75.0000 |
| safety | 58.3333 |

Loading took 105.5 seconds; `generation_seconds`/report wall time was 1,461.4 seconds (24 minutes 21.4 seconds), measured from before loading and therefore including loading, not pure generation time.
The official run counted 557 chat and 3,062 logprob API calls. TerminalBench
completed 2/2 with no trial exception: install-windows-3.11 reward 0, 268.240
seconds, 9 agent steps / 24 tool calls; portfolio-optimization reward 1, 443.964
seconds, 5 agent steps / 12 tool calls. The last response was
`bbeh:sarc_triples:141`. A subsequent read-only metrics request raced with the
authorized service restoration and got connection refused; it was a monitoring
request after benchmark completion, not a WSP item error.

## Restoration, incomplete close acknowledgement and preserved evidence

The exact original ordinary F02 CID `e275b7…` was restored healthy. The owner
completed normally in 1,999.49 seconds, including 192.29 seconds of CPU merge;
the global evaluation lock was released. Retired observer `584418…` was stopped
with PID 0, container exit code 0 and OOMKilled=false. This audit did not delete
it or deploy another model.

Both writers report **closed=false**, although queue depth and pending bytes
were zero, and no record/batch drops, quota exhaustion or writer errors were
reported. The status was preserved exactly. Safe stdout/stderr show vLLM's
`abort` shutdown with timeout 0, followed by force-killing EngineCore and both TP
workers. CaptureWorker closes through its shutdown override and atexit hook;
that termination path can prevent close acknowledgement. The deleted AV retired
container had no separately exported stdout/stderr, so the difference from AV's
closed=true cannot be established from paired logs. Container exit 0 does not
prove successful child-writer finalization.

The last timed client completion was epoch 1794031724.844392 (06:08:44.844 UTC).
Writer last completed batches were epoch 1794031711.433/435; final status mtime
was 1794031727.448934. The queue-drained receipt followed at 1794031727.504932,
and API shutdown began 06:08:47.612847. Sparse position sampling can finish
before a request, so these times do not establish that all scheduled activations
were captured. Two trial-scope client records have no request timing fields.

The standard merge and one compact-batch audit both reached gzip EOF without
error. Each rank parsed 3,479,400 activation records; TP0 additionally parsed
119,163 acceptance records. Their sums exactly match each worker's
`records_written` counter. This supports file/accounting consistency, not a
missing final close acknowledgement or proof of all intended measurements.
The capture stays `partial_or_running`.

All 778 official items and all 1,590 recorded official client traces (1,588 HTTP request-scope records plus two trial-lifecycle records) have
observations on both ranks. All 26 code items have one traced call, both ranks
and all 60 layers × attention/MLP/residual. Trial-scoped agent traces can contain
several backend calls; their expected independent call count is not inferred.
There are 6,958,800 parsed activation records: 6,934,680 official, 18,360
separately labelled synthetic records excluded, and 5,760 unlinked pre-run
records retained. Missing positions/values are never zero-filled.

[The compact capture audit](artifacts/f02-observed-diagnostic-capture-20261107.json)
retains all code coverage, identities and layer-25 exact-offset64 numeric
profiles; the full 5.84 MB audit remains on NVMe with SHA
`b47de1e83b4451fd4beed2219e7a4f899b97c6431e212ec5b922211c6a651188`.
Merged output is 2,862,310,628 bytes, SHA
`a56a03136c7da8b1bcd0b25720885a82ac030802397477c4637c8ae2e83ebdf4`.

The small [log-preservation recipe](scripts/fresh-retired-observer-logs.py) checks
exact ownership, stopped observer, healthy restored original and drained queues.
It allows missing close acknowledgement but records it explicitly. Logs are
redacted in memory against known environment/command secrets and credential
formats before private 0700/0600 host storage; no environment dump or log payload
is printed. Synthetic redaction checks and Ruff passed. The preserved stdout is
127,924 bytes and stderr 22,641 bytes, with no credential matches in this run.
Their SHA identities and chronology are in the
[shutdown/log receipt](artifacts/f02-observed-shutdown-and-logs-20261107.json).
The recipe performs no service mutation or removal.

The [bounded four-task observation readout](2026-11-07-av-f02-four-code-observation.md) adds the three unchanged code failures and declared HumanEval/30 control using existing merged data. It retains full input hashes, all sampling and close-acknowledgement limits, and the initial extraction guard failure. It adds no benchmark request.
