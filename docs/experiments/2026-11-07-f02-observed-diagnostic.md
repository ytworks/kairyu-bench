# F02 matched observed diagnostic, 2026-11-07

Status: prepared, not executed. The saved F02 checkpoint is fully verified;
serving deployment, healthy candidate CID and exclusive-lock handoff remain
owned by the deployment agent. No candidate observation or benchmark request
has been sent by this preparation.

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
identity hashing directly. This operation will therefore use its ordinary fresh
full hash and require equality with the verified tree before attachment. No
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

The execution plan remains blocked by its own planned status and unresolved
serving CID. Starting the owner requires the root agent's explicit handoff after
deployment and release of the global evaluation lock; the owner then reacquires
the lock, checks exact CID and idle state, and performs the full run. This is an
internal execution dependency, not a new user approval request.
