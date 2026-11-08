# Planning-only iteration efficiency audit

No implementation, proposal, collection, replay, model call or service action was
performed for this audit.

The existing [finite replay](../../src/kairyu_bench/wsp/weights/finite_replay.py) load_plan accepts 1–8 fixed specs. replay_sequence
loads the model once, evaluates the base, each candidate, and restored base.
ParameterEdits.apply first restores the selected union from the original parent,
then applies that candidate and reads the union back exactly. Six candidates
therefore require eight sweeps rather than eighteen in six separate runs.
They are alternatives from one parent, not six cumulative adoptions.

The current one-candidate owner explicitly rejects more than one spec in its
preflight and summary check. Minimal reuse is a count-aware owner and expectation
map; the numerical replay needs no new implementation. Failure stops the
remaining candidates and attempts restoration, so incomplete results must stay
partial rather than silently retried. New adoption requires new-parent
derivatives before any further cumulative proposal.

AC's observed base/candidate/restoration sweeps were 20.393/18.986/18.970 seconds,
with 15.865 seconds of other replay time. Reusing those timings suggests roughly
169 seconds for six candidates versus 445 seconds across six independent
replays, before additional savings from repeated serving restoration. This is an
illustration from existing timing, not a new benchmark or isolated speed claim.

The four S collectors performed 136 native-LL + 272 native-margin + 136
matrix-LL + 272 matrix-margin = 816 forward/backward passes. Their reported
compute times total 263.128 seconds. A future optional combined collector could
perform one correct and one signed foil VJP per authored case, 272 total:
request native gradients while the existing matrix output hooks capture sampled
derivatives during that same backward. Correct gradients feed both objectives;
signed foil gradients feed margin only.

Preserve pair-derived padding, per-answer fold-size scaling, native
BF16-gradient-to-FP32 accumulation order, and FP32 matrix products with FP64
sums. Do not subtract two already-aggregated fold arrays to construct margin.
Release each answer graph before the next; no full matrix gradient buffers are
needed. The same four output schemas, source/frame checks and eight existing
zero-additive probe cells can remain. This changes collector orchestration and
provenance, so it is not silently identical to the old implementation.

The separate native collector peaked at 84.945 GiB reserved (75.222 GiB
allocated); matrix peak was 80.648 GiB reserved. At 509 tokens, the sampled-input
hook copies alone total about 0.40 GiB. Joint graph requirements may add more:
the combined memory fit and speed are unmeasured, not guaranteed by those
separate peaks.

Prefer the existing multi-candidate replay first. Consider the optional fused
collector when a newly adopted parent already needs fresh observations, avoiding
a redundant real-model collection solely to demonstrate speed.

These are planning notes retained with the experiment record; no batch or fused
collector was implemented or executed. Any adopted-parent change requires new
current-parent derivatives. Evidence: iteration-efficiency-audit-v1.md SHA
73371adaa817d573a9994f49298815b4abfe78665a0e5c5aa890fc98e80033b3.
