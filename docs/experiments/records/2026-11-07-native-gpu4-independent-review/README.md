# Independent GPU4 owner review

One P1 cleanup-order defect was found in source c70af9c1… and is fixed in
9a8d9255312a97ae2762ec97c34c9701ac5d64dcb0dc273e01f8d7f6700b7183.
No additional concrete source blocker was found after the fix. This is a
source/plan and pure CPU mock review, **not GPU execution validation**.

The old finish checked both collector names/CIDs before waiting for the Docker
client. A daemon-side container appearing during that wait could invalidate the
absence proof, yet original gemma-2 start still followed. The same independent
mock now shows client completion → fresh cidfile and daemon lookup → owned
collector stop/kill → original exact-CID start. The author's added regression
also covers creation of the previously absent cidfile during client.wait.
Both old/new source snapshots and the complete independent reproduction are
retained here; no real Docker/API/GPU/service calls were made.

Reviewed invariants: exact train12 A..F singleton folds and full shared frames;
transfer exclusion; all9 AV prior-content hashes plus current stat continuity;
complete executable source pins; same immutable image; GPU4-only container,
readonly AV/input and NVMe output; 900-second total two-collector deadline
including model loads; nonblocking global/GPU4 locks; only gemma-2 exact CID
stop/start and other26 exact snapshot protection. Unknown ownership, unavailable
Docker or unfinished client prevents kill/restart; failed/partial work is retained.
The original serving health restoration has its separate600-second budget.

The existing unapproved plan f47f6a6c… is historical: old owner pin and null native
behavior result prevent execution. A newly bound result/source plan, successful
full preview and root's exact-SHA authorization remain required. Review does not
authorize a run. CPU v5 evidence records13 tests plus Ruff passed against the
fixed source; the fixed owner has not yet been exercised on a GPU.

Reproduction and limits: run independent-late-create-check.py.txt with existing
CPU Python only; both source files are pinned. Its first draft incorrectly
ignored the mock docker ps name filter; this yielded a failed restored assertion
and was corrected in the mock, not the owner. That draft and refinement record
remain. An earlier mock setup chose nonexistent infra/tmp and failed before the
mock; its receipt records the zero-service effect. These are diagnostic setup
failures, not model/benchmark trials. Full hashes and plan state are in review.json.

V2 follow-up: plan 8edba22f5aa38ec5eebf443a3b4b2eab2794b23c9968fcd0c09d13d003f20316 now binds fixed9a8 source,79 source pins and completed native behavior identity. Independent validate(plan) passed all12 IDs/frames/pins using CPU/file reads only. Work is absent and approved=false; exact root authorization remains required. See review-v2-plan.json. No Docker/API/GPU/service call occurred.
