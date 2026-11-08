# 2026-11-07: single-replica Compose scope independent review

Reviewed at 2026-11-07T05:32:19.134046+00:00 by the independent audit agent. This is a source/diff review; it performed no Docker mutations, model calls, benchmark runs, or additional test executions.

The original failure involved a stopped, renamed observer container retaining the target's Compose project/service labels. Checking only the canonical container name did not cover Compose's selection. The reviewed [helper](scripts/redeploy-replica.py) now queries all matching containers, including stopped ones, requires the exact one expected CID at preview and immediately before apply, and refuses multiple members during rollback. The CLI [fixtures](../../tests/test_redeploy_replica.py) cover duplicate preview/apply refusal, a duplicate appearing before apply, ordinary single-member success, and refusal of a second Compose call when rollback membership is ambiguous.

Both requested review corrections are present:

- The helper forcibly sets COMPOSE_REMOVE_ORPHANS=0 and records compose_remove_orphans=false. The fixture deliberately starts with value 1 and asserts value 0 for both ordinary apply and rollback. This closes the inherited environment route to deleting other project orphans.
- Zero-member rollback is accepted because recreation may remove the former container before failing. Its fixture uses genuinely empty docker-ps stdout, fails the first Compose call, requires the second call to use the old checkpoint/path and tree, and requires rollback_completed=true. An ambiguous two-member rollback still records failure and makes no second Compose call.

For the inspected actual control, COMPOSE_PROJECT_NAME agrees with the project filter, the Compose command uses its explicit compose.yaml, and target services have no scale/replicas override. The reviewed changes address the observed fault and the two review findings. The query and Compose operation remain non-atomic: the coordinated global owner must exclude other actors. Rollback establishes the prior runtime configuration, checkpoint and healthy state; it does not promise resurrection of the original CID.

Reviewed helper SHA-256: 2e9da820fe30ba9b2436fb600cf14a533b1c496b2fed48ac3649fffb9d2bc9ae.
Reviewed tests SHA-256: 8d065622c1487ef4824d59e7a0b9c4e3b8208f1f1f7a4cc3888245d8107479e5.

The implementing agent reports 3 tests passing in 0.037 s, Ruff PASS and diff-check PASS. That execution report is separate from this independent source review; it is not presented as a second test run or a real serving validation.
