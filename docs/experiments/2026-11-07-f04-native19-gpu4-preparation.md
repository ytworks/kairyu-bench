# F04 parent and explicit native cohort preparation

The existing GPU4 owner now accepts an explicit native checkpoint/tree and ordered singleton cohort while retaining its previous AV/12 defaults. Native collection, objective math, frame policy, image, resource bounds, global/GPU locks, owned-client cleanup and delayed-create protection are unchanged. The historical finite branch retains its AV-bound scope and is not being executed. Source before this revision is preserved byte-for-byte (SHA `39707865f4cc4851290d218d0f4cb6a3fa8768a940a124a9f0ea953ee0d2c5d7`).

Two new plan fields specify the current scientific input:

```json
{
  "checkpoint_tree_sha256": "7b9e9488e3cd4e0b6c5f61ae341e5eeccdf5bd31c46d1c8354950dbc29d0f924",
  "native_cohort": {
    "count": 19,
    "rows": [{"purpose": "A", "fold": 0, "id": "behavior-A1"}]
  }
}
```

The displayed row is illustrative; the actual plan contains every ordered row, A–I with two singleton folds each and J with one. The exact cases SHA and train-only metadata are checked against that declaration; omissions, duplicate IDs/folds, changed frame membership and transfer rows fail before service work. The new checkpoint path must be an ordinary NVMe checkpoint directory with all nine content hashes, canonical tree and current file stats matching the pinned identity. F04 shared hash provenance remains its prior full build hash plus stat continuity, not a new whole-model hash.

The target remains the named ordinary `gemma-2` service. Its current exact CID comes from the new 27-container snapshot and is used consistently for metrics, stop, verification and restart. In this plan it is `fc3e14a0b216e622004db9b0cf754c7d33e4410d2ab63a28b762ef32607327ca`; the other 26 exact snapshots stay protected. All 421 native tensors/1,331,516 values remain the required observation inventory. No ordinary model deployment is needed.

Both existing native collector metadata previews passed (0.319/0.315 seconds), using the same frozen collection source as AV and the F04 checkpoint, train19 SHA `74e1cd2000fda0abdc03f4454b18f4231b9891360af266391744e13976f199f1` and frame SHA `e9aee3b1e3af5929b47fc3148b090785147d18c30e1ee706691320cc9810245b`. The full owner plan remains **unapproved and not executable**, awaiting a pinned identity for F04's native generation plus CPU scoring. No GPU, model request, stop or restart occurred during this preparation.

The owner CLI is unchanged: `python fresh-av-native-gpu4-20261107.py --plan PLAN` performs metadata validation only. Only a separately root-approved exact plan may use `--run --approved-plan-sha256 SHA`. The 900-second deadline covers the two collector subprocesses (including model loading) after the ordinary target is stopped; stop and health restoration have separate existing deadlines.

Focused validation: 23 tests passed in 0.422 seconds (previous 22, plus one new 19-case/non-AV-parent preview-and-omission regression). The existing exact-restart test now uses a new target CID and proves the old AV CID is never started. All existing late-create/unknown-owner/partial finite/restoration failure tests remain. Ruff passed. Current owner SHA `56dca4ce4b1508084664a9e8ce9e9a2bb3b7f528d840fb35fb352c07e92f0bd9`; test SHA `9487f5f77f3a1e0775b7507afb9423695b13a30432f9e0c2e18032474776996b`. [Exact sources, diff, unapproved plans, previews and validation records](records/2026-11-07-f04-native19-gpu4-preparation/) are retained.

The subsequent F04 generation and CPU execution completed for all 19: 16 passed, A1 failed 0/16, F2 and J1 each failed 4/16, unknown 0; 264/304 individual fixtures passed. CPU sandbox cleanup returned to 27 containers with no labeled workers remaining. Root bound those immutable receipts in `root-native-behavior-identity.json` and produced the final metadata preview; the earlier unbound plans remain. Independent read-only owner review reported no blocking finding. Launch and GPU results are separate from this preparation.

F2 and J1 deliberately share the correct answer and public prompt but differ in foil. Their correct-LL goals are semantically duplicated, not asserted to be numerically identical: serial preparation pads to the larger correct/foil frame, so actual widths and measured gradients must remain visible.

The separately approved GPU execution subsequently completed: [all 19×2 measurements, exact restoration, resource counts and local dependency results](2026-11-07-f04-native19-sensitivity-results.md). The preparation source remained unchanged.
