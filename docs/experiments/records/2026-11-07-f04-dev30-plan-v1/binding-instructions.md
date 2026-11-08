# F04 canonical dev30 binding handoff (not executable yet)

Reuse the unchanged fresh-av-dev30 owner. The unbound plan has no final tree, checkpoint verification, four CIDs, closure pins or capture; no owner preview is attempted until those exist. The checkpoint path is the faststore bundle's checkpoint subdirectory, never the bundle root.

1. Require the completed comparable-parent decision and root authorization. If the parent changes, retain this unexecuted template and record parent_changed.
2. Read the actual faststore manifest and successful verify receipt, pin their SHA, require the frozen spec SHA and exact logical AV parent. Check all nine files against manifest.candidate_inventory_stat, including shared.safetensors inode/ctime/nlink; use all_checkpoint_files hashes. This is fresh private hashes plus prior full shared hash/stat continuity, not a new full hash.
3. Construct the checkpoint part using existing Checkpoint(checkpoint).identity(hash_files=False), then join every path/size to the manifest hash rows and derive the canonical sorted path/size/sha256 tree; require equality to manifest.control_tree_sha256. Preserve complete-model scope semantics. Copy the existing capture template, set a fresh capture_id/created_at, output path and this identity. Write actual http-trace/capture.json only now. Separately record all provenance, before/after full stat continuity and no payload rehash in http-trace-identity-check.json.
4. Bind four exact ordinary healthy CIDs, four per-replica closure path/SHA pairs, combined closure, endpoint closure, prior owner PID and normal global release. The gateway and all direct logprob endpoints must serve F04; preserve TP2/MTP4/image/native flags and 23 unrelated services.
5. Preserve launch-plan-unbound.json; write launch-plan-approved.json only under root authorization. Replace only final label and candidate-specific evidence in the planned command. Keep all 2,338 items, 14 areas +96 probes, 8 environment values, limits, immutable f91 runner and 256/32/4 concurrency unchanged.
6. Run the existing owner once without --run for CPU preview, then send the approved plan SHA and preview to root. Root launches once with --run. No baseline rerun, extra finite/proxy/transfer or new owner feature.

Future root command after binding only:

    PYTHON /path/to/docs/experiments/scripts/fresh-av-dev30-20261107.py --plan N4/launch-plan-approved.json --run

All current paths/arguments and source pins are in launch-plan-unbound.json. Model requests, service/lock changes, actual capture creation and owner preview/run in this preparation: zero.
