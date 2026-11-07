# Matched comparison through existing Compose replicas

[redeploy-replica.py](scripts/redeploy-replica.py) is an experiment recipe around an
operator-provided example `control.py`. It previews one declared chat service by
default. It does not add a serving stack, update the global homogeneous-checkpoint
record, or run a benchmark.

Before the first mutation, a shared baseline records all declared chat replicas.
The recipe compares the target's current container ID, image content ID, effective
command and environment, effective user and working directory, CPU/GPU allocation,
IPC, mounts, and checkpoint label. Healthcheck and network identity are also
compared after startup.
It refuses a changed baseline. Compose configuration is rendered again immediately
before applying. Environment values remain in memory; receipts contain hashes.

```sh
PYTHONPATH=src python docs/experiments/scripts/redeploy-replica.py \
  --control /absolute/example/control.py --service gemma-0 \
  --candidate /absolute/checkpoints/candidate \
  --manifest /absolute/manifests/candidate.json \
  --creation-receipt /absolute/creation-receipt.json \
  --baseline /absolute/experiment/replica-baseline.json \
  --output /absolute/experiment/preview-candidate
```

The creation receipt must contain `jobs`, with exactly one matching
`candidate`, `manifest`, `spec_sha256`, `exit_code: 0`, `started_epoch`, and
`seconds`. The native editor manifest supplies every file's size and content hash.
This path explicitly relies on current private-file ownership/link counts and
file ctime/mtime preceding manifest publication. It assumes a monotonic filesystem
clock and no privileged ctime manipulation. It is **not** a new full payload hash
or a historical post-copy stat snapshot. The receipt records that limitation.

For an already frozen checkpoint, use `--freeze` instead of `--manifest` and
`--creation-receipt`. The freeze supplies `checkpoint_path`,
`candidate_inventory_stat`, `all_checkpoint_files` (path/size/sha256), and
`control_tree_sha256`. Every saved stat must still match.

Review the preview, then use a **new output directory** and explicit `--apply`.
Only the named service is recreated through the example's existing Compose
command with `--no-deps`. A running target must have zero running/waiting requests.
The operator remains responsible for exclusive GPU ownership when a target was
already stopped. No other workload is stopped automatically.

On an apply failure, the recipe attempts to recreate the same service with its
previous checkpoint and then restore its previous stopped/running state. Signals
are handled during this interval; SIGKILL, host loss, or daemon loss can still
require operator recovery. Rollback success or failure is recorded. The helper
does not coordinate independent operators who ignore its per-service lock.

For mixed weights, benchmark **fresh direct replica IPs**, including a matched
control replica with identical image, BF16, TP, MTP, sampling, concurrency, and
task manifest. Never send a comparison through the mixed gateway. Use the common
benchmark image and explicit NVMe result paths. Record each service's checkpoint
identity and runtime receipt separately. Restore the uniform serving arrangement
with the example's normal control workflow after the comparison.

This saves repeated model loads; it does not eliminate checkpoint copies or prove
a speedup. Parent-supervised measurements must establish throughput and score.
