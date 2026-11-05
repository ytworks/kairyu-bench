# Deploying optional live observation

Run deployment commands as your normal host user. Live collectors require the
documented vLLM worker and middleware extension points; no framework patch is
needed. Use an immutable source snapshot and a fresh vLLM compilation cache for
each observer implementation/site selection. Existing cached graphs may omit hooks.
Stop benchmark traffic before changing a replica.

## Tracing through Kairyu

Kairyu must enable its existing `server.tracing` option and an OpenTelemetry
exporter so the benchmark trace reaches the model replica. If the production
configuration must remain untouched, the optional recipe
`scripts/observe-trace-gateway.py` creates an internal-only gateway with the same
image, command and configuration, enables tracing, and uses separate writable
directories. It runs as the invoking non-root UID and exposes no host ports.
It supports the standard configuration/placement mounts and rejects other layouts.

```sh
python3 scripts/observe-trace-gateway.py --source serving-kairyu-1 \
  --name wsp-observe-gateway --directory /mnt/nvme/experiment/trace-gateway
# Inspect the dry-run summary, then repeat with --apply.
```

Use `http://wsp-observe-gateway:8000/v1` from the benchmark runner on the same
Docker network. Environment and configuration values are kept private; the
receipt contains only image/container identifiers. The source gateway is never
recreated. The exact package in the source container is copied read-only so its
permissions permit the non-root UID; its contents are unchanged.

After requests finish and worker queues drain, remove only the temporary gateway
named in its receipt with `docker rm -f <name>`. Keep the capture and receipt.
A console exporter writes traces to the temporary gateway's Docker log; treat
those logs as experiment data and configure ordinary Docker retention as needed.

## Attaching and restoring a vLLM replica

`scripts/observe-docker.py` recreates one explicitly named replica with the
bench-owned worker and HTTP middleware. The default is a read-only dry-run.
Prepare the capture with `wsp observe init` first, and use the same immutable
`src/` snapshot in every replica.

```sh
python3 scripts/observe-docker.py attach \
  --container serving-gemma-0 \
  --capture /mnt/nvme/experiment/capture/capture.json \
  --source /mnt/nvme/experiment/observer-src \
  --runtime-cache /mnt/nvme/experiment/observer-cache \
  --backup-dir /mnt/nvme/experiment/observer-deploy \
  --replica gemma-0
# Inspect the plan, then repeat with --apply.
```

The source directory contains `kairyu_bench/`. Source and checkpoint mounts are
read-only; only the capture output and dedicated compilation cache are newly
mounted writable. The checkpoint must already be covered by a read-only bind in
the original container. Source, output, cache and backup paths must be absolute,
canonical and disjoint. The tool uses one cache and backup subdirectory per
container, and rejects a nonempty cache, an existing container backup, an existing
worker override, and duplicate capture replica directories.

The original immutable image ID, command prefix, entrypoint, execution user,
device requests, other host settings, and network definitions are retained.
No user override, `sudo`, or added privileged mode is used. The existing execution
user may itself be root; this tool does not change it or claim to reduce that
container's privileges. Bridge-network dynamic IP addresses can change. Existing
healthchecks are awaited for up to 900 seconds, configurable with
`--health-timeout`; without a healthcheck, readiness means only that Docker
reports the process running. Verify actual API health and capture records before
benchmarking. Auto-remove containers and shared-container network namespaces
are rejected.

On apply, the original container is stopped and renamed, but never removed.
`original.json` is a full Docker inspect backup, stored with mode 0600; it can
contain environment secrets and must not be printed, uploaded, or committed.
The normal plan and receipt omit environment values. If creation, start, or the
healthcheck fails, only the replacement identified by this operation is removed
and the original container is renamed and restarted. TERM/INT/HUP also attempt
this rollback. SIGKILL, host failure, or Docker unavailability can interrupt
recovery; preserve the private backup and state receipt for manual recovery.

Restore after traffic has stopped and the observer queues have drained:

```sh
python3 scripts/observe-docker.py restore \
  --container serving-gemma-0 \
  --backup-dir /mnt/nvme/experiment/observer-deploy
# Inspect the plan, then repeat with --apply.
```

Restore requires the saved original container and the matching observer identity.
It restarts the original container, retaining the stopped observer as recovery
evidence; it never prunes unrelated containers, images, caches, or captures.
If restoring the original fails, it attempts to return to the observer. Keep
`state.json` and verify the reported container identity and health. Worker
`closed` acknowledgement and queue drain are separate from Docker readiness;
unacknowledged captures remain partial even when their recorded depth is zero.

These deployment scripts do not run a benchmark or grant permission to resume
one. Their automated tests use a fake Docker API and synthetic metadata; real
serving acceptance and observation overhead must be checked separately.

### Capture validation before stopping a replica

The deployment plan now loads `CaptureConfig.read` from the exact selected
collector source and validates the capture before creating a backup or stopping
any container. That module's validation path is standard-library-only: it does
not import Torch/vLLM or run a model. Bytecode writes are disabled during this
check. Use a trusted, immutable bench source snapshot.

This check was added after a restart attempt configured an 8GiB capture while the
collector supported at most 2GiB. The Docker configuration was otherwise valid,
so all four workers rejected the setting after restart and the deployment tool
rolled back. Matching the source parser catches that error during dry-run.
Increasing the quota requires a reviewed collector/tool change; changing only the
JSON does not add support.

When a created observer fails, the tool attempts to save at most 64KiB of the
Docker log response from its last 200 lines to `failed-container.log` (mode 0600)
before removing that owned container. Docker stream framing can be present and
the bounded excerpt can be truncated. It may contain sensitive diagnostic
values; it is never printed or uploaded. Failure to retrieve a log does not
prevent the rollback, and the receipt records whether the excerpt was saved.

### Releasing GPUs for a finite probe

After a benchmark finishes and every writer queue drains, `restore --leave-stopped
--apply` restores the original container name/configuration while keeping both the
original and retired observer stopped. This avoids loading the original model only
to stop it again for a separate finite probe. The receipt says `restored_stopped`.
Normal `restore --apply` keeps its original restart behavior. Restart serving via
the owning example lifecycle after the probe. Before Compose manages that service
again, preserve any required retired-observer logs and remove only its exact owned
container ID from the receipt, so old Compose service labels do not create duplicate
service instances. Never prune unrelated containers.
