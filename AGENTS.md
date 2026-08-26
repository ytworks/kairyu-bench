# Repository agent instructions

## Benchmark concurrency

LiveCodeBench Pro, SWE-bench Pro, and Terminal-Bench default to four concurrent
problems. Override their bounded concurrency (1 through 16) with
`KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS`,
`KAIRYU_BENCH_SWEBENCH_PRO_WORKERS`, and
`KAIRYU_BENCH_TERMINAL_BENCH_WORKERS`, respectively. Each harness immediately
refills a slot when one problem finishes; completion order is not guaranteed.

Terminal-Bench's installed agents (`claude-code` and `codex`) additionally honor
`KAIRYU_BENCH_TERMINAL_BENCH_SETUP_TIMEOUT_MULTIPLIER` (integer 1 through 16,
default 4). Harbor allows agent setup 360 seconds times this multiplier; the
default covers slow `apt-get`/npm installs under four-way parallelism. The
`terminus-2` agent has no install step and ignores this variable.

## LiveCodeBench Pro full run

Use the repository wrapper, `./kairyu-bench`, for the official run. Do not
replace it with an ad-hoc Python invocation.

The pinned dataset contains 1,404 rows across six overlapping splits. The
official `get_problem_set()` deduplicates them by `problem_id`, producing 864
problems. A full run therefore means 864 evaluated problems with no `--limit`.

### Preconditions

- The local OpenAI-compatible API is available at `127.0.0.1:8003/v1`.
- From the runner container, address that API as
  `http://host.docker.internal:8003/v1`.
- `HF_TOKEN` is defined by `/home/y-takagi/.bashrc`, and its Hugging Face
  account has accepted the gated LiveCodeBench Pro dataset terms.
- Never print, log, or persist the token in the repository or results.
- Docker is available. The wrapper uses `--privileged` and mounts the Docker
  socket because LightCPVerifier starts a nested task container. Obtain the
  user's explicit approval for those two capabilities before starting.

### Storage

Keep benchmark caches on NVMe, not on the root filesystem. Create a dedicated
directory beneath `/mnt/nvme/kairyu/bench-cache/` and export it as
`KAIRYU_BENCH_CACHE_DIR`. Preserve `results/` in the repository. Do not prune
images or remove containers used by the current Kairyu API or Open WebUI.

The wrapper honors `KAIRYU_BENCH_CACHE_DIR`; leave it unset only for small
smoke tests where the repository `.cache/` is intentional.

### Command

Run from the repository root in an interactive Bash so `.bashrc` supplies the
token. Choose a unique run ID if the example already exists.

```bash
/bin/bash -ic '
set -eu
test -n "${HF_TOKEN:-}"
export KAIRYU_BENCH_CACHE_DIR=/mnt/nvme/kairyu/bench-cache/livecodebench-pro-full
export KAIRYU_BENCH_LIVECODEBENCH_PRO_WORKERS=4
export KAIRYU_BENCH_LIVECODEBENCH_PRO_RETRIES=3
# When KAIRYU_BENCH_DOCKER_SOCKET points at a separate Docker daemon, also set
# KAIRYU_LIGHTCPVERIFIER_HOST to that daemon container's reachable IP address.
exec ./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only livecodebench-pro \
  --run-id livecodebench-pro-full
'
```

Do not add `--limit`. Do not stop a healthy full run merely because it is slow.
The Pro shim keeps four problems active and immediately starts the next problem
when any one finishes. Completion log indices may therefore arrive out of order.

### Verification and progress reporting

Immediately after launch, verify all of the following:

- `results/<run-id>/run.json` exists and has status `running`.
- `model_id` is the model currently returned by the local API.
- `results/<run-id>/logs/livecodebench-pro.log` reaches
  `Generating solution 1/864` and then `Judged solution 1/864`.
- The LightCPVerifier container is healthy while judging.

Count completed problems from `Judged solution` log lines and cross-check that
count against the length of
`results/<run-id>/raw/livecodebench-pro/benchmark_result.json`. Report the
completed/864 count, accepted count, current problem, elapsed time, API health,
and root/NVMe free space. While actively monitoring, send progress updates at
least once per minute and immediately report retries or failures.

On completion, require `run.json` and the normalized result to say `completed`,
then report the official pass@1 score from `report.md` with the accepted and
evaluated counts. Preserve the complete result directory. After successful
report generation, the dedicated NVMe cache may be removed only after
rechecking its exact path and confirming it is not shared with another run.

## SWE-bench Pro full run

Use `./kairyu-bench` with `--only swe-bench-pro`. The pinned public test split
contains 731 problems, so a full public run means 731 generated patches and 731
official evaluations with no `--limit`. The official evaluator is the pinned
`scaleapi/SWE-bench_Pro-os` source and its `jefzda/sweap-images` task images.

Run nested task containers through a dedicated privileged Docker-in-Docker
daemon whose data root and Unix socket are both beneath `/mnt/nvme/kairyu/`.
Export that socket as `KAIRYU_BENCH_DOCKER_SOCKET`, and export
`KAIRYU_BENCH_CLEAN_TASK_IMAGES=1`. The Pro harness keeps up to
`KAIRYU_BENCH_SWEBENCH_PRO_WORKERS` problems active concurrently, immediately
claims the next problem whenever one worker finishes, then removes each
completed problem's task container and image. Set it to the API's tested
concurrency (four for the current four-replica local API). This
keeps no more than the worker count of task images live and avoids
using or pruning the host Docker daemon that serves the Kairyu API and Open
WebUI. Obtain explicit user approval before using the privileged runner,
privileged Docker-in-Docker daemon, and shared Docker socket.

Keep the benchmark cache on NVMe, preserve `results/`, and mount both the
results directory and cache into the dedicated daemon at their identical host
paths so the official evaluator's workspace bind mounts resolve correctly.

Run from an interactive Bash so `.bashrc` supplies `HF_TOKEN`, using unique
paths and a unique run ID:

```bash
/bin/bash -ic '
set -eu
test -n "${HF_TOKEN:-}"
export KAIRYU_BENCH_CACHE_DIR=/mnt/nvme/kairyu/bench-cache/swe-bench-pro-full
export KAIRYU_BENCH_DOCKER_SOCKET=/mnt/nvme/kairyu/docker/swe-bench-pro-full/run/docker.sock
export KAIRYU_BENCH_CLEAN_TASK_IMAGES=1
export KAIRYU_BENCH_SWEBENCH_PRO_WORKERS=4
exec ./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only swe-bench-pro \
  --run-id swe-bench-pro-full
'
```

Immediately verify `run.json` is `running`, its `model_id` matches the local
API, and `selected.jsonl` and `instance-ids.txt` both contain 731 problems.
Count progress from the number of `SWE-bench Pro completed` log lines and
cross-check against the number of per-item `evaluation/eval_results.json`
files. Completion numbers may arrive out of order with multiple workers.
Report resolved count, current instances, elapsed time, API/Docker health, and
root/NVMe free space at least once per minute while monitoring. Report cleanup failures
immediately. Do not stop a healthy full run merely because it is slow.

On completion, require all 731 prediction IDs and official boolean outcomes,
`run.json` and the normalized result to say `completed`, and report the
official resolved percentage from `report.md` with resolved/evaluated counts.
Preserve the full result directory. Only then may the dedicated cache and
Docker data root be removed after their exact paths are rechecked.
