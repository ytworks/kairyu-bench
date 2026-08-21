# Repository agent instructions

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
exec ./kairyu-bench run http://host.docker.internal:8003/v1 \
  --only livecodebench-pro \
  --run-id livecodebench-pro-full
'
```

Do not add `--limit`. Do not stop a healthy full run merely because it is slow.

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

