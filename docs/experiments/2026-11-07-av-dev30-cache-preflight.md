# Canonical AV dev30 cache preflight

2026-11-07. CPU/read-only preparation; no model calls, GPU/serving changes,
`wsp prepare`, pulls, pruning or exclusions. This is not a capability
result. Serving ownership is separately covered by the
[homogeneous AV plan](2026-11-07-av-dev30-homogeneous-serving-plan.md).

The full canonical selection has **2,338 items, 14 capability areas and 96
diagnostic items**. Code has **90: HumanEval+ 30, MBPP+ 30, LiveCodeBench 20,
SciCode 10**. Agent tasks are Terminal-Bench 6 plus SWE-bench Pro 4.
[All selected IDs and 32 source counts](evidence-2026-11-07-av-dev30-cache/runtime.json)
are retained. Manifest SHA:
`a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b`.
A 90-code-only screen is not canonical dev30.

## Data and environment evidence

**All 2,338 input records exist in the cache.** In the immutable f91 runner,
the official loaders read **29 sources / 2,262 selected records** offline
with zero missing IDs. Three sources required independent direct-file checks:

- LiveCodeBench 20 IDs: both pinned test5/test6 JSONL files exist, 557,699,297
  and 134,303,240 bytes; full SHA256 and ID matching are recorded. Its official
  JSON loader still requests HF revision metadata in offline mode.
- Japanese Wikipedia diagnostic 40 IDs: all exist in the pinned
  176,347,554-byte Parquet and satisfy the same >=1,500-character filter.
  Offline inference requests config `default-8decd48c1a817b17`, whereas
  the existing normal-run cache is `default-5a7ad5581de16bef`.
- MRCR 16 raw indices: all exist in the pinned Arrow with `n_needles=8`.
  The tokenizer/bin calculation was not reproduced under network isolation:
  `o200k_base.tiktoken` was absent from inspected NVMe tmp/xdg and fresh
  f91 /tmp and /root/.cache directories, so the normal loader requested it.

These checks prove data presence, not a substitute official loader/scorer.
The audit is **not strict-offline ready**. The next normal online run retains
the same official protocol, allowing existing HF metadata requests and the
standard tokenizer fetch; no source is excluded. Direct-check results are in
[direct.stdout.json](evidence-2026-11-07-av-dev30-cache/direct.stdout.json).

**SciCode H5** is 1,049,345,865 bytes. Its newly computed SHA256 matches:
`48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890`.
Using the existing SciCode interpreter and pinned official
`process_hdf5_to_tuple`, all **39 tested steps / 128 targets** for the
10 selected problems were readable. No model program or reference answer
was executed. This checks data/environment, not SciCode correctness.

Runner:
`sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8`.
WSP/EvalPlus/SciCode environment versions (142/88/47 distributions) and all
11 source revision-marker hashes exactly match the fresh AV live10 receipt.
This is package-version/marker identity, not a full package/source-content hash.

Agent ready markers match physical cache paths. Harbor has 90 distributions
(harbor 0.21.0, litellm 1.106.6). The SWE harness selects the
`...-litellm-1.99.0` environment: 81 distributions, datasets 5.0.0,
docker 7.1.0, pandas 2.3.2, mini-swe-agent 2.4.6, litellm 1.99.0.
The older 84-distribution cache is retained but is not selected by that branch.
The separate SWE harness HF cache has 731 rows and all four selected IDs.
[Package inventory](evidence-2026-11-07-av-dev30-cache/agent-packages.json)
and [metadata](evidence-2026-11-07-av-dev30-cache/metadata.stdout.json)
record the exact paths and versions.

## Ten cached task images

All ten images are present on
`unix:///mnt/nvme/kairyu/docker/wsp/run/docker.sock`. Names are obtained
through the existing `_task_images` reader from Terminal task.toml and
pinned SWE `dockerhub_tag` fields. Each was checked with
`docker -H SOCKET image inspect IMAGE`. Full tags, IDs and repo digests:
[task-images.json](evidence-2026-11-07-av-dev30-cache/task-images.json).
No image pull is needed for these tasks.

| Task | Image ID prefix | Present |
|---|---|---|
| instance_ansible__ansible-11c1777d56664b1acb56b387a1ad6aeadef1391d-v0f01c69f1e2528b935359cfe578530722bca2c59 | `857c630e3ff2619c` | yes |
| instance_qutebrowser__qutebrowser-ed19d7f58b2664bb310c7cb6b52c5b9a06ea60b2-v059c6fdc75567943479b23ebca7c07b5e9a7f34c | `eb6c87d3587835bc` | yes |
| instance_internetarchive__openlibrary-431442c92887a3aece3f8aa771dd029738a80eb1-v76304ecdb3a5954fcf13feb710e8c40fcf24b73c | `af9fc5aab7f02147` | yes |
| instance_NodeBB__NodeBB-da0211b1a001d45d73b4c84c6417a4f1b0312575-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e | `4752997a9b70ee41` | yes |
| portfolio-optimization | `b0ef263fdd85ca60` | yes |
| install-windows-3.11 | `c476efafe0214c2c` | yes |
| bn-fit-modify | `e22baa862c29eb50` | yes |
| cobol-modernization | `14e6e693be70ee0c` | yes |
| reshard-c4-data | `a1ea118e6dbb4420` | yes |
| polyglot-c-py | `e1956576c394084e` | yes |

## NVMe and normal online tokenizer behavior

Host `docker info --format '{{.DockerRootDir}}'` returns
`/mnt/nvme/docker`; `findmnt` resolves this to `/dev/md2`,
ext4, mounted at `/mnt/nvme`. Thus the normal runner rootfs and
`/tmp/data-gym-cache` are NVMe-backed. Dedicated WSP DinD Docker data,
socket directory, benchmark cache and result binds are also on NVMe.
The final storage snapshot has **71,321,071,616 bytes free**. This is not a
reservation or an estimate of future task writable-layer growth.

The f91 tiktoken 0.13.0 implementation fixes:

- URL: `https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken`
- Expected SHA256:
  `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`
- URL-derived cache key:
  `fb374d419588a4632f3f557e76b4b70aebbca790`.

The normal wrapper does not override HOME, TMPDIR or TIKTOKEN_CACHE_DIR.
Its normal tokenizer fetch can therefore remain unchanged; no wrapper or
serving change is required to satisfy NVMe storage. The preflight did not
download it. [Docker storage](evidence-2026-11-07-av-dev30-cache/host-docker-storage.json),
[exact tiktoken function](evidence-2026-11-07-av-dev30-cache/tiktoken-source.py.txt)
and [safe host snapshot](evidence-2026-11-07-av-dev30-cache/host-state.json)
contain supporting evidence without environment secrets.

## Minimal canonical command, not executed

Require the serving owner's proof that all four replicas provide the same
frozen AV tree, TP2 x DP4, MTP4, normal settings, healthy/idle, under the
global evaluation lock. The observed mixed F02/AD/original/AR gateway is not
an eligible baseline. The bound command enables client HTTP tracing only. No CaptureWorker or
CaptureMiddleware is deployed, and no server activation claim is made.
Later comparisons must match these runtime conditions. The official runner
uses its existing capability discovery to select gemma-4-31b from the gateway.

```bash
R=/mnt/nvme/kairyu/wsp/gemma4-coding-20261104
BENCH_REPO="$R/worktrees/kairyu-bench-weight-tools"
cd "$R/results"
export KAIRYU_BENCH_RUNNER_IMAGE_ID=sha256:f91bdcf997af6d7b194bec229812fd018277158f6795de7d91e58448936e7ad8
export KAIRYU_BENCH_CACHE_DIR=/mnt/nvme/kairyu/bench-cache/wsp
export KAIRYU_BENCH_DOCKER_SOCKET=/mnt/nvme/kairyu/docker/wsp/run/docker.sock
export KAIRYU_BENCH_DOCKER_NETWORK=gemma-4-31b-8gpu_default
export KAIRYU_BENCH_CLEAN_TASK_IMAGES=0
"$BENCH_REPO/kairyu-bench" wsp run http://kairyu:8000/v1 \
  --tier 30 --split dev --label 5d056eb03f954db80313a85167153c875fe5e436e4f97c281a3f55202d9b3eaf \
  --run-id gemma4-fresh-av-dev30-dp4-20261107-v1 \
  --chat-concurrency 256 --logprob-concurrency 32 --agent-concurrency 4 \
  --logprob-endpoint http://gemma-0:8000/v1 \
  --logprob-endpoint http://gemma-1:8000/v1 \
  --logprob-endpoint http://gemma-2:8000/v1 \
  --logprob-endpoint http://gemma-3:8000/v1 \
  --observe-config "$R/analysis/fresh-purpose-20261107-av-dev30-dp4-v1/http-trace/capture.json" \
  --results-dir "$R/results"
```

Require a fresh run ID. Agent concurrency four is shared across sources, not
four workers per source. Tier30 adds SciCode/SWE/LongBench; its code source
mean cannot be ranked directly against tier10's three-source mean.

## Reproduction and all preparation failures

The [recipe](scripts/fresh-wsp-cache-preflight.py) reuses existing manifest,
adapter and task-image readers. It ran in f91, network none, read-only root
and cache, no GPUs, two CPU quota, 12 GiB memory, all capabilities dropped
and no-new-privileges. Only new NVMe output/tmp were writable. Container
default root was needed for root-owned 0700 cached venv directories;
no host root or sudo was used. HOME was not redefined.

[Exact commands and every outcome](evidence-2026-11-07-av-dev30-cache/attempts.json)
preserve four preparation attempts:

1. v1 output permission error after metadata/H5 reads, 2.359 seconds.
2. v2 UID1003 could not traverse the root-only cached venv, exit126.
3. v3 audit completed in 6.698 seconds but read-only advisory locks blocked
   most HF loaders. Every source error remains recorded.
4. v4 audit completed in 13.572 seconds after relocating only advisory lock
   paths into new output, retaining actual locks and read-only cached data.
   This exposed the three remaining offline limitations. Exit0 means the audit
   report was written; `all_selected_rows_available=false` explicitly
   records failure of the complete official offline-loader check.

These are not WSP/model reruns. The first recipe is preserved with its verified
original hash in
[preflight-source-v1](evidence-2026-11-07-av-dev30-cache/preflight-source-v1.py.txt).
Exact one-off CPU programs are preserved for
[metadata](evidence-2026-11-07-av-dev30-cache/metadata-program.py.txt),
[direct-file/H5 checks](evidence-2026-11-07-av-dev30-cache/direct-program.py.txt),
and [agent packages](evidence-2026-11-07-av-dev30-cache/agent-package-program.py.txt).
They use the recorded v4 container command, replacing its Python payload with
`-c PROGRAM`. No preparation hook is called. Ruff passed for the recipe;
actual cache/adapter/H5 reads provide the relevant validation. No additional
benchmark or broad test suite was run.
