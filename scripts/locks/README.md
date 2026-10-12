# DeepSWE runtime lock

`deepswe-pier.txt` targets CPython 3.12 on Linux. It combines the dependencies
exported from `datacurve-ai/pier@0c802fc067a425345b24d1c69411aa98acf61a1d/uv.lock`
with mini-swe-agent 2.4.6, then pins every resolved package and distribution hash.
Pier itself is installed without dependencies from that Git commit. The mini
source revision is `a83fcae82d2a08f0ee0c688f9d137b3566c097f8` (2.4.6).

`src/kairyu_bench/data/deepswe-agent-lock.json` pins the mini agent's dependency
closure inside task images, including LiteLLM 1.83.0 and OpenAI 2.32.0. The
upstream installer installs these after mini-swe-agent. Its floating cost-map
refresh is removed; the packaged LiteLLM map is used, and local API costs are
reported as unavailable. The official mini prompt is unchanged.

To update deliberately: export the selected Pier `uv.lock` with `uv export
--frozen --no-dev --no-emit-project`, merge the pinned mini dependency closure,
and run `uv pip compile --python-version 3.12 --python-platform linux
--generate-hashes`. Re-run the dedicated CI runtime job and source validation.
No runtime command follows a moving Git branch. OS package repositories and
container tags can still change; actual image IDs/digests are retained in raw
artifacts and task image provenance is checked before comparing local runs.

# Terminal-Bench 4.0 runtime lock

`terminal-bench-4-harbor.txt` targets CPython 3.12 on Linux. It is the hashed
dependency closure of `harbor==0.24.0` from PyPI (wheel sha256
`23b7ba616a3aae4eff561ced5e2c51c7186f2981e53a770dea9c689d3969877c`), which is
the release of `harbor-framework/harbor@b53b8134e1241686dca7759af188f987ecc48e8b`.
Harbor 0.22/0.23 abort the whole job when a GPU task meets a Docker daemon
without GPUs; 0.24 fails only that trial, which the GPU policy requires.

The dataset itself is pinned by digest in
`src/kairyu_bench/data/terminal-bench-4.json`: the 66 per-task digests of
`terminal-bench/terminal-bench@sha256:39d9f44b…7732` (the leaderboard's
`DATASET_REF`), the digest-pinned task images, and the agent versions from the
official leaderboard run configs at
`harbor-framework/terminal-bench@209679e34327a78ce8e9bf5300b2f317992863c9`.

To update deliberately: change the pin, run `uv pip compile` with the command
in the lock header, regenerate the dataset lock from the registry, and re-run
the `terminal-bench-4-runtime` CI job.
