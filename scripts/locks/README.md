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
