#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter terminal-bench-4
require_command docker
lock="$ROOT/scripts/locks/terminal-bench-4-harbor.txt"
source_revision=$(context_get source.revision)
lock_hash=$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$lock")
environment=$(ensure_venv terminal-bench-4 "$source_revision-$lock_hash" -r "$lock")
raw=$(raw_directory terminal-bench-4)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HARBOR_TELEMETRY=0

harbor_version=$("$environment/bin/harbor" --version)
set +e
"$environment/bin/python" -m kairyu_bench.terminal_bench4_cli prepare "$raw" --harbor-version "$harbor_version"
code=$?
set -e
if [ "$code" -eq 10 ]; then
    unsupported "$(cat "$raw/unsupported.txt")"
fi
if [ "$code" -ne 0 ]; then
    normalize_official "$raw"
fi

# Official CLI path; the plugin records the plan, retry evidence, progress and
# cleanup. Harbor never needs the Hugging Face token or the raw Kairyu key.
env -u HF_TOKEN -u KAIRYU_API_KEY "$environment/bin/harbor" run \
    --config "$raw/job-config.json" \
    --plugin kairyu_bench.terminal_bench4_plugin:KairyuTB4Plugin \
    --plugin-kwarg "raw_dir=$raw" \
    --yes \
    --quiet &
harbor_pid=$!
trap 'kill -TERM "$harbor_pid" 2>/dev/null || true' TERM INT
set +e
wait "$harbor_pid"
code=$?
# A trapped signal interrupts wait; keep waiting so Harbor's finalizers finish.
while kill -0 "$harbor_pid" 2>/dev/null; do
    wait "$harbor_pid"
    code=$?
done
set -e
trap - TERM INT
if [ "$code" -ne 0 ]; then
    echo "Terminal-Bench 4.0 harness exited $code; preserving partial official results" >&2
    printf '{"type": "HarborExit", "code": %s}\n' "$code" >"$raw/runtime-error.json"
fi
set +e
"$environment/bin/python" -m kairyu_bench.terminal_bench4_cli finalize "$raw"
set -e
normalize_official "$raw"
