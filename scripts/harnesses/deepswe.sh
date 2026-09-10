#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter deepswe
require_command docker
require_command git
source_revision=$(context_get source.revision)
generator_revision=$(context_get generator.revision)
source_path=$(checkout_source pier "$(context_get source.repository)" "$source_revision")
task_path=$(checkout_source deepswe "$(context_get dataset.id)" "$(context_get dataset.revision)")
generator_path=$(checkout_source mini-swe-agent "$(context_get generator.repository)" "$generator_revision")
# The exported official uv.lock pins the host runtime; install local source separately
# because pip's hash-checking mode does not allow a local directory requirement.
lock_hash=$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$ROOT/scripts/locks/deepswe-pier.txt")
environment=$(ensure_venv deepswe "$source_revision-$generator_revision-$lock_hash" -r "$ROOT/scripts/locks/deepswe-pier.txt")
"$environment/bin/python" -m pip install --disable-pip-version-check --no-deps "$source_path" "$generator_path" >&2
raw=$(raw_directory deepswe)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=true
export MSWEA_COST_TRACKING=ignore_errors
set +e
"$environment/bin/python" -m kairyu_bench.deepswe_runtime "$task_path/tasks" "$raw"
code=$?
set -e
if [ "$code" -ne 0 ]; then
    echo "DeepSWE harness exited $code; preserving partial official results" >&2
fi
normalize_official "$raw"
