#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter tau-bench-banking
require_command git
require_command bwrap
require_command rg
require_command socat

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
limit=$(context_get limit)
run_id=$(context_get run_id)
source_path=$(checkout_source tau2-bench "$source_repository" "$source_revision")
environment=$(ensure_venv tau2 "$source_revision" "$source_path[knowledge]")
raw=$(raw_directory tau-bench-banking)
save_name="$run_id-tau-bench-banking"

set -- \
    run \
    --domain banking_knowledge \
    --retrieval-config alltools \
    --agent-llm "openai/$KAIRYU_MODEL" \
    --user-llm "openai/$KAIRYU_MODEL" \
    --agent-llm-args '{"temperature":0}' \
    --user-llm-args '{"temperature":0}' \
    --num-trials 4 \
    --max-concurrency 1 \
    --save-to "$save_name"
if [ "$limit" != "null" ]; then
    set -- "$@" --num-tasks "$limit"
fi

export TAU2_DATA_DIR="$source_path/data"
"$environment/bin/tau2" "$@"
result="$source_path/data/simulations/$save_name/results.json"
[ -f "$result" ] || {
    echo "kairyu-bench: tau2 did not emit results.json" >&2
    exit 2
}
cp "$result" "$raw/results.json"
normalize_official "$raw/results.json"
