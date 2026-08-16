#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter tau-bench-banking
require_command git
require_command bwrap
require_command rg
require_command socat
require_command srt

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
limit=$(context_get limit)
run_id=$(context_get run_id)
embedding_model=$(context_get conditions.embedding_model_id)
[ -n "$embedding_model" ] || {
    echo "kairyu-bench: tau-bench-banking context has no embedding model" >&2
    exit 2
}
[ "${KAIRYU_EMBEDDING_MODEL:-}" = "$embedding_model" ] || {
    echo "kairyu-bench: tau-bench-banking embedding model environment differs from context" >&2
    exit 2
}
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
"$environment/bin/python" "$ROOT/scripts/shims/tau_bench_banking.py" "$@"
result="$source_path/data/simulations/$save_name/results.json"
[ -f "$result" ] || {
    echo "kairyu-bench: tau2 did not emit results.json" >&2
    exit 2
}
cp "$result" "$raw/results.json"
normalize_official "$raw/results.json"
