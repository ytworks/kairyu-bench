#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter terminal-bench
require_command docker
require_command git

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
dataset_id=$(context_get dataset.id)
limit=$(context_get limit)
run_id=$(context_get run_id)
source_path=$(checkout_source harbor "$source_repository" "$source_revision")
environment=$(ensure_venv harbor "$source_revision" "$source_path")
raw=$(raw_directory terminal-bench)
mkdir -p "$raw/jobs"

set -- \
    run \
    --dataset "$dataset_id" \
    --agent terminus-2 \
    --model "openai/$KAIRYU_MODEL" \
    --agent-kwarg "api_base=$KAIRYU_ENDPOINT" \
    --jobs-dir "$raw/jobs" \
    --job-name "$run_id-terminal-bench" \
    --n-concurrent 1 \
    --n-attempts 1
if [ "$limit" != "null" ]; then
    set -- "$@" --n-tasks "$limit"
fi

"$environment/bin/harbor" "$@"
normalize_official "$raw/jobs"
