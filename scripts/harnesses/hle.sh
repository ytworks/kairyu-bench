#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter hle
require_command git

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
dataset_revision=$(context_get dataset.revision)
dataset_id=$(context_get dataset.id)
limit=$(context_get limit)
source_path=$(checkout_source simple-evals "$source_repository" "$source_revision")
environment=$(ensure_venv \
    hle \
    "$source_revision" \
    -r "$source_path/requirements/base.txt" \
    "datasets==5.0.0")
raw=$(raw_directory hle)
hf_home="${KAIRYU_BENCH_CACHE_DIR:-/work/cache}/hf/hle-$dataset_revision"
mkdir -p "$hf_home"
export HF_HOME="$hf_home"
export HF_HUB_DISABLE_TELEMETRY=1

python -m kairyu_bench.harness_data --split test --cache-only
python -m kairyu_bench.harness_config hle "$raw/models.json"

set -- \
    --model=kairyu \
    --judge_model=kairyu \
    --dataset="$dataset_id" \
    --output_file="$raw/hle.jsonl" \
    --models_config="$raw/models.json" \
    --max_concurrent=1 \
    --redo=True
if [ "$limit" != "null" ]; then
    set -- "$@" --max_samples="$limit"
fi

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
(cd "$source_path" && "$environment/bin/python" hle/hle_eval.py "$@")
normalize_official "$raw/hle.jsonl"
