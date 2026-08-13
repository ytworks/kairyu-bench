#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter scicode
require_command git

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
dataset_revision=$(context_get dataset.revision)
limit=$(context_get limit)
source_path=$(checkout_source scicode "$source_repository" "$source_revision")
environment=$(ensure_venv scicode "$source_revision" "datasets==5.0.0" "$source_path")
raw=$(raw_directory scicode)
test_data="${KAIRYU_BENCH_CACHE_DIR:-/work/cache}/scicode/test_data.h5"
if [ ! -f "$test_data" ]; then
    unsupported "SciCode requires the official numeric test file at .cache/scicode/test_data.h5"
fi

hf_home="${KAIRYU_BENCH_CACHE_DIR:-/work/cache}/hf/scicode-$dataset_revision"
mkdir -p "$hf_home" "$raw/logs" "$raw/output"
export HF_HOME="$hf_home"
export HF_HUB_DISABLE_TELEMETRY=1
python -m kairyu_bench.harness_data --split test --cache-only

set -- \
    eval scicode.py \
    --model "openai/$KAIRYU_MODEL" \
    --temperature 0 \
    --max-connections 1 \
    --log-dir "$raw/logs" \
    --log-format json \
    -T split=test \
    -T "output_dir=$raw/output" \
    -T "h5py_file=$test_data" \
    -T with_background=False \
    -T mode=normal
if [ "$limit" != "null" ]; then
    set -- "$@" --limit "$limit"
fi

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
(cd "$source_path/eval/inspect_ai" && "$environment/bin/inspect" "$@")

inspect_log=
for candidate in "$raw/logs"/*.json; do
    if [ -f "$candidate" ]; then
        if [ -n "$inspect_log" ]; then
            echo "kairyu-bench: SciCode emitted multiple Inspect logs" >&2
            exit 2
        fi
        inspect_log=$candidate
    fi
done
[ -n "$inspect_log" ] || {
    echo "kairyu-bench: SciCode did not emit an Inspect log" >&2
    exit 2
}
python -m kairyu_bench.inspect_summary "$inspect_log" "$raw/inspect-summary.json"
normalize_official "$raw/inspect-summary.json"
