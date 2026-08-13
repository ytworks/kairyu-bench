#!/bin/sh
set -eu

benchmark=$1
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter "$benchmark"
require_command docker
require_command git

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
generator_repository=$(context_get generator.repository)
generator_revision=$(context_get generator.revision)
dataset_id=$(context_get dataset.id)
dataset_revision=$(context_get dataset.revision)
limit=$(context_get limit)
run_id=$(context_get run_id)

source_path=$(checkout_source swebench "$source_repository" "$source_revision")
generator_path=$(checkout_source mini-swe-agent "$generator_repository" "$generator_revision")
environment=$(ensure_venv \
    swebench \
    "$source_revision-$generator_revision" \
    "datasets==5.0.0" \
    "$source_path" \
    "$generator_path")

raw=$(raw_directory "$benchmark")
hf_home="${KAIRYU_BENCH_CACHE_DIR:-/work/cache}/hf/$benchmark-$dataset_revision"
mkdir -p "$hf_home" "$raw/generation" "$raw/reports"
export HF_HOME="$hf_home"
export HF_HUB_DISABLE_TELEMETRY=1

python -m kairyu_bench.harness_data \
    --split test \
    --id-field instance_id \
    --output "$raw/selected.jsonl" \
    --ids "$raw/instance-ids.txt"

set -- \
    --subset "$dataset_id" \
    --split test \
    --output "$raw/generation" \
    --workers 1 \
    --model "openai/$KAIRYU_MODEL"
if [ "$limit" != "null" ]; then
    set -- "$@" --slice "0:$limit"
fi

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
"$environment/bin/mini-extra" swebench "$@"

set --
while IFS= read -r instance_id; do
    [ -n "$instance_id" ] && set -- "$@" "$instance_id"
done <"$raw/instance-ids.txt"

"$environment/bin/python" -m swebench.harness.run_evaluation \
    --dataset_name "$raw/selected.jsonl" \
    --split test \
    --predictions_path "$raw/generation/preds.json" \
    --max_workers 1 \
    --run_id "$run_id-$benchmark" \
    --report_dir "$raw/reports" \
    --instance_ids "$@"

report=
for candidate in "$raw/reports"/*.json; do
    if [ -f "$candidate" ]; then
        if [ -n "$report" ]; then
            echo "kairyu-bench: SWE-bench emitted multiple reports" >&2
            exit 2
        fi
        report=$candidate
    fi
done
[ -n "$report" ] || {
    echo "kairyu-bench: SWE-bench did not emit its report" >&2
    exit 2
}
normalize_official "$report"
