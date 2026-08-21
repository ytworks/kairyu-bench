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

source_path=$(checkout_source "$benchmark" "$source_repository" "$source_revision")
generator_path=$(checkout_source mini-swe-agent "$generator_repository" "$generator_revision")
if [ "$benchmark" = "swe-bench-pro" ]; then
    environment=$(ensure_venv \
        swebench-pro \
        "$source_revision-$generator_revision" \
        "datasets==5.0.0" \
        "docker==7.1.0" \
        "pandas==2.3.2" \
        "$generator_path")
else
    environment=$(ensure_venv \
        swebench \
        "$source_revision-$generator_revision" \
        "datasets==5.0.0" \
        "$source_path" \
        "$generator_path")
fi

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

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MSWEA_COST_TRACKING=ignore_errors

if [ "$benchmark" = "swe-bench-pro" ]; then
    total=$(wc -l <"$raw/instance-ids.txt" | tr -d ' ')
    items="$raw/items"
    outcomes="$raw/evaluation/eval_results.json"
    workers=${KAIRYU_BENCH_SWEBENCH_PRO_WORKERS:-1}
    case "$workers" in
        ''|*[!0-9]*)
            echo "kairyu-bench: KAIRYU_BENCH_SWEBENCH_PRO_WORKERS must be an integer from 1 to 16" >&2
            exit 2
            ;;
    esac
    if [ "$workers" -lt 1 ] || [ "$workers" -gt 16 ]; then
        echo "kairyu-bench: KAIRYU_BENCH_SWEBENCH_PRO_WORKERS must be an integer from 1 to 16" >&2
        exit 2
    fi
    mkdir -p "$items" "$raw/evaluation"

    cleanup_task_image() {
        image=$1
        [ -n "$image" ] || return 0
        [ "${KAIRYU_BENCH_CLEAN_TASK_IMAGES:-0}" = 1 ] || return 0
        container_ids=$(docker ps -aq --filter "ancestor=$image" 2>/dev/null) || container_ids=
        for container_id in $container_ids; do
            docker rm -f "$container_id" >/dev/null 2>&1 || true
        done
        docker image rm -f "$image" >/dev/null 2>&1 || true
        if docker image inspect "$image" >/dev/null 2>&1; then
            echo "SWE-bench Pro storage warning: could not remove $image" >&2
        else
            echo "SWE-bench Pro storage: removed completed task image $image"
        fi
    }

    cleanup_all_task_objects() {
        [ "${KAIRYU_BENCH_CLEAN_TASK_IMAGES:-0}" = 1 ] || return 0
        docker ps -a --format '{{.ID}} {{.Image}}' 2>/dev/null |
            while IFS=' ' read -r container_id image; do
                case "$image" in
                    jefzda/sweap-images:*)
                        docker rm -f "$container_id" >/dev/null 2>&1 || true
                        ;;
                esac
            done
        docker image ls --format '{{.Repository}}:{{.Tag}}' 2>/dev/null |
            while IFS= read -r image; do
                case "$image" in
                    jefzda/sweap-images:*)
                        docker image rm -f "$image" >/dev/null 2>&1 || true
                        ;;
                esac
            done
    }

    pids=
    stop_pro_workers() {
        trap - EXIT HUP INT TERM
        for worker_pid in $pids; do
            kill "$worker_pid" >/dev/null 2>&1 || true
        done
        for worker_pid in $pids; do
            wait "$worker_pid" >/dev/null 2>&1 || true
        done
        cleanup_all_task_objects
    }
    trap 'stop_pro_workers' EXIT
    trap 'stop_pro_workers; exit 130' HUP INT TERM

    run_pro_item() (
        index=$1
        instance_id=$2
        key=$(printf '%04d' "$index")
        item="$items/$key"
        item_dataset="$item/dataset"
        item_generation="$item/generation"
        item_predictions="$item/predictions.json"
        item_evaluation="$item/evaluation"
        item_ids="$item/instance-ids.txt"
        current_image=
        trap 'cleanup_task_image "$current_image"' EXIT
        trap 'exit 130' HUP INT TERM

        mkdir -p "$item_dataset" "$item_generation" "$item_evaluation"
        printf '%s\n' "$instance_id" >"$item_ids"
        python -m kairyu_bench.swebench_pro dataset \
            "$raw/selected.jsonl" \
            "$item_dataset/test.jsonl" \
            --instance-id "$instance_id"
        current_image=$(python -m kairyu_bench.swebench_pro image \
            "$raw/selected.jsonl" "$instance_id")

        echo "SWE-bench Pro problem $index/$total: generating $instance_id"
        "$environment/bin/mini-extra" swebench \
            --subset "$item_dataset" \
            --split test \
            --output "$item_generation" \
            --workers 1 \
            --model "openai/$KAIRYU_MODEL" \
            --config swebench.yaml \
            --config model.model_kwargs.max_tokens=8192 \
            --config environment.cwd=/app \
            --config environment.pull_timeout=1800 \
            --config 'environment.run_args=["--rm","--entrypoint",""]' \
            --config 'run.env_startup_command=test -e /testbed || ln -s /app /testbed'

        python -m kairyu_bench.swebench_pro predictions \
            "$item_generation/preds.json" \
            "$item_predictions" \
            --expected-ids "$item_ids" \
            --select-id "$instance_id"
        (
            cd "$source_path"
            "$environment/bin/python" swe_bench_pro_eval.py \
                --raw_sample_path "$item_dataset/test.jsonl" \
                --patch_path "$item_predictions" \
                --output_dir "$item_evaluation" \
                --scripts_dir "$source_path/run_scripts" \
                --dockerhub_username jefzda \
                --num_workers 1 \
                --use_local_docker
        )
        resolved=$(python -m kairyu_bench.swebench_pro record-outcome \
            "$item_evaluation/eval_results.json" \
            "$item/outcome.json" \
            --expected-id "$instance_id")
        cleanup_task_image "$current_image"
        current_image=
        trap - EXIT HUP INT TERM
        echo "SWE-bench Pro completed $index/$total: resolved=$resolved instance=$instance_id"
    )

    wait_pro_batch() {
        batch_failed=0
        for worker_pid in $pids; do
            wait "$worker_pid" || batch_failed=1
        done
        pids=
        batch=0
        [ "$batch_failed" -eq 0 ]
    }

    index=0
    batch=0
    while IFS= read -r instance_id; do
        [ -n "$instance_id" ] || continue
        index=$((index + 1))
        run_pro_item "$index" "$instance_id" &
        pids="$pids $!"
        batch=$((batch + 1))
        if [ "$batch" -eq "$workers" ]; then
            if ! wait_pro_batch; then
                echo "kairyu-bench: one or more SWE-bench Pro workers failed" >&2
                exit 2
            fi
        fi
    done <"$raw/instance-ids.txt"
    if [ "$batch" -gt 0 ]; then
        if ! wait_pro_batch; then
            echo "kairyu-bench: one or more SWE-bench Pro workers failed" >&2
            exit 2
        fi
    fi

    python -m kairyu_bench.swebench_pro aggregate-items \
        "$items" \
        "$raw/predictions.json" \
        "$outcomes" \
        "$raw/instance-ids.txt"
    python -m kairyu_bench.swebench_pro verify \
        "$raw/predictions.json" \
        "$outcomes" \
        "$raw/instance-ids.txt"
    trap - EXIT HUP INT TERM
    normalize_official "$outcomes"
    exit 0
fi

set -- \
    --subset "$dataset_id" \
    --split test \
    --output "$raw/generation" \
    --workers 1 \
    --model "openai/$KAIRYU_MODEL" \
    --config swebench.yaml \
    --config model.model_kwargs.max_tokens=8192
if [ "$limit" != "null" ]; then
    set -- "$@" --slice "0:$limit"
fi
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
