#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter terminal-bench
harbor_agent=${KAIRYU_HARBOR_AGENT:-terminus-2}
workers=${KAIRYU_BENCH_TERMINAL_BENCH_WORKERS:-4}

case "$workers" in
    ''|*[!0-9]*)
        echo "kairyu-bench: KAIRYU_BENCH_TERMINAL_BENCH_WORKERS must be an integer from 1 to 16" >&2
        exit 2
        ;;
esac
if [ "$workers" -lt 1 ] || [ "$workers" -gt 16 ]; then
    echo "kairyu-bench: KAIRYU_BENCH_TERMINAL_BENCH_WORKERS must be an integer from 1 to 16" >&2
    exit 2
fi
setup_timeout_multiplier=${KAIRYU_BENCH_TERMINAL_BENCH_SETUP_TIMEOUT_MULTIPLIER:-4}

case "$setup_timeout_multiplier" in
    ''|*[!0-9]*)
        echo "kairyu-bench: KAIRYU_BENCH_TERMINAL_BENCH_SETUP_TIMEOUT_MULTIPLIER must be an integer from 1 to 16" >&2
        exit 2
        ;;
esac
if [ "$setup_timeout_multiplier" -lt 1 ] || [ "$setup_timeout_multiplier" -gt 16 ]; then
    echo "kairyu-bench: KAIRYU_BENCH_TERMINAL_BENCH_SETUP_TIMEOUT_MULTIPLIER must be an integer from 1 to 16" >&2
    exit 2
fi

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
dataset_id=$(context_get dataset.id)
limit=$(context_get limit)
run_id=$(context_get run_id)
raw=$(raw_directory terminal-bench)
mkdir -p "$raw/jobs"

set -- \
    run \
    --dataset "$dataset_id" \
    --jobs-dir "$raw/jobs" \
    --job-name "$run_id-terminal-bench" \
    --extra-docker-compose "$ROOT/scripts/harnesses/harbor-host-gateway.yaml" \
    --n-concurrent "$workers" \
    --n-attempts 1
case "$harbor_agent" in
    terminus-2)
        set -- "$@" \
            --agent terminus-2 \
            --model "openai/$KAIRYU_MODEL" \
            --agent-kwarg "api_base=$KAIRYU_ENDPOINT"
        ;;
    claude-code)
        set -- "$@" \
            --agent claude-code \
            --model "$KAIRYU_MODEL" \
            --agent-setup-timeout-multiplier "$setup_timeout_multiplier" \
            --agent-env "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1" \
            --agent-env "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1" \
            --agent-env "CLAUDE_CODE_ATTRIBUTION_HEADER=0"
        ;;
    codex)
        case "$KAIRYU_MODEL" in
            */*)
                unsupported "Harbor's Codex agent cannot preserve model IDs containing '/': $KAIRYU_MODEL"
                ;;
        esac
        set -- "$@" \
            --agent codex \
            --model "$KAIRYU_MODEL" \
            --agent-setup-timeout-multiplier "$setup_timeout_multiplier"
        ;;
    *)
        echo "kairyu-bench: unsupported Harbor agent: $harbor_agent" >&2
        exit 2
        ;;
esac
if [ "$limit" != "null" ]; then
    set -- "$@" --n-tasks "$limit"
fi
# KAIRYU_BENCH_TERMINAL_BENCH_TASKS (comma-separated task names) runs exactly
# those tasks; Harbor matches them as terminal-bench/<task>.
requested_tasks=0
if [ -n "${KAIRYU_BENCH_TERMINAL_BENCH_TASKS:-}" ]; then
    old_ifs=$IFS
    IFS=,
    for task in $KAIRYU_BENCH_TERMINAL_BENCH_TASKS; do
        set -- "$@" --include-task-name "terminal-bench/$task"
        requested_tasks=$((requested_tasks + 1))
    done
    IFS=$old_ifs
fi

require_command docker
require_command git
source_path=$(checkout_source harbor "$source_repository" "$source_revision")
environment=$(ensure_venv harbor "$source_revision" "$source_path")

"$environment/bin/harbor" "$@"
if [ "$requested_tasks" -gt 0 ]; then
    trials=$(find "$raw/jobs" -mindepth 3 -maxdepth 3 -name result.json | wc -l | tr -d ' ')
    if [ "$trials" -ne "$requested_tasks" ]; then
        echo "kairyu-bench: requested $requested_tasks Terminal-Bench tasks, Harbor ran $trials" >&2
        exit 2
    fi
fi
normalize_official "$raw/jobs"
