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

# KAIRYU_BENCH_HARBOR_CACHE_DIR keeps Harbor's task cache (~/.cache/harbor)
# in a persistent directory instead of the container's home.
if [ -n "${KAIRYU_BENCH_HARBOR_CACHE_DIR:-}" ]; then
    mkdir -p "$KAIRYU_BENCH_HARBOR_CACHE_DIR" "$HOME/.cache"
    if [ ! -e "$HOME/.cache/harbor" ]; then
        ln -s "$KAIRYU_BENCH_HARBOR_CACHE_DIR" "$HOME/.cache/harbor"
    fi
fi
# KAIRYU_BENCH_SETUP_ONLY=1 prepares Harbor and the dataset's task packages
# without running an agent (wsp prepare).
if [ "${KAIRYU_BENCH_SETUP_ONLY:-0}" = 1 ]; then
    source_path=$(checkout_source harbor "$source_repository" "$source_revision")
    environment=$(ensure_venv harbor "$source_revision" "$source_path")
    # --cache: the content-addressed layout `harbor run` reads
    # (~/.cache/harbor/tasks/packages), not the default export layout.
    "$environment/bin/harbor" download "$dataset_id" --cache
    exit 0
fi
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
# KAIRYU_BENCH_TERMINAL_BENCH_MAX_TURNS bounds terminus-2 by turns instead of
# the task's agent time limit, so the score does not depend on serving speed
# (wsp). The time limits then only detect hangs: 48x the agent limit (12 hours
# for a 900-second task, the SWE-bench Pro container lifetime) and the
# SWE-bench Pro harness's 1200 seconds per model call.
if [ -n "${KAIRYU_BENCH_TERMINAL_BENCH_MAX_TURNS:-}" ]; then
    case "$harbor_agent:$KAIRYU_BENCH_TERMINAL_BENCH_MAX_TURNS" in
        terminus-2:*[!0-9]*|terminus-2:0)
            echo "kairyu-bench: KAIRYU_BENCH_TERMINAL_BENCH_MAX_TURNS must be a positive integer" >&2
            exit 2
            ;;
        terminus-2:*)
            ;;
        *)
            unsupported "KAIRYU_BENCH_TERMINAL_BENCH_MAX_TURNS needs the terminus-2 agent"
            ;;
    esac
    set -- "$@" \
        --agent-kwarg "max_turns=$KAIRYU_BENCH_TERMINAL_BENCH_MAX_TURNS" \
        --agent-kwarg 'llm_call_kwargs={"timeout": 1200}' \
        --agent-timeout-multiplier 48
fi
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
