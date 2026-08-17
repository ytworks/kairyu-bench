#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter terminal-bench
harbor_agent=${KAIRYU_HARBOR_AGENT:-terminus-2}

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
    --n-concurrent 1 \
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
        set -- "$@" --agent codex --model "$KAIRYU_MODEL"
        ;;
    *)
        echo "kairyu-bench: unsupported Harbor agent: $harbor_agent" >&2
        exit 2
        ;;
esac
if [ "$limit" != "null" ]; then
    set -- "$@" --n-tasks "$limit"
fi

require_command docker
require_command git
source_path=$(checkout_source harbor "$source_repository" "$source_revision")
environment=$(ensure_venv harbor "$source_revision" "$source_path")

"$environment/bin/harbor" "$@"
normalize_official "$raw/jobs"
