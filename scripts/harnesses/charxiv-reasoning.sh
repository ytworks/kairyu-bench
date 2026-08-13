#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter charxiv-reasoning
require_command git

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
source_path=$(checkout_source charxiv "$source_repository" "$source_revision")
raw=$(raw_directory charxiv-reasoning)
export KAIRYU_BENCH_RAW_DIR="$raw"
export KAIRYU_BENCH_SOURCE_PATH="$source_path"
export PYTHONPATH="$ROOT/src:$source_path/src"

python "$ROOT/scripts/shims/charxiv_reasoning.py"
normalize_official "$raw/reasoning_summary.json"
