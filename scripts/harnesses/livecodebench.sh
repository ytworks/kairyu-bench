#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter livecodebench
require_command git

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
dataset_revision=$(context_get dataset.revision)
source_path=$(checkout_source livecodebench "$source_repository" "$source_revision")
environment=$(ensure_venv \
    livecodebench \
    "$source_revision" \
    "datasets==5.0.0" \
    "numpy" \
    "pebble" \
    "tqdm")
raw=$(raw_directory livecodebench)
hf_home="${KAIRYU_BENCH_CACHE_DIR:-/work/cache}/hf/livecodebench-$dataset_revision"
mkdir -p "$hf_home"
export HF_HOME="$hf_home"
export HF_HUB_DISABLE_TELEMETRY=1
export KAIRYU_BENCH_RAW_DIR="$raw"
export PYTHONPATH="$ROOT/src:$source_path"

"$environment/bin/python" "$ROOT/scripts/shims/livecodebench.py"
normalize_official "$raw/summary.json"
