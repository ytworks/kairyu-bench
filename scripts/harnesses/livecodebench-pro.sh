#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_adapter livecodebench-pro
require_command docker
require_command git
if [ -z "${HF_TOKEN:-}" ]; then
    unsupported "LiveCodeBench Pro requires HF_TOKEN and accepted gated dataset terms"
fi

source_repository=$(context_get source.repository)
source_revision=$(context_get source.revision)
verifier_repository=$(context_get secondary_sources.1.id)
verifier_revision=$(context_get secondary_sources.1.revision)
dataset_revision=$(context_get dataset.revision)
source_path=$(checkout_source livecodebench-pro "$source_repository" "$source_revision")
verifier_path=$(checkout_source lightcpverifier "$verifier_repository" "$verifier_revision")
verifier_image="lightcpverifier:${verifier_revision}-kairyu"
docker build \
    --file "$ROOT/scripts/harnesses/livecodebench-pro.Dockerfile" \
    --tag "$verifier_image" \
    "$verifier_path"
environment=$(ensure_venv \
    livecodebench-pro \
    "$source_revision" \
    -r "$source_path/requirements.txt")
raw=$(raw_directory livecodebench-pro)
work="$raw/work"
hf_home="${KAIRYU_BENCH_CACHE_DIR:-/work/cache}/hf/livecodebench-pro-$dataset_revision"
mkdir -p "$work" "$hf_home"
ln -s "$verifier_path" "$work/LightCPVerifier"
export HF_HOME="$hf_home"
export HF_HUB_DISABLE_TELEMETRY=1
export KAIRYU_BENCH_RAW_DIR="$raw"
export KAIRYU_LIGHTCPVERIFIER_IMAGE="$verifier_image"
export KAIRYU_LIGHTCPVERIFIER_HOST="${KAIRYU_LIGHTCPVERIFIER_HOST:-container}"
export PYTHONPATH="$ROOT/src:$source_path"

(cd "$work" && "$environment/bin/python" "$ROOT/scripts/shims/livecodebench_pro.py")
normalize_official "$raw/benchmark_result.json"
