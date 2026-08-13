#!/bin/sh

# Shared POSIX-shell support for pinned official benchmark harnesses.

context_get() {
    python -m kairyu_bench.adapter_context "$1"
}

require_adapter() {
    expected=$1
    actual=$(context_get benchmark)
    if [ "$actual" != "$expected" ]; then
        echo "kairyu-bench: adapter $expected received context for $actual" >&2
        exit 2
    fi
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        unsupported "$1 is required by the official harness"
    fi
}

unsupported() {
    reason=$1
    set +e
    python -m kairyu_bench.official unsupported "$reason"
    code=$?
    set -e
    exit "$code"
}

normalize_official() {
    raw_path=$1
    set +e
    python -m kairyu_bench.official normalize "$raw_path"
    code=$?
    set -e
    exit "$code"
}

checkout_source() {
    source_name=$1
    source_url=$2
    source_revision=$3
    cache_root=${KAIRYU_BENCH_CACHE_DIR:-/work/cache}
    sources_root="$cache_root/sources"
    destination="$sources_root/$source_name-$source_revision"
    mkdir -p "$sources_root" "$cache_root/tmp"

    if [ -f "$destination/.kairyu-bench-revision" ] &&
        [ "$(sed -n '1p' "$destination/.kairyu-bench-revision")" = "$source_revision" ]; then
        printf '%s\n' "$destination"
        return 0
    fi

    temporary=$(mktemp -d "$cache_root/tmp/source.XXXXXX")
    cleanup_source() {
        rm -rf "$temporary"
    }
    trap cleanup_source EXIT HUP INT TERM
    git -C "$temporary" init -q
    git -C "$temporary" remote add origin "$source_url"
    git -C "$temporary" fetch -q --depth=1 origin "$source_revision"
    git -C "$temporary" checkout -q --detach FETCH_HEAD
    actual_revision=$(git -C "$temporary" rev-parse HEAD)
    if [ "$actual_revision" != "$source_revision" ]; then
        echo "kairyu-bench: fetched $actual_revision, expected $source_revision" >&2
        exit 2
    fi
    printf '%s\n' "$source_revision" >"$temporary/.kairyu-bench-revision"
    if ! mv "$temporary" "$destination" 2>/dev/null; then
        if [ ! -f "$destination/.kairyu-bench-revision" ]; then
            echo "kairyu-bench: could not publish cached source $source_name" >&2
            exit 2
        fi
    fi
    trap - EXIT HUP INT TERM
    printf '%s\n' "$destination"
}

ensure_venv() {
    environment_name=$1
    environment_revision=$2
    shift 2
    cache_root=${KAIRYU_BENCH_CACHE_DIR:-/work/cache}
    environments_root="$cache_root/venvs"
    destination="$environments_root/$environment_name-$environment_revision"
    mkdir -p "$environments_root" "$cache_root/tmp"

    if [ -f "$destination/.kairyu-bench-ready" ] &&
        [ "$(sed -n '1p' "$destination/.kairyu-bench-ready")" = "$environment_revision" ]; then
        printf '%s\n' "$destination"
        return 0
    fi

    temporary=$(mktemp -d "$cache_root/tmp/venv.XXXXXX")
    cleanup_venv() {
        rm -rf "$temporary"
    }
    trap cleanup_venv EXIT HUP INT TERM
    python -m venv "$temporary/environment"
    "$temporary/environment/bin/pip" install --disable-pip-version-check "$@" >&2
    printf '%s\n' "$environment_revision" >"$temporary/environment/.kairyu-bench-ready"
    if ! mv "$temporary/environment" "$destination" 2>/dev/null; then
        if [ ! -f "$destination/.kairyu-bench-ready" ]; then
            echo "kairyu-bench: could not publish environment $environment_name" >&2
            exit 2
        fi
    fi
    trap - EXIT HUP INT TERM
    printf '%s\n' "$destination"
}

raw_directory() {
    name=$1
    directory="$KAIRYU_BENCH_RUN_DIR/raw/$name"
    mkdir -p "$directory"
    printf '%s\n' "$directory"
}
