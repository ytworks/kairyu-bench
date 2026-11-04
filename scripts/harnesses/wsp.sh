#!/bin/sh
set -eu

# Weight-space programming tiers: pinned official sources and the scorer
# environment, then `python -m kairyu_bench.wsp.cli` inside it.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
. "$ROOT/scripts/lib/official.sh"
require_command git
require_command curl

cache_root=${KAIRYU_BENCH_CACHE_DIR:-/work/cache}

# google-research is too large to fetch; its IFEval checker is four files.
IFEVAL_REVISION=e6890f85757dd84e27ca6df2dd30651dafad28e0
fetch_ifeval() {
    destination="$cache_root/sources/ifeval-$IFEVAL_REVISION"
    package="$destination/instruction_following_eval"
    if [ -f "$destination/.kairyu-bench-revision" ]; then
        printf '%s\n' "$destination"
        return 0
    fi
    temporary=$(mktemp -d "$cache_root/tmp/ifeval.XXXXXX")
    mkdir -p "$temporary/instruction_following_eval"
    while read -r digest name; do
        curl -fsSL -o "$temporary/instruction_following_eval/$name" \
            "https://raw.githubusercontent.com/google-research/google-research/$IFEVAL_REVISION/instruction_following_eval/$name"
        actual=$(sha256sum "$temporary/instruction_following_eval/$name" | cut -d' ' -f1)
        if [ "$actual" != "$digest" ]; then
            echo "kairyu-bench: $name hash $actual, expected $digest" >&2
            rm -rf "$temporary"
            exit 2
        fi
    done <<HASHES
35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c evaluation_lib.py
60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf instructions.py
ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc instructions_registry.py
a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b instructions_util.py
HASHES
    printf '%s\n' "$IFEVAL_REVISION" >"$temporary/.kairyu-bench-revision"
    mkdir -p "$(dirname "$destination")"
    mv "$temporary" "$destination" 2>/dev/null || rm -rf "$temporary"
    printf '%s\n' "$destination"
}

mkdir -p "$cache_root/tmp"
ifeval=$(fetch_ifeval)
ifbench=$(checkout_source ifbench https://github.com/allenai/IFBench cd189a9c2889526d8af901d2103fd3e1ffcea6df)
xstest=$(checkout_source xstest https://github.com/paul-rottger/xstest d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d)
bbeh=$(checkout_source bbeh https://github.com/google-deepmind/bbeh 80d12ca916b7158f22293fcf3144f4d3d854d4be)
musr=$(checkout_source musr https://github.com/Zayne-Sprague/MuSR b1f4d4168a9cfc6760e8b74d728e4516023dfaa5)
livecodebench=$(checkout_source livecodebench https://github.com/LiveCodeBench/LiveCodeBench 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24)
scicode=$(checkout_source scicode https://github.com/scicode-bench/SciCode e3158ea011d4235245a547460d3688d7ccbf9900)
ruler=$(checkout_source ruler https://github.com/NVIDIA/RULER db43f92ff634bbce1c33d1fb15858aec4bba509a)

# Three environments: the runner and most scorers, EvalPlus (its tree-sitter
# conflicts with BFCL's), and the interpreter that runs SciCode step tests.
# Each lock lists its whole closure with hashes.
lock_environment() {
    lock="$ROOT/scripts/locks/$1.txt"
    ensure_venv "$1" "$(sha256sum "$lock" | cut -c1-16)" --require-hashes --no-deps -r "$lock"
}
environment=$(lock_environment wsp)
evalplus_environment=$(lock_environment wsp-evalplus)
scicode_environment=$(lock_environment wsp-scicode)
lock_revision=$(sha256sum "$ROOT/scripts/locks/wsp.txt" | cut -c1-16)
data_root="$cache_root/wsp"
mkdir -p "$data_root"
export XDG_CACHE_HOME="$cache_root/xdg"

# SciCode's test targets (1,049,345,865 bytes). The official copy is on Google
# Drive; when its download quota is exhausted, a Hugging Face mirror is used.
# Either must match the hash (four independent mirrors agree on it).
SCICODE_H5_SHA256=48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890
scicode_h5="$data_root/scicode-test_data.h5"
if [ ! -f "$scicode_h5.verified" ]; then
    temporary="$scicode_h5.download"
    page="$data_root/scicode-drive.html"
    drive_id=17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW
    curl -fsSL -o "$page" "https://drive.google.com/uc?export=download&id=$drive_id" || true
    uuid=$(sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' "$page" 2>/dev/null | head -1)
    curl -fsSL -o "$temporary" \
        "https://drive.usercontent.google.com/download?id=$drive_id&export=download&confirm=t&uuid=$uuid" || true
    if [ "$(sha256sum "$temporary" 2>/dev/null | cut -d' ' -f1)" != "$SCICODE_H5_SHA256" ]; then
        curl -fsSL -o "$temporary" \
            "https://huggingface.co/datasets/Srimadh/Scicode-test-data-h5/resolve/72c247d3a8410921b2e848e046d71ed63d9a0ddb/test_data.h5"
    fi
    if [ "$(sha256sum "$temporary" | cut -d' ' -f1)" != "$SCICODE_H5_SHA256" ]; then
        echo "kairyu-bench: SciCode test_data.h5 does not match $SCICODE_H5_SHA256" >&2
        exit 2
    fi
    mv "$temporary" "$scicode_h5"
    rm -f "$page"
    touch "$scicode_h5.verified"
fi

# The pinned HumanEval+/MBPP+ prompts, from EvalPlus's own release data.
evalplus_tasks="$data_root/evalplus-0.3.1-tasks.json"
if [ ! -f "$evalplus_tasks" ]; then
    PYTHONPATH="$ROOT/src" "$evalplus_environment/bin/python" -m kairyu_bench.wsp.scorers.evalplus_score \
        --dump "$evalplus_tasks.tmp"
    mv "$evalplus_tasks.tmp" "$evalplus_tasks"
fi

# NLTK refuses data in a directory that it or an ancestor lets group or
# others write, which a shared cache may; the data is kept as an archive in
# the cache and unpacked into a private directory for each run.
nltk_archive="$cache_root/nltk/wsp-$lock_revision.tar"
nltk_data="${HOME:-/root}/.kairyu-bench-nltk-$lock_revision"
if [ ! -d "$nltk_data" ]; then
    (umask 077 && mkdir -p "$nltk_data")
    if [ -f "$nltk_archive" ]; then
        tar -xf "$nltk_archive" -C "$nltk_data"
    else
        "$environment/bin/python" -c 'import sys, nltk
for name in ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"):
    assert nltk.download(name, download_dir=sys.argv[1], quiet=True), name' "$nltk_data"
        mkdir -p "$(dirname "$nltk_archive")"
        tar -cf "$nltk_archive.tmp" -C "$nltk_data" . && mv "$nltk_archive.tmp" "$nltk_archive"
    fi
    chmod -R go-w "$nltk_data"
fi

export NLTK_DATA="$nltk_data"
export HF_HOME="$cache_root/hf/wsp"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="$ROOT/src"
export KAIRYU_BENCH_WSP_ENV=1
export KAIRYU_BENCH_WSP_DATA="$data_root"
json_map() {
    "$environment/bin/python" -c 'import json, sys
pairs = sys.argv[1:]
print(json.dumps(dict(zip(pairs[0::2], pairs[1::2]))))' "$@"
}
KAIRYU_BENCH_WSP_SOURCES=$(json_map ifeval "$ifeval" ifbench "$ifbench" xstest "$xstest" bbeh "$bbeh" \
    musr "$musr" livecodebench "$livecodebench" scicode "$scicode" ruler "$ruler")
KAIRYU_BENCH_WSP_PYTHONS=$(json_map wsp "$environment/bin/python" evalplus "$evalplus_environment/bin/python" \
    scicode "$scicode_environment/bin/python")
KAIRYU_BENCH_WSP_FILES=$(json_map scicode-h5 "$scicode_h5" evalplus-tasks "$evalplus_tasks")
export KAIRYU_BENCH_WSP_SOURCES KAIRYU_BENCH_WSP_PYTHONS KAIRYU_BENCH_WSP_FILES
exec "$environment/bin/python" -m kairyu_bench.wsp.cli "$@"
