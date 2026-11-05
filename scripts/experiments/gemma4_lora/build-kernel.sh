#!/bin/sh
set -eu
# Explicit output path keeps compiled machine code outside the checkout.
if [ "$#" -ne 1 ]; then echo "usage: build-kernel.sh OUTPUT.so" >&2; exit 2; fi
if [ -e "$1" ] || [ -L "$1" ]; then echo "output exists" >&2; exit 2; fi
exec c++ -std=c++17 -O3 -fPIC -shared -fopenmp -ffp-contract=off -fno-fast-math "$(dirname "$0")/bf16_lora_kernel.cpp" -o "$1"
