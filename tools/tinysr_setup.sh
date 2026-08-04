#!/bin/bash
# Clone + patch + build the vendored TinySR (tinysr/ is gitignored).
# Patches applied by tools/patch_tinysr.py (idempotent):
#  - full_reco stdout line-buffering
#  - iterative FFT + heap DSP tables + tinysr_free_scratch() for RP2040
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d tinysr ]; then
    git clone --depth 1 https://github.com/petersn/tinysr
fi
python3 tools/patch_tinysr.py
make -C tinysr
echo "tinysr ready: apps/full_reco, apps/store_utters"
