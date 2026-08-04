#!/bin/bash
# Enroll one vocabulary word: streams board audio into TinySR's utterance
# splitter. Say the word repeatedly (10-15x) with ~1s pauses while it runs.
# Usage: ./tools/enroll_word.sh <word> [seconds=35]
set -euo pipefail
cd "$(dirname "$0")/.."
WORD="${1:?usage: enroll_word.sh <word> [seconds]}"
SECS="${2:-35}"
DIR="tinysr_data/$WORD"
mkdir -p "$DIR"
BEFORE=$(ls "$DIR" 2>/dev/null | wc -l | tr -d ' ')
echo "=== Enrolling '$WORD' for ${SECS}s — say \"$WORD\" ~12 times with 1s pauses ==="
python3 tools/mic_stream.py "$SECS" | ./tinysr/apps/store_utters "$DIR"
AFTER=$(ls "$DIR" | wc -l | tr -d ' ')
echo "=== '$WORD': captured $((AFTER - BEFORE)) new utterances (total $AFTER) ==="
