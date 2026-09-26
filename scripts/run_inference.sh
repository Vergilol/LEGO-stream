#!/usr/bin/env bash
#
# Transcribe and diarize one recording.
#
#   ./scripts/run_inference.sh /data/EN2002b.wav \
#       --speakerlab-root /path/to/3D-Speaker
#
# The audio length is probed and the protocol defaults are filled in, so no
# config has to be kept in sync. The speech LLM must already be serving; see
# README "Install". Transcript lines are printed, and also written to
# <out>/hypothesis.json.
#
# Options are passed straight through to `scripts/infer.py --help`.
# Environment defaults: SPEAKERLAB_ROOT, SPEAKER_MODEL, MOSS_ENDPOINT, PYTHON.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

AUDIO=""
REST=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            echo
            "$PYTHON" "$REPO_ROOT/scripts/infer.py" --help
            exit 0
            ;;
        --*) REST+=("$1"); shift ;;
        *)
            if [ -z "$AUDIO" ]; then
                AUDIO="$1"
            else
                REST+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "$AUDIO" ]; then
    echo "usage: $(basename "$0") AUDIO [--speakerlab-root DIR] [options]" >&2
    echo "see --help" >&2
    exit 2
fi

has_flag() {
    local flag="$1" item
    for item in ${REST[@]+"${REST[@]}"}; do
        [ "$item" = "$flag" ] && return 0
    done
    return 1
}

DEFAULTS=()
if ! has_flag --speakerlab-root; then
    if [ -z "${SPEAKERLAB_ROOT:-}" ]; then
        echo "run_inference.sh: --speakerlab-root (or \$SPEAKERLAB_ROOT) is required" >&2
        exit 2
    fi
    DEFAULTS+=(--speakerlab-root "$SPEAKERLAB_ROOT")
fi
has_flag --speaker-model ||
    DEFAULTS+=(--speaker-model "${SPEAKER_MODEL:-$REPO_ROOT/checkpoints/eres2netv2}")
has_flag --endpoint ||
    DEFAULTS+=(--endpoint "${MOSS_ENDPOINT:-http://127.0.0.1:19300/v1}")

if ! has_flag --out; then
    DEFAULTS+=(--out "$REPO_ROOT/runs/$(basename "$AUDIO" | sed 's/\.[^.]*$//')")
fi

cd "$REPO_ROOT"
exec "$PYTHON" scripts/infer.py \
    --audio "$AUDIO" \
    "${DEFAULTS[@]}" \
    ${REST[@]+"${REST[@]}"}
