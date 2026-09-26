"""Transcribe and diarize a single recording.

    python scripts/infer.py --audio /data/EN2002b.wav \
        --speakerlab-root /path/to/3D-Speaker --out runs/en2002b

The protocol defaults -- the ones behind every reported number -- are not
re-spelled here: the config is built by ``eval.run_eval.build_config``, the same
function the corpus runner uses, and the audio length is probed from the file.
The speech LLM must already be serving; see README "Install".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_eval import (  # noqa: E402
    DEFAULT_CONTEXT_SEC,
    DEFAULT_STEP_SEC,
    build_config,
    recording_duration_sec,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", required=True, help="path to one recording")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--speakerlab-root",
        required=True,
        help="a 3D-Speaker checkout providing the `speakerlab` package",
    )
    parser.add_argument(
        "--speaker-model",
        default="checkpoints/eres2netv2",
        help="ERes2NetV2 checkpoint or its directory",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:19300/v1",
        help="OpenAI-compatible endpoint serving the speech LLM",
    )
    parser.add_argument(
        "--served-model-name", default="moss-transcribe-diarize"
    )
    parser.add_argument("--device", default="cuda:0", help="embedding device")
    parser.add_argument("--context-sec", type=float, default=DEFAULT_CONTEXT_SEC)
    parser.add_argument("--step-sec", type=float, default=DEFAULT_STEP_SEC)
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    if not audio.is_file():
        parser.error(f"not a file: {audio}")

    recording = {"audio_name": audio.stem, "audio_path": str(audio)}
    config = build_config(
        recording,
        endpoint=args.endpoint,
        served_model_name=args.served_model_name,
        speaker_model=args.speaker_model,
        speakerlab_root=args.speakerlab_root,
        embedding_device=args.device,
        embedding_batch_size=64,
        context_sec=args.context_sec,
        step_sec=args.step_sec,
        calibration=False,
        calibration_min_gain=0.0,
        calibration_min_cosine=0.0,
        crfs=False,
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"{audio.name}: {recording_duration_sec(recording):.1f}s -> {out}")

    from lego_stream import run_streaming_file

    run_streaming_file(config, out)
    result = json.loads((out / "hypothesis.json").read_text(encoding="utf-8"))
    segments = result[0].get("segments", []) if result else []
    for segment in segments:
        print(
            f"[{float(segment['start']):7.2f} - {float(segment['end']):7.2f}] "
            f"{segment['speaker']}  {segment['text']}"
        )
    print(f"{len(segments)} segments -> {out / 'hypothesis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
