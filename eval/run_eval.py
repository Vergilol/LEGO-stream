"""Run LEGO-Stream over a corpus and write one hypothesis file per recording.

Every path is supplied by the caller. Nothing is discovered by walking the
filesystem, and no dataset, checkpoint, or endpoint is baked in, so a run on
another machine differs only in the flags.

    python -m eval.run_eval \
        --manifest data/ami_ihm.json \
        --run-root runs/ami_ihm \
        --speaker-model /path/to/eres2netv2 \
        --speakerlab-root /path/to/3D-Speaker \
        --endpoint http://127.0.0.1:19300/v1

The manifest is a JSON list of recordings:

    [{"audio_name": "EN2002a",
      "audio_path": "/abs/path/EN2002a.wav",
      "segments": [{"start": 0.0, "end": 1.9, "speaker": "A", "text": "..."}]}]

``segments`` is the reference. It is read only to be copied into
``reference.json`` for scoring; the streaming run itself never sees it, which is
what keeps the evaluation causal. Scoring is a separate step -- see
``eval/README.md`` -- because the metric implementation is deliberately not
bundled with the system under test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

# The configuration behind every LEGO-Stream result. Changing one puts you
# off-protocol, so they are named constants rather than scattered literals.
DEFAULT_CONTEXT_SEC = 60.0
DEFAULT_STEP_SEC = 2.0
DEFAULT_RIGHT_CONTEXT_SEC = 2.0
# Used only when ``--calibration`` is passed; see ``eval/README.md``.
CALIBRATION_MIN_GAIN = 0.05  # gamma in Eq. (3)
CALIBRATION_MIN_COSINE = 0.50  # kappa in Eq. (3)
CALIBRATION_MIN_DURATION_SEC = 1.5
# The identity policy is deliberately not re-spelled here. It is ``LEGO_POLICY``,
# named once in the library that reads it, and imported where the config is built.


def _audio_file_duration(audio_path: str) -> float | None:
    """Duration read from the audio file itself, or ``None`` if unreadable."""

    path = Path(audio_path).expanduser()
    if not path.is_file():
        return None
    try:
        import soundfile  # type: ignore

        info = soundfile.info(str(path))
        duration = float(info.frames) / float(info.samplerate)
    except Exception:
        try:
            import wave

            with wave.open(str(path), "rb") as handle:
                duration = handle.getnframes() / float(handle.getframerate())
        except Exception:
            return None
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def recording_duration_sec(recording: Mapping[str, Any]) -> float:
    """Length of one recording, preferring the audio file over the reference.

    The audio file is authoritative when it is readable, because the reference
    stops at the last annotated word and so understates the tail -- and the
    window schedule must cover audio the reference never mentions. A declared
    ``duration_sec`` in the manifest wins over both, so a corpus with a known
    evaluation extent can pin it.
    """

    declared = recording.get("duration_sec")
    if declared not in (None, ""):
        duration = float(declared)
        if math.isfinite(duration) and duration > 0.0:
            return duration
        raise ValueError(
            f"manifest declares a non-positive duration_sec for "
            f"{recording.get('audio_name')!r}"
        )

    from_audio = _audio_file_duration(str(recording["audio_path"]))
    if from_audio is not None:
        return from_audio

    segments = recording.get("segments")
    ends = [
        float(item["end"])
        for item in (segments if isinstance(segments, Sequence) else ())
        if isinstance(item, Mapping) and item.get("end") not in (None, "")
    ]
    reference_extent = max(ends) if ends else 0.0
    if not math.isfinite(reference_extent) or reference_extent <= 0.0:
        raise ValueError(
            "cannot determine a duration for "
            f"{recording.get('audio_name')!r}: the audio file is unreadable and "
            "the manifest supplies neither duration_sec nor reference segments"
        )
    return reference_extent


def build_config(
    recording: Mapping[str, Any],
    *,
    endpoint: str,
    served_model_name: str,
    speaker_model: str,
    speakerlab_root: str,
    embedding_device: str,
    embedding_batch_size: int,
    context_sec: float,
    step_sec: float,
    calibration: bool,
    calibration_min_gain: float,
    calibration_min_cosine: float,
    crfs: bool,
) -> dict[str, Any]:
    """Build the per-recording config consumed by ``run_streaming_file``."""

    # Deferred for the same reason as the import in ``main``: pulling the
    # package in costs torch, and ``--help`` must not.
    from lego_stream.core.identity.speakers import LEGO_POLICY

    audio_path = str(recording["audio_path"])
    audio_name = str(recording.get("audio_name") or Path(audio_path).stem)
    association: dict[str, Any] = {
        "mode": "causal_embedding",
        "policy": LEGO_POLICY,
        "min_overlap_sec": 0.1,
        "core_min_duration_sec": 1.5,
        "core_min_quality": 0.25,
        "best_effort_after_ticks": 1,
        "publish_pending_speakers": False,
    }
    if calibration:
        association["centroid_correction_enabled"] = True
        association["centroid_correction_min_gain"] = float(calibration_min_gain)
        association["centroid_correction_min_cosine"] = float(calibration_min_cosine)
        association["centroid_correction_min_duration_sec"] = (
            CALIBRATION_MIN_DURATION_SEC
        )
    return {
        "endpoint": endpoint,
        "served_model_name": served_model_name,
        "sample": {
            "id": audio_name,
            "audio_name": audio_name,
            "audio_path": audio_path,
            "duration_sec": recording_duration_sec(recording),
        },
        "stream": {
            "step_sec": float(step_sec),
            "context_sec": float(context_sec),
            "right_context_sec": DEFAULT_RIGHT_CONTEXT_SEC,
            "heuristic_overlap_min_sec": 0.20,
            "heuristic_overlap_min_fraction": 0.15,
            "stitch_mode": "incremental",
        },
        "inference": {
            "window_max_new_tokens": 4096,
            "short_window_max_new_tokens": 512,
            "short_window_threshold_sec": 9.0,
            "timeout_sec": 600.0,
            "sse_soft_deadline_sec": 0.0,
            "continue_on_decode_error": True,
            "causal_frontier_request_ids": bool(crfs),
        },
        "embedding": {
            "speakerlab_root": speakerlab_root,
            "model_dir": speaker_model,
            "device": embedding_device,
            "batch_size": int(embedding_batch_size),
            "chunk_duration_sec": 1.5,
            "chunk_step_sec": 0.75,
            "dimension": 192,
            "checkpoint_name": "pretrained_eres2netv2w24s4ep4.ckpt",
        },
        "association": association,
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"manifest must be a JSON list of recordings: {path}")
    for item in value:
        if not isinstance(item, Mapping) or "audio_path" not in item:
            raise ValueError(f"every manifest entry needs an audio_path: {path}")
    return [dict(item) for item in value]


def select(
    recordings: Sequence[Mapping[str, Any]],
    *,
    only: Sequence[str],
    limit: int | None,
    shard_index: int,
    shard_count: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if shard_count < 1 or not (0 <= shard_index < shard_count):
        raise ValueError("require 0 <= shard-index < shard-count")
    wanted = set(only)
    chosen: list[tuple[int, dict[str, Any]]] = []
    for index, recording in enumerate(recordings):
        name = str(recording.get("audio_name") or Path(str(recording["audio_path"])).stem)
        if wanted and name not in wanted:
            continue
        chosen.append((index, dict(recording)))
    if limit is not None:
        chosen = chosen[:limit]
    for position, entry in enumerate(chosen):
        if position % shard_count == shard_index:
            yield entry


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    recordings = load_manifest(Path(args.manifest).expanduser().resolve())
    selected = list(
        select(
            recordings,
            only=args.only,
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    )
    if not selected:
        print("no recordings selected", file=sys.stderr)
        return 2

    # Import late so that --help and argument errors do not require torch.
    from lego_stream import run_streaming_file
    from lego_stream.core.identity.speakers import LEGO_POLICY

    hypotheses: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failures = 0

    for index, recording in selected:
        audio_path = str(recording["audio_path"])
        name = str(recording.get("audio_name") or Path(audio_path).stem)
        destination = run_root / "recordings" / name
        config = build_config(
            recording,
            endpoint=args.endpoint,
            served_model_name=args.served_model_name,
            speaker_model=str(Path(args.speaker_model).expanduser().resolve()),
            speakerlab_root=str(Path(args.speakerlab_root).expanduser().resolve()),
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            context_sec=args.context_sec,
            step_sec=args.step_sec,
            calibration=args.calibration,
            calibration_min_gain=args.calibration_min_gain,
            calibration_min_cosine=args.calibration_min_cosine,
            crfs=args.crfs,
        )
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        started = time.monotonic()
        try:
            summary = run_streaming_file(config, destination)
        except Exception as exc:  # one bad recording must not lose the sweep
            failures += 1
            elapsed = time.monotonic() - started
            print(f"[{index}] FAILED {name}: {exc}", file=sys.stderr)
            rows.append(
                {"audio_name": name, "status": "failed", "error": str(exc)[:512],
                 "wall_sec": round(elapsed, 3)}
            )
            (destination / "error.json").write_text(
                json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if args.fail_fast:
                break
            continue
        elapsed = time.monotonic() - started

        emitted = json.loads(
            (destination / "hypothesis.json").read_text(encoding="utf-8")
        )
        hypotheses.extend(emitted)
        references.append(
            {
                "audio_name": name,
                "audio_path": audio_path,
                "segments": list(recording.get("segments") or []),
            }
        )
        # ``run_streaming_file`` reports the stream under a "stream" key rather
        # than flattening it, and the audio length is not echoed back at all --
        # we already resolved it to build the config, so read it from there.
        # Reading these from the top level instead silently yields rtf=None and
        # status="unknown" for every recording, which is how RTF (a headline
        # number in the paper) goes missing without any error.
        duration = float(config["sample"]["duration_sec"])
        stream = summary.get("stream")
        stream = stream if isinstance(stream, Mapping) else {}
        rtf = (elapsed / duration) if duration > 0 else None
        rows.append(
            {
                "audio_name": name,
                "status": str(stream.get("status") or "unknown"),
                "duration_sec": round(duration, 3),
                "wall_sec": round(elapsed, 3),
                "rtf": None if rtf is None else round(rtf, 4),
                "segment_count": len(emitted[0].get("segments", [])) if emitted else 0,
            }
        )
        print(
            f"[{index}] {name}: {rows[-1]['status']} "
            f"rtf={rows[-1]['rtf']} segments={rows[-1]['segment_count']}"
        )

    _write(run_root / "hypothesis.json", hypotheses)
    _write(run_root / "reference.json", references)

    total_wall = sum(float(row.get("wall_sec") or 0.0) for row in rows)
    total_audio = sum(float(row.get("duration_sec") or 0.0) for row in rows)
    _write(
        run_root / "run_summary.json",
        {
            "recordings": rows,
            "completed": len(rows) - failures,
            "failed": failures,
            "duration_weighted_rtf": (
                round(total_wall / total_audio, 4) if total_audio > 0 else None
            ),
            "protocol": {
                "context_sec": args.context_sec,
                "step_sec": args.step_sec,
                "calibration": bool(args.calibration),
                "calibration_min_gain": args.calibration_min_gain,
                "calibration_min_cosine": args.calibration_min_cosine,
                "crfs": bool(args.crfs),
                "policy": LEGO_POLICY,
            },
        },
    )
    print(
        f"\n{len(rows) - failures} completed, {failures} failed -> {run_root}\n"
        f"hypothesis.json and reference.json are ready for scoring."
    )
    return 1 if failures and args.fail_fast else 0


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Run LEGO-Stream over a corpus manifest.",
    )
    parser.add_argument("--manifest", required=True, help="JSON list of recordings")
    parser.add_argument("--run-root", required=True, help="output directory")
    parser.add_argument(
        "--speaker-model",
        required=True,
        help="directory holding the ERes2NetV2 checkpoint",
    )
    parser.add_argument(
        "--speakerlab-root",
        required=True,
        help="path to a 3D-Speaker checkout providing the speakerlab package",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MOSS_ENDPOINT", "http://127.0.0.1:19300/v1"),
        help=(
            "OpenAI-compatible endpoint serving the speech LLM "
            "(default: $MOSS_ENDPOINT, else http://127.0.0.1:19300/v1)"
        ),
    )
    parser.add_argument(
        "--served-model-name",
        default=os.environ.get("MOSS_SERVED_MODEL_NAME", "moss-transcribe-diarize"),
        help=(
            "--served-model-name the endpoint was launched with "
            "(default: $MOSS_SERVED_MODEL_NAME, else moss-transcribe-diarize)"
        ),
    )
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--context-sec",
        type=float,
        default=DEFAULT_CONTEXT_SEC,
        help=f"rolling window C (paper: {DEFAULT_CONTEXT_SEC})",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=DEFAULT_STEP_SEC,
        help=f"tick spacing delta (paper: {DEFAULT_STEP_SEC})",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help=(
            "enable release-time label calibration. Off by default; see "
            "eval/README.md. Uses "
            f"gamma={CALIBRATION_MIN_GAIN}, kappa={CALIBRATION_MIN_COSINE}"
        ),
    )
    parser.add_argument(
        "--calibration-min-gain",
        type=float,
        default=CALIBRATION_MIN_GAIN,
        help="gamma: margin a rival label must win by",
    )
    parser.add_argument(
        "--calibration-min-cosine",
        type=float,
        default=CALIBRATION_MIN_COSINE,
        help="kappa: absolute floor a rival label must clear",
    )
    parser.add_argument(
        "--crfs",
        action="store_true",
        help="enable frontier speculation (2.22x median measured). Needs the "
        "server started with --speculative-config "
        "'{\"method\":\"suffix\",\"num_speculative_tokens\":16}'; inert "
        "otherwise. OFF for every accuracy number in the paper -- it shifts "
        "turn boundaries, so do not report quality with this on",
    )
    parser.add_argument("--only", action="append", default=[], help="audio_name filter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
