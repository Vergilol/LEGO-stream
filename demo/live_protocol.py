"""The ``WS /api/live`` wire format.

Client to server, all JSON except the audio itself:

    {"type": "start", "sample_rate": 16000, "channels": 1,
     "format": "pcm_s16le", "step_sec": 2.0}
    {"type": "chunk", "sequence": 0, "byte_length": 32000}
        followed immediately by exactly that many bytes in one binary frame
    {"type": "eos"}
    {"type": "cancel"}

Server to client:

    {"type": "session_started", ...}     answer to ``start``
    {"type": "chunk_accepted", "sequence": 0, "duration_sec": 2.0}
    {"type": "tick", ...}                 one per processed tick
    {"type": "done", ...}                 the committed transcript
    {"type": "error", "message": "..."}   terminal
    {"type": "closed"}                    internal sentinel, see below

``closed`` never reaches the page: it tells the route the worker is done.

There is no rate negotiation. The browser resamples to 16 kHz, and any other
rate is rejected rather than misread as 16 kHz.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

#: The only rate the embedding store and the decoder accept.
SAMPLE_RATE = 16000
#: Mono only: the diarizer has no channel notion.
CHANNELS = 1
#: Little-endian signed 16-bit, which is what ``AudioWorklet`` hands back.
SAMPLE_FORMAT = "pcm_s16le"
BYTES_PER_SAMPLE = 2

#: One second of audio; the page sends 0.1 s frames.
MAX_FRAME_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE
#: How long an error message may be before it reaches the page.
MAX_ERROR_CHARS = 500
#: "Policy violation". Sent after an ``error``, so the page can show the reason
#: before the socket dies.
CLOSE_POLICY_VIOLATION = 1008

#: Default ceiling on how far capture may run ahead of the last processed tick.
DEFAULT_MAX_BACKLOG_SEC = 30.0
#: Default ceiling on one session; also the nominal ``sample.duration_sec``
#: until the real length is known.
DEFAULT_MAX_SESSION_SEC = 3600.0


class LiveProtocolError(ValueError):
    """The client sent something the wire format does not allow."""


def _mapping(message: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(message, Mapping):
        raise LiveProtocolError(f"{what} must be a JSON object")
    return message


def _finite_number(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LiveProtocolError(f"{what} must be a number") from None
    if not math.isfinite(number):
        raise LiveProtocolError(f"{what} must be finite")
    return number


def _whole_number(value: Any, what: str) -> int:
    # ``bool`` is an ``int`` in Python; ``{"sequence": true}`` is not a sequence.
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiveProtocolError(f"{what} must be an integer")
    return value


def parse_start(
    message: Any,
    *,
    step_sec: float | None = None,
) -> float:
    """Validate a ``start`` frame; return the tick step to drive the session.

    ``step_sec`` is the server's ``--step-sec``. A client may omit the field
    but not pick another value: the step decides which windows are decoded.
    """

    start = _mapping(message, "the live start message")
    if str(start.get("type", "")) != "start":
        raise LiveProtocolError("the first live message must be a start message")

    sample_rate = _whole_number(start.get("sample_rate"), "sample_rate")
    if sample_rate != SAMPLE_RATE:
        raise LiveProtocolError(
            f"the live microphone must be {SAMPLE_RATE} Hz, got {sample_rate}"
        )
    channels = _whole_number(start.get("channels"), "channels")
    if channels != CHANNELS:
        raise LiveProtocolError(f"live audio must be mono, got {channels} channels")
    sample_format = str(start.get("format", ""))
    if sample_format != SAMPLE_FORMAT:
        raise LiveProtocolError(
            f"live audio must be {SAMPLE_FORMAT}, got {sample_format or 'nothing'}"
        )

    if step_sec is None:
        return _positive(start.get("step_sec"), "step_sec")

    server_step = _positive(step_sec, "step_sec")
    requested = start.get("step_sec")
    if requested is None:
        return server_step
    if abs(_positive(requested, "step_sec") - server_step) > 1e-9:
        raise LiveProtocolError(
            f"this server ticks every {server_step:g}s; a client may not pick "
            "its own step"
        )
    return server_step


def _positive(value: Any, what: str) -> float:
    number = _finite_number(value, what)
    if number <= 0.0:
        raise LiveProtocolError(f"{what} must be positive")
    return number


def control_type(message: Any) -> str:
    """The type of a client control frame, or a protocol error."""

    control = _mapping(message, "a live control message")
    message_type = str(control.get("type", "") or "")
    if message_type not in CONTROL_TYPES:
        raise LiveProtocolError(
            "unsupported live control message: "
            f"{message_type or '(no type field)'}"
        )
    return message_type


CONTROL_TYPES = frozenset({"start", "chunk", "eos", "cancel"})


def parse_chunk_header(
    message: Any,
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> tuple[int, int]:
    """Validate a ``chunk`` header; return ``(sequence, byte_length)``."""

    header = _mapping(message, "a live chunk header")
    sequence = _whole_number(header.get("sequence"), "chunk sequence")
    if sequence < 0:
        raise LiveProtocolError("chunk sequence must not be negative")
    byte_length = _whole_number(header.get("byte_length"), "chunk byte_length")
    if byte_length <= 0:
        raise LiveProtocolError("chunk byte_length must be positive")
    if byte_length > max_frame_bytes:
        raise LiveProtocolError(
            f"chunk of {byte_length} bytes exceeds the {max_frame_bytes}-byte "
            "limit; send audio in smaller frames"
        )
    if byte_length % BYTES_PER_SAMPLE:
        # Clearer here than numpy's buffer-size error.
        raise LiveProtocolError(
            f"chunk byte_length {byte_length} is not a whole number of "
            f"{SAMPLE_FORMAT} samples"
        )
    return sequence, byte_length


def decode_pcm16(payload: bytes, *, expected_bytes: int) -> np.ndarray:
    """One binary frame of little-endian PCM16 as mono float32 in [-1, 1)."""

    if len(payload) != expected_bytes:
        raise LiveProtocolError(
            f"chunk header announced {expected_bytes} bytes but the frame "
            f"carried {len(payload)}"
        )
    samples = np.frombuffer(payload, dtype="<i2")
    # /32768 maps the asymmetric int16 range onto [-1, 1), as the page encodes.
    return samples.astype(np.float32) / 32768.0


def error_message(exc: BaseException) -> str:
    """A failure, short enough to put on screen."""

    text = str(exc).strip() or exc.__class__.__name__
    return text[:MAX_ERROR_CHARS]


def session_started(*, sample_rate: int = SAMPLE_RATE, step_sec: float) -> dict[str, Any]:
    return {
        "type": "session_started",
        "sample_rate": sample_rate,
        "channels": CHANNELS,
        "format": SAMPLE_FORMAT,
        "step_sec": float(step_sec),
    }


def chunk_accepted(*, sequence: int, duration_sec: float) -> dict[str, Any]:
    return {
        "type": "chunk_accepted",
        "sequence": int(sequence),
        "duration_sec": float(duration_sec),
    }


def error_frame(exc: BaseException) -> dict[str, Any]:
    return {"type": "error", "message": error_message(exc)}


__all__ = [
    "BYTES_PER_SAMPLE",
    "CHANNELS",
    "CLOSE_POLICY_VIOLATION",
    "CONTROL_TYPES",
    "DEFAULT_MAX_BACKLOG_SEC",
    "DEFAULT_MAX_SESSION_SEC",
    "LiveProtocolError",
    "MAX_ERROR_CHARS",
    "MAX_FRAME_BYTES",
    "SAMPLE_FORMAT",
    "SAMPLE_RATE",
    "chunk_accepted",
    "control_type",
    "decode_pcm16",
    "error_frame",
    "error_message",
    "parse_chunk_header",
    "parse_start",
    "session_started",
]
