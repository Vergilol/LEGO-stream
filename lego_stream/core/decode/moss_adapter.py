"""Model-I/O adapter for the canonical streaming implementation.

This module intentionally owns the small amount of MOSS wire compatibility
needed by streaming.  It does not import the upstream model repository, its
rolling evaluator, or its Python dependencies: the endpoint is just HTTP, and
the transcript grammar is parsed locally.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import re
import time
import wave
from pathlib import Path
from typing import Any, List, Mapping, Sequence
import urllib.error
import urllib.request
import uuid

import numpy as np

from ..audio_io import load_audio_frames
from .engine import DecodeOutput, WindowSpec
from ..types import Segment, canonical_speaker_label


DEFAULT_TRANSCRIPTION_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳。"
)


@dataclass(frozen=True)
class ParsedTranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str


_MARKER_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\[([SG][^\]]*)\]")
_END_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]")


def _read_audio_file(audio_path: Path) -> tuple[np.ndarray, int]:
    """Read the PCM WAV inputs used by streaming without a libsndfile dependency."""

    values, sample_rate, _loader = load_audio_frames(audio_path)
    return values, sample_rate


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode normalized mono/multichannel float audio as PCM16 WAV bytes."""

    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError("audio must have shape [frames] or [frames, channels]")
    if int(sample_rate) <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("audio and sample_rate must be finite and valid")
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(int(values.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def parse_transcript(text: str) -> List[ParsedTranscriptSegment]:
    """Parse MOSS ``[start][Sxx]text[end]`` output without nested imports.

    The parser deliberately keeps punctuation and only strips surrounding
    whitespace.  Ordinary leading comments are ignored, while malformed
    speaker-like markers fail closed.  Equivalent speaker spellings such as
    ``S3``/``S03`` or legacy ``G3`` are normalized to the same local label.
    """

    if not isinstance(text, str):
        raise TypeError("transcript text must be a string")
    markers = list(_MARKER_RE.finditer(text))
    canonical_markers: List[tuple[Any, str]] = []
    for marker in markers:
        raw_speaker = marker.group(2)
        speaker = canonical_speaker_label(raw_speaker)
        canonical_markers.append((marker, speaker))
    parsed: List[ParsedTranscriptSegment] = []
    for marker, speaker in canonical_markers:
        start = float(marker.group(1))
        end_match = _END_RE.search(text, marker.end())
        if end_match is None:
            continue
        end = float(end_match.group(1))
        if end < start:
            continue
        body = text[marker.end() : end_match.start()].strip()
        if not body:
            continue
        parsed.append(
            ParsedTranscriptSegment(
                start=start,
                end=end,
                speaker=speaker,
                text=body,
            )
        )
    return parsed


def _multipart_body(
    *,
    boundary: str,
    fields: Mapping[str, str],
    file_bytes: bytes,
    filename: str = "audio.wav",
) -> bytes:
    chunks: List[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                ("--%s\r\n" % boundary).encode("utf-8"),
                ('Content-Disposition: form-data; name="%s"\r\n\r\n' % name).encode(
                    "utf-8"
                ),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            ("--%s\r\n" % boundary).encode("utf-8"),
            (
                'Content-Disposition: form-data; name="file"; filename="%s"\r\n'
                % filename
            ).encode("utf-8"),
            b"Content-Type: audio/wav\r\n\r\n",
            file_bytes,
            b"\r\n",
            ("--%s--\r\n" % boundary).encode("utf-8"),
        ]
    )
    return b"".join(chunks)


def _extract_text(payload: Mapping[str, Any]) -> str:
    value = payload.get("text")
    if isinstance(value, str):
        return value
    choices = payload.get("choices")
    if isinstance(choices, Sequence):
        parts: List[str] = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                parts.append(str(delta["content"]))
                continue
            for key in ("text", "content"):
                candidate = choice.get(key)
                if isinstance(candidate, str):
                    parts.append(candidate)
                    break
        return "".join(parts)
    return ""


def _consume_sse(
    response: Any,
    *,
    soft_deadline_sec: float | None = None,
    time_fn: Any = time.perf_counter,
    audit_tap: Any = None,
    request_started_sec: float | None = None,
) -> Mapping[str, Any]:
    usage: dict[str, int] = {}
    cumulative_text: dict[int, str] = {}
    text = ""
    repetition_loop = False
    marker_text_loop = False
    sse_soft_deadline_reached = False
    deadline = float(soft_deadline_sec or 0.0)
    started = (
        float(request_started_sec)
        if request_started_sec is not None
        else float(time_fn())
    )
    first_text_sec: float | None = None
    last_receipt_sec = started
    observer_elapsed_sec = 0.0

    def audit_line(raw_text: str, sequence_id: int) -> None:
        nonlocal observer_elapsed_sec
        if not callable(audit_tap):
            return
        audit_started = time.perf_counter()
        try:
            audit_tap(
                {
                    "raw_line": raw_text,
                    "receipt_monotonic_sec": time.perf_counter(),
                    "sequence_id": int(sequence_id),
                }
            )
        except Exception:
            pass
        finally:
            observer_elapsed_sec += max(0.0, time.perf_counter() - audit_started)

    for sequence_id, raw_line in enumerate(response):
        receipt_sec = float(time_fn())
        last_receipt_sec = receipt_sec
        if isinstance(raw_line, bytes):
            raw_text = raw_line.decode("utf-8", errors="replace")
        else:
            raw_text = str(raw_line)
        line = raw_text.strip()
        if not line.startswith("data:"):
            audit_line(raw_text, sequence_id)
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            audit_line(raw_text, sequence_id)
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            audit_line(raw_text, sequence_id)
            continue
        if not isinstance(chunk, Mapping):
            audit_line(raw_text, sequence_id)
            continue
        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, Mapping):
            for key in ("prompt_tokens", "completion_tokens"):
                if isinstance(chunk_usage.get(key), int):
                    usage[key] = int(chunk_usage[key])
        choices = chunk.get("choices")
        if isinstance(choices, Sequence):
            for choice_index, choice in enumerate(choices):
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                    delta_text = str(delta["content"])
                    candidate = text + delta_text
                    if _looks_like_marker_only_loop(candidate):
                        text = candidate
                        repetition_loop = True
                        break
                    if _looks_like_marker_text_loop(candidate):
                        text = ""
                        repetition_loop = True
                        marker_text_loop = True
                        break
                    if _looks_like_repetition_loop(candidate):
                        repetition_loop = True
                        break
                    text = candidate
                    if delta_text and first_text_sec is None:
                        first_text_sec = receipt_sec
                elif isinstance(choice.get("text"), str):
                    # Some OpenAI-compatible servers expose ``choice.text``
                    # as the complete cumulative hypothesis on every SSE
                    # event, while others expose an incremental piece.  A
                    # blind append turns ``a``, ``ab``, ``abc`` into
                    # ``aababc``.  Prefer the longest-common-prefix suffix
                    # when the new value extends the previous one; otherwise
                    # treat it as an incremental piece.  Exact replays and
                    # shorter cumulative snapshots are ignored.
                    piece = str(choice["text"])
                    previous = cumulative_text.get(choice_index, "")
                    if piece == previous or (previous and previous.startswith(piece)):
                        continue
                    if previous and piece.startswith(previous):
                        delta_text = piece[len(previous) :]
                    else:
                        delta_text = piece
                    cumulative_text[choice_index] = piece
                    candidate = text + delta_text
                    if _looks_like_marker_only_loop(candidate):
                        text = candidate
                        repetition_loop = True
                        break
                    if _looks_like_marker_text_loop(candidate):
                        text = ""
                        repetition_loop = True
                        marker_text_loop = True
                        break
                    if _looks_like_repetition_loop(candidate):
                        repetition_loop = True
                        break
                    text = candidate
                    if delta_text and first_text_sec is None:
                        first_text_sec = receipt_sec
            if repetition_loop:
                break
        audit_line(raw_text, sequence_id)
        if (
            deadline > 0.0
            and receipt_sec - float(started) - observer_elapsed_sec >= deadline
        ):
            sse_soft_deadline_reached = True
            break
    return {
        "text": text,
        "usage": usage,
        "repetition_loop": repetition_loop,
        "marker_text_loop": marker_text_loop,
        "sse_soft_deadline_reached": sse_soft_deadline_reached,
        "time_to_first_text_sec": (
            None
            if first_text_sec is None
            else max(0.0, first_text_sec - started)
        ),
        "generation_tail_sec": (
            0.0
            if first_text_sec is None
            else max(0.0, last_receipt_sec - first_text_sec)
        ),
    }


def _looks_like_repetition_loop(text: str) -> bool:
    """Detect a runaway short-tail repetition loop in generated text."""

    compact = re.sub(r"\s+", "", str(text))
    if len(compact) < 16:
        return False
    # Runaway MOSS windows sometimes repeat a whole timestamp/speaker/text
    # micro-segment such as ``[1.56][S01]对。[1.56][1.28][S04]对。[1.56]``.
    # That unit is much longer than a plain lexical loop, so the detector must
    # inspect enough tail context and allow longer repeated units.
    tail = compact[-1024:]
    min_run = 8
    max_unit = min(128, len(tail) // min_run)
    if max_unit <= 0:
        return False
    for unit_len in range(1, max_unit + 1):
        unit = tail[-unit_len:]
        if not unit or not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in unit):
            continue
        run = 1
        idx = len(tail) - unit_len
        while idx >= unit_len and tail[idx - unit_len : idx] == unit:
            run += 1
            idx -= unit_len
        if run >= min_run and idx <= len(tail) - unit_len * min_run:
            return True
    return False


def _looks_like_marker_only_loop(text: str) -> bool:
    """Detect loops made only of timestamp/speaker markers."""

    compact = re.sub(r"\s+", "", str(text))
    if len(_MARKER_RE.findall(compact)) < 8:
        return False
    without_start_markers = _MARKER_RE.sub("", compact)
    without_markers = _END_RE.sub("", without_start_markers)
    return without_markers.strip(".,，。!！?？;；:：-—…[]") == ""


def _looks_like_marker_text_loop(text: str) -> bool:
    """Detect dense timestamp/speaker loops with the same very short text.

    These are not valid fast turn-taking.  In observed bad cases MOSS emits
    dozens of one-character acknowledgements (``对``/``嗯``) with sub-second
    timestamp hops and alternating speakers until the generation budget is
    exhausted.  Dropping the window is safer than publishing this hallucinated
    speaker churn; adjacent overlapping windows can still publish real speech.
    """

    compact = re.sub(r"\s+", "", str(text))
    if len(_MARKER_RE.findall(compact)) < 24:
        return False
    tail = compact[-2048:]
    try:
        parsed = parse_transcript(tail)
    except ValueError:
        return False
    if len(parsed) < 24:
        return False
    recent = parsed[-36:]
    punctuation = ".,，。!！?？;；:：-—…[]、"
    keys: list[tuple[str, ParsedTranscriptSegment]] = []
    for segment in recent:
        key = re.sub(r"\s+", "", segment.text).strip(punctuation)
        if 0 < len(key) <= 2:
            keys.append((key, segment))
    if len(keys) < 24:
        return False
    counts: dict[str, int] = {}
    for key, _segment in keys:
        counts[key] = counts.get(key, 0) + 1
    dominant_key, dominant_count = max(counts.items(), key=lambda item: item[1])
    if dominant_count < 24 or dominant_count / max(1, len(recent)) < 0.65:
        return False
    dominant_segments = [segment for key, segment in keys if key == dominant_key]
    span_start = min(segment.start for segment in dominant_segments)
    span_end = max(segment.end for segment in dominant_segments)
    span = max(0.1, span_end - span_start)
    return dominant_count / span >= 2.5


@dataclass
class MossWindowDecoder:
    """Decode one absolute window through an OpenAI-compatible vLLM endpoint.

    The adapter uses its local parser and standard-library HTTP client, so no
    nested checkout or third-party package is required.
    """

    audio_path: Path
    endpoint: str
    model: str
    prompt: str = ""
    max_new_tokens: int = 4096
    short_window_max_new_tokens: int = 512
    short_window_threshold_sec: float = 9.0
    timeout_sec: float = 600.0
    api_key: str = "EMPTY"
    timestamp_tail_tolerance_sec: float = 0.25
    sse_soft_deadline_sec: float = 0.0
    causal_frontier_request_ids: bool = False
    causal_frontier_time_warp: bool = False
    attribution_observer: Any = None
    audio_source: Any = None

    def __post_init__(self) -> None:
        self.audio_path = Path(self.audio_path).expanduser().resolve()
        if self.audio_source is None:
            if not self.audio_path.is_file():
                raise FileNotFoundError(self.audio_path)
            audio, sample_rate = _read_audio_file(self.audio_path)
            self._audio = np.mean(audio, axis=1).astype(np.float32, copy=False)
        else:
            sample_rate = int(getattr(self.audio_source, "sample_rate", 0))
            self._audio = None
        if sample_rate != 16000:
            raise ValueError(
                "MossWindowDecoder expects 16 kHz audio; got %s for %s"
                % (sample_rate, self.audio_path)
            )
        self._sample_rate = int(sample_rate)
        self._causal_frontier_session_id = (
            uuid.uuid4().hex if bool(self.causal_frontier_request_ids) else ""
        )
        if self.causal_frontier_time_warp and not self.causal_frontier_request_ids:
            raise ValueError(
                "causal_frontier_time_warp requires causal_frontier_request_ids"
            )
        self.endpoint = str(self.endpoint).rstrip("/")
        self.model = str(self.model)
        if not self.model:
            raise ValueError("model must be non-empty")
        if (
            not np.isfinite(float(self.timestamp_tail_tolerance_sec))
            or float(self.timestamp_tail_tolerance_sec) < 0.0
        ):
            raise ValueError(
                "timestamp_tail_tolerance_sec must be finite and non-negative"
            )
        if (
            not np.isfinite(float(self.sse_soft_deadline_sec))
            or float(self.sse_soft_deadline_sec) < 0.0
        ):
            raise ValueError("sse_soft_deadline_sec must be finite and non-negative")

    def __call__(self, window: WindowSpec, prompt: str) -> DecodeOutput:
        return self._decode_window(window, prompt)

    def _decode_window(
        self,
        window: WindowSpec,
        prompt: str,
    ) -> DecodeOutput:
        if self.audio_source is None:
            current_start = max(
                0,
                int(round(float(window.start) * self._sample_rate)),
            )
            current_end = min(
                len(self._audio),
                int(round(float(window.end) * self._sample_rate)),
            )
            if current_end <= current_start:
                return DecodeOutput(segments=(), text="")
            current_audio = self._audio[current_start:current_end]
        else:
            current_audio = np.asarray(
                self.audio_source.read(float(window.start), float(window.end)),
                dtype=np.float32,
            )
            if current_audio.ndim != 1 or current_audio.size <= 0:
                return DecodeOutput(segments=(), text="")

        started = time.perf_counter()
        current_duration = float(window.end) - float(window.start)
        request_max_tokens: int | None = None
        if (
            self.short_window_max_new_tokens > 0
            and current_duration < float(self.short_window_threshold_sec)
        ):
            request_max_tokens = min(int(self.max_new_tokens), int(self.short_window_max_new_tokens))
        request_prompt = str(prompt or self.prompt).strip() or DEFAULT_TRANSCRIPTION_PROMPT
        self._active_audit_request = {
            "request_id": "%s:%.6f" % (self.audio_path.name, float(window.tick)),
            "tick": float(window.tick),
            "window_end": float(window.end),
            "window_start": float(window.start),
        }
        # ``_request`` keeps its two-argument form: the optional keyword belongs
        # to the short-window guard only and must not reach the normal contract.
        if request_max_tokens is None:
            response = self._request(current_audio, request_prompt)
        else:
            response = self._request(
                current_audio,
                request_prompt,
                max_completion_tokens=request_max_tokens,
            )
        raw_extracted_text = _extract_text(response)
        text = raw_extracted_text
        repetition_loop = bool(
            isinstance(response, Mapping) and response.get("repetition_loop")
        )
        marker_text_loop = bool(
            isinstance(response, Mapping) and response.get("marker_text_loop")
        )
        sse_soft_deadline_reached = bool(
            isinstance(response, Mapping) and response.get("sse_soft_deadline_reached")
        )
        if not marker_text_loop and _looks_like_marker_text_loop(text):
            marker_text_loop = True
            repetition_loop = True
            text = ""
        if not repetition_loop and _looks_like_repetition_loop(text):
            repetition_loop = True
        parsed = parse_transcript(text)
        marker_only_loop = bool(not parsed and len(_MARKER_RE.findall(text)) >= 8)
        if marker_only_loop:
            text = ""
        segments: list[Segment] = []
        mapping_drops: list[dict[str, object]] = []
        tail_tolerance_sec = float(self.timestamp_tail_tolerance_sec)
        for parsed_index, item in enumerate(parsed):
            if (
                item.start < -tail_tolerance_sec - 1e-3
                or item.end > current_duration + tail_tolerance_sec + 1e-3
            ):
                mapping_drops.append(
                    {
                        "parsed_index": int(parsed_index),
                        "reason": "timestamp_outside_window_tolerance",
                    }
                )
                continue
            start = item.start + float(window.start)
            end = item.end + float(window.start)
            if end <= start:
                mapping_drops.append(
                    {"parsed_index": int(parsed_index), "reason": "nonpositive_mapped_span"}
                )
                continue
            if end < float(window.start) - 1e-3 or start > float(window.end) + 1e-3:
                mapping_drops.append(
                    {"parsed_index": int(parsed_index), "reason": "mapped_span_outside_window"}
                )
                continue
            segments.append(
                Segment(
                    start=max(float(window.start), start),
                    end=min(float(window.end), end),
                    speaker=item.speaker,
                    text=item.text,
                    source_tick=window.tick,
                )
            )
        try:
            from ..attribution import (
                parser_diagnostics,
                safe_observer_record,
            )

            diagnostics = parser_diagnostics(text)
            safe_observer_record(
                self.attribution_observer,
                {
                    "filtered_text": text,
                    "mapped_segments": [item.to_dict() for item in segments],
                    "mapping_drops": mapping_drops,
                    "parser_diagnostics": diagnostics,
                    "raw_extracted_text": raw_extracted_text,
                    "request_id": self._active_audit_request["request_id"],
                    "stage": "decoder_snapshot",
                    "tick": float(window.tick),
                    "window_end": float(window.end),
                    "window_start": float(window.start),
                },
            )
        except Exception:
            pass
        usage = response.get("usage") if isinstance(response, Mapping) else {}
        if not isinstance(usage, Mapping):
            usage = {}
        time_to_first_text_sec = (
            response.get("time_to_first_text_sec")
            if isinstance(response, Mapping)
            else None
        )
        generation_tail_sec = (
            response.get("generation_tail_sec", 0.0)
            if isinstance(response, Mapping)
            else 0.0
        )
        return DecodeOutput(
            segments=segments,
            text=text,
            elapsed_sec=time.perf_counter() - started,
            metadata={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "time_to_first_text_sec": (
                    None
                    if time_to_first_text_sec is None
                    else float(time_to_first_text_sec)
                ),
                "generation_tail_sec": float(generation_tail_sec or 0.0),
                "request_max_completion_tokens": int(
                    request_max_tokens if request_max_tokens is not None else self.max_new_tokens
                ),
                "marker_only_loop": marker_only_loop,
                "marker_text_loop": marker_text_loop,
                "repetition_loop": repetition_loop,
                "sse_soft_deadline_reached": sse_soft_deadline_reached,
            },
        )

    def _request(
        self,
        audio: np.ndarray,
        prompt: str,
        *,
        max_completion_tokens: int | None = None,
    ) -> Mapping[str, Any]:
        file_bytes = _encode_wav(audio, self._sample_rate)
        boundary = "----moss-stream-%s" % uuid.uuid4().hex
        fields = {
            "model": self.model,
            "prompt": str(prompt).strip(),
            "response_format": "json",
            "stream": "true",
            "stream_include_usage": "true",
            "max_completion_tokens": str(
                int(max_completion_tokens if max_completion_tokens is not None else self.max_new_tokens)
            ),
            "temperature": "0",
        }
        body = _multipart_body(
            boundary=boundary,
            fields=fields,
            file_bytes=file_bytes,
        )
        url = self.endpoint
        if not url.endswith("/audio/transcriptions"):
            url += "/audio/transcriptions" if url.endswith("/v1") else "/v1/audio/transcriptions"
        headers = {
            "Content-Type": "multipart/form-data; boundary=%s" % boundary,
            "Authorization": "Bearer %s" % self.api_key,
        }
        if self._causal_frontier_session_id:
            context = dict(getattr(self, "_active_audit_request", {}))
            tick = context.get("tick")
            if tick is not None and np.isfinite(float(tick)):
                tick_us = int(round(float(tick) * 1_000_000.0))
                if self.causal_frontier_time_warp:
                    window_start = context.get("window_start")
                    if window_start is None or not np.isfinite(float(window_start)):
                        raise RuntimeError(
                            "causal frontier time warp requires finite window_start"
                        )
                    window_start_us = int(round(float(window_start) * 1_000_000.0))
                    if window_start_us < 0 or window_start_us > tick_us:
                        raise RuntimeError(
                            "causal frontier window_start must satisfy 0 <= start <= tick"
                        )
                    headers["X-Request-Id"] = (
                        "lego-crfs-v2-%s-%d-%d"
                        % (
                            self._causal_frontier_session_id,
                            tick_us,
                            window_start_us,
                        )
                    )
                else:
                    headers["X-Request-Id"] = (
                        "lego-crfs-v1-%s-%d"
                        % (self._causal_frontier_session_id, tick_us)
                    )
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        request_started_sec = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=float(self.timeout_sec)) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    soft_deadline = float(self.sse_soft_deadline_sec)
                    if soft_deadline <= 0.0:
                        soft_deadline = float(self.timeout_sec)
                    request_id = str(
                        getattr(self, "_active_audit_request", {}).get(
                            "request_id", "unknown"
                        )
                    )

                    def audit_tap(value: Mapping[str, object]) -> None:
                        try:
                            from ..attribution import safe_observer_record

                            safe_observer_record(
                                self.attribution_observer,
                                {
                                    **dict(value),
                                    "request_id": request_id,
                                    "stage": "transport_chunk",
                                },
                            )
                        except Exception:
                            pass

                    if self.attribution_observer is None:
                        return _consume_sse(
                            response,
                            soft_deadline_sec=soft_deadline,
                            request_started_sec=request_started_sec,
                        )
                    return _consume_sse(
                        response,
                        soft_deadline_sec=soft_deadline,
                        audit_tap=audit_tap,
                        request_started_sec=request_started_sec,
                    )
                raw = response.read().decode("utf-8", errors="replace")
                try:
                    from ..attribution import safe_observer_record

                    safe_observer_record(
                        self.attribution_observer,
                        {
                            "raw_body": raw,
                            "receipt_monotonic_sec": time.perf_counter(),
                            "request_id": str(
                                getattr(self, "_active_audit_request", {}).get(
                                    "request_id", "unknown"
                                )
                            ),
                            "sequence_id": 0,
                            "stage": "transport_body",
                        },
                    )
                except Exception:
                    pass
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value = {"text": raw}
                return value if isinstance(value, Mapping) else {"text": str(value)}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("vLLM request failed with HTTP %s: %s" % (exc.code, detail)) from exc
        except TimeoutError as exc:
            raise TimeoutError(
                "vLLM request timed out after %.3fs" % float(self.timeout_sec)
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("failed to connect to vLLM API: %s" % (exc.reason,)) from exc


__all__ = [
    "DEFAULT_TRANSCRIPTION_PROMPT",
    "MossWindowDecoder",
    "ParsedTranscriptSegment",
    "parse_transcript",
]
