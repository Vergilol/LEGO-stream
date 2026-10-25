"""Drive one live session in-process.

A live source has no file and no known duration, so :class:`LiveSession` owns
an :class:`~lego_stream.core.online_audio.OnlinePcmBuffer` that the decoder and
the embedding store read instead, a
:class:`~lego_stream.StreamingSpeakerASRSession` over it, and a worker thread
that ticks as the audio passes each step. The audio is never written to disk.

The session ends by ``seal()`` (the client sent ``eos``), ``cancel()``, or a
failure in the worker. Only a seal keeps anything: the transcript.
"""

from __future__ import annotations

import math
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from lego_stream.core.online_audio import OnlinePcmBuffer

from .live_protocol import (
    DEFAULT_MAX_BACKLOG_SEC,
    SAMPLE_RATE,
    decode_pcm16,
    error_message,
)
from .frames import provisional_lanes, read_committed
from .metrics import PerformanceTracker

#: Slack on every "is the audio there yet" comparison.
_EPSILON = 1e-9
#: How long the worker sleeps before re-checking, in case a wakeup is missed.
_WAKE_INTERVAL_SEC = 0.1
#: The one file a finished session keeps.
_TRANSCRIPT = "hypothesis.json"
#: Kept beside it only when the run failed validation: the page points there.
_VALIDATION = "stream_validation.json"

#: The worker's last frame, so the route stops reading without polling.
CLOSED_FRAME: dict[str, Any] = {"type": "closed"}


class LiveSessionError(RuntimeError):
    """A live session could not start, or had to be aborted."""


def _positive_seconds(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LiveSessionError(f"{what} must be a number") from None
    if not math.isfinite(number) or number <= 0.0:
        raise LiveSessionError(f"{what} must be finite and positive")
    return number


def _non_negative_seconds(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LiveSessionError(f"{what} must be a number") from None
    if not math.isfinite(number) or number < 0.0:
        raise LiveSessionError(f"{what} must be finite and non-negative")
    return number


def _streaming_session(config: Mapping[str, Any], output_dir: Path, *, audio_source: Any) -> Any:
    """Build the real session; imports the model stack, so it is deferred."""

    from lego_stream import StreamingSpeakerASRSession

    return StreamingSpeakerASRSession(
        config, output_dir, audio_source=audio_source
    )


class LiveSession:
    """One live session: buffer, model session, worker thread.

    ``config`` comes from :func:`demo.server.build_config`. ``output_dir`` must
    be empty or absent: everything in it but the transcript is removed.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        output_dir: str | Path,
        *,
        emit: Callable[[dict[str, Any]], None] | None = None,
        session_factory: Callable[..., Any] | None = None,
        max_backlog_sec: float = DEFAULT_MAX_BACKLOG_SEC,
        metrics: PerformanceTracker | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        stream = _mapping_block(config, "stream")
        self.step_sec = _positive_seconds(stream.get("step_sec"), "stream.step_sec")
        context_sec = _non_negative_seconds(
            stream.get("context_sec"), "stream.context_sec"
        )
        right_sec = _non_negative_seconds(
            stream.get("right_context_sec", 0.0), "stream.right_context_sec"
        )
        self.max_backlog_sec = _positive_seconds(max_backlog_sec, "max_backlog_sec")

        # A tick reads a whole context window back, so keep that plus the
        # allowed backlog; ``_require_progress`` fails before audio a tick
        # still needs is evicted. The 2 s is slack.
        retention_sec = (
            context_sec + self.max_backlog_sec + right_sec + self.step_sec + 2.0
        )
        self.sample_rate = SAMPLE_RATE
        self.metrics = metrics if metrics is not None else PerformanceTracker()
        self._buffer = OnlinePcmBuffer(
            sample_rate=self.sample_rate, retention_sec=retention_sec
        )

        self._output_dir = Path(output_dir).expanduser().resolve()
        if self._output_dir.is_dir() and any(self._output_dir.iterdir()):
            raise LiveSessionError(
                f"{self._output_dir} must be empty: a live session removes "
                "everything but its transcript when it ends, so give each one "
                "a directory of its own"
            )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._emit = emit
        # Frees the server's slot; called once the last tick has returned.
        self._on_finish = on_finish
        self._messages: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)

        self._sequence = 0
        self._last_tick = 0.0
        self._published: list[dict[str, Any]] = []
        self._sealed = False
        self._cancelled = False
        self._cancel_reason: str | None = None
        self._finished = False
        self._failure: BaseException | None = None

        self._session = (session_factory or _streaming_session)(
            config, self._output_dir, audio_source=self._buffer
        )
        self._worker: threading.Thread | None = threading.Thread(
            target=self._drive,
            name="lego-stream-live",
            daemon=True,  # never holds the server up on exit
        )
        self._worker.start()

    # -- producer side, called from the event loop -------------------------

    def append(self, payload: bytes) -> float:
        """Take one binary frame of PCM16 and return the audio held so far."""

        samples = decode_pcm16(payload, expected_bytes=len(payload))
        with self._lock:
            if self._failure is not None:
                raise LiveSessionError(error_message(self._failure))
            if self._sealed or self._cancelled:
                raise LiveSessionError("the live session is already finishing")
            # The buffer rejects a gap in the sequence.
            self._buffer.append(samples, sequence=self._sequence)
            self._sequence += 1
            available = self._buffer.available_until_sec
            self._wake.notify_all()
        return available

    def seal(self) -> None:
        """Close the audio stream normally; the worker flushes and finalizes."""

        with self._lock:
            if self._finished:
                return
            self._sealed = True
            self._wake.notify_all()

    def cancel(self, reason: str | None = None) -> None:
        """Abandon the session; the worker records the failure and stops."""

        with self._lock:
            if self._finished:
                return
            self._cancelled = True
            self._cancel_reason = (
                reason or "the live microphone session was canceled"
            )
            self._wake.notify_all()

    def close(self, *, timeout: float | None = 5.0) -> None:
        """Cancel if the session is still running and join the worker."""

        with self._lock:
            unfinished = not self._finished
        if unfinished:
            self.cancel()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout)

    def drain_messages(self) -> list[dict[str, Any]]:
        """Take the frames emitted so far (only when no ``emit`` was given)."""

        with self._lock:
            messages, self._messages = self._messages, []
            return messages

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    # -- the worker thread --------------------------------------------------

    def _drive(self) -> None:
        try:
            self._stream()
        except BaseException as exc:  # reported to the page
            self._abort(exc)
        finally:
            try:
                # A run that did not finish keeps nothing on disk.
                if self._failure is not None:
                    shutil.rmtree(self._output_dir, ignore_errors=True)
            finally:
                try:
                    # Before the sentinel, so the slot is free once it arrives.
                    if self._on_finish is not None:
                        self._on_finish()
                finally:
                    # Sentinel first: ``finished`` implies it was queued.
                    self._emit_frame(dict(CLOSED_FRAME))
                    with self._lock:
                        self._finished = True
                        self._wake.notify_all()

    def _stream(self) -> None:
        step = self.step_sec
        next_tick = step
        while True:
            with self._lock:
                # Mirrors the loop below, so a wakeup always makes progress.
                while (
                    not self._sealed
                    and not self._cancelled
                    and self._buffer.available_until_sec <= next_tick + _EPSILON
                ):
                    self._wake.wait(_WAKE_INTERVAL_SEC)
                sealed = self._sealed
                cancelled = self._cancelled
                reason = self._cancel_reason
            if cancelled:
                raise LiveSessionError(
                    reason or "the live microphone session was canceled"
                )
            if sealed:
                self._seal(next_tick)
                return
            self._require_progress()
            while next_tick < self._buffer.available_until_sec - _EPSILON:
                self._process(next_tick, final=False)
                next_tick += step

    def _require_progress(self) -> None:
        """Fail loudly rather than fall further behind than the buffer retains."""

        lag = self._buffer.available_until_sec - self._last_tick
        if lag > self.max_backlog_sec:
            raise LiveSessionError(
                f"the live streamer fell {lag:.1f}s behind the microphone "
                f"(limit {self.max_backlog_sec:g}s); the session was stopped "
                "instead of silently dropping audio"
            )

    def _seal(self, next_tick: float) -> None:
        duration = self._buffer.available_until_sec
        if duration <= _EPSILON:
            raise LiveSessionError(
                "the live session ended before any audio arrived"
            )
        # Ticks still due before the end, if the seal landed mid-step.
        while next_tick < duration - _EPSILON:
            self._process(next_tick, final=False)
            next_tick += self.step_sec
        # ``final=True`` drops the right-context hold, so the tail publishes.
        if duration > self._last_tick + _EPSILON:
            self._process(duration, final=True)

        result = self._session.finalize()
        stream_result = result.get("stream") if isinstance(result, dict) else None
        stream_result = stream_result if isinstance(stream_result, dict) else {}
        # Read before the clean-up below.
        committed = read_committed(stream_result)
        transcript = self._keep_only_the_transcript(stream_result.get("status"))
        self._emit_frame(
            {
                "type": "done",
                # Release order: exactly what the page was sent.
                "segments": list(self._published),
                # Read back from ``hypothesis.json``, in timeline order.
                "committed": committed,
                "validation_status": stream_result.get("status"),
                "committed_segments": stream_result.get("committed_segments"),
                "speaker_count": stream_result.get("speaker_count"),
                "duration_sec": float(duration),
                # Its path on disk, or ``None``.
                "transcript": transcript,
                "metrics": self.metrics.snapshot(),
            }
        )

    def _process(self, tick: float, *, final: bool) -> None:
        """One tick. Worker-thread only; nothing here takes the lock."""

        self.metrics.start_tick(tick)
        row = self._session.process_tick(tick, final=final)
        published = row.get("delta_segments") or []
        if isinstance(published, list):
            self._published.extend(published)
        else:
            published = []
        snapshot = self.metrics.finish_tick(tick, publications=published)
        self._last_tick = tick
        self._emit_frame(
            {
                "type": "tick",
                "tick": float(tick),
                "final": bool(final),
                "appended": published,
                "pending": row.get("pending_segments") or [],
                "provisional": provisional_lanes(row, self._published),
                "published_count": len(self._published),
                "metrics": snapshot,
            }
        )

    def _abort(self, exc: BaseException) -> None:
        # Best effort: failing to record it must not mask the failure itself.
        try:
            self._session.fail(exc)
        except Exception:  # pragma: no cover - best effort, then report anyway
            pass
        with self._lock:
            self._failure = exc
        self._emit_frame(
            {"type": "error", "message": error_message(exc)}
        )

    def _emit_frame(self, message: dict[str, Any]) -> None:
        if self._emit is None:
            with self._lock:
                self._messages.append(message)
            return
        try:
            self._emit(message)
        except Exception:
            # The event loop is gone at shutdown; nobody is left to tell.
            pass

    # -- what stays on disk -------------------------------------------------

    def _keep_only_the_transcript(self, status: Any) -> str | None:
        """Remove everything the run wrote but its transcript; return its path.

        The validation report stays too if the run failed validation, since
        the page then points to it.
        """

        keep = {_TRANSCRIPT} if status == "passed" else {_TRANSCRIPT, _VALIDATION}
        try:
            entries = list(self._output_dir.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.name in keep and entry.is_file():
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        transcript = self._output_dir / _TRANSCRIPT
        if transcript.is_file():
            return str(transcript)
        # Nothing worth keeping, so not even the directory.
        shutil.rmtree(self._output_dir, ignore_errors=True)
        return None


def _mapping_block(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    block = config.get(key) if isinstance(config, Mapping) else None
    if not isinstance(block, Mapping):
        raise LiveSessionError(f"a live config needs a {key} block")
    return block


__all__ = ["CLOSED_FRAME", "LiveSession", "LiveSessionError"]
