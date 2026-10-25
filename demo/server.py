"""Local demo: streaming, append-only speaker-attributed transcription.

The page streams 16 kHz mono PCM16 over ``WS /api/live`` and
:class:`demo.live.LiveSession` ticks through it in real time. No audio is kept;
a finished session leaves only its ``hypothesis.json`` in a temp directory.

    python -m demo.server --speaker-model DIR --speakerlab-root DIR

Binds to localhost, runs one session at a time, and has no authentication: do
not expose it to a network as-is.
"""

# No ``from __future__ import annotations``: FastAPI could not resolve a string
# ``WebSocket`` annotation, since it is imported inside ``create_app``.

import argparse
import asyncio
import collections
from contextlib import asynccontextmanager
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Callable, Collection

from .live import LiveSession
from .live_protocol import (
    CLOSE_POLICY_VIOLATION,
    DEFAULT_MAX_BACKLOG_SEC,
    DEFAULT_MAX_SESSION_SEC,
    LiveProtocolError,
    chunk_accepted,
    control_type,
    error_frame,
    parse_chunk_header,
    parse_start,
    session_started,
)

WEB_ROOT = Path(__file__).resolve().parent / "web"

BUSY_MESSAGE = (
    "another session is already running on this single GPU; wait for it to "
    "finish, or raise --max-concurrent if the machine can take it"
)
DEFAULT_KEEP_SESSIONS = 5

_SESSION_COUNTER = itertools.count(1)

_STOP = object()
#: So a stuck tick cannot hold a thread on the way out.
_LIVE_JOIN_TIMEOUT_SEC = 5.0


def build_config(
    name: str,
    *,
    endpoint: str,
    served_model_name: str,
    speaker_model: str,
    speakerlab_root: str,
    device: str,
    context_sec: float,
    step_sec: float,
    duration_sec: float = DEFAULT_MAX_SESSION_SEC,
) -> dict[str, Any]:
    """The config for one live session called ``name``."""

    # Deferred: it imports torch.
    from lego_stream.core.identity.speakers import LEGO_POLICY

    return {
        "endpoint": endpoint,
        "served_model_name": served_model_name,
        "sample": {
            "id": name,
            "audio_name": name,
            # A label; the session reads the live buffer.
            "audio_path": f"live:{name}",
            # A ceiling; the real length is known only at the end.
            "duration_sec": float(duration_sec),
        },
        "stream": {
            "step_sec": step_sec,
            "context_sec": context_sec,
            "right_context_sec": 2.0,
            "heuristic_overlap_min_sec": 0.2,
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
            "causal_frontier_request_ids": False,
        },
        "embedding": {
            "speakerlab_root": speakerlab_root,
            "model_dir": speaker_model,
            "checkpoint_name": "pretrained_eres2netv2w24s4ep4.ckpt",
            "device": device,
            "batch_size": 64,
            "chunk_duration_sec": 1.5,
            "chunk_step_sec": 0.75,
            "dimension": 192,
        },
        "association": {
            "mode": "causal_embedding",
            "policy": LEGO_POLICY,
            "min_overlap_sec": 0.1,
            "core_min_duration_sec": 1.5,
            "core_min_quality": 0.25,
            "best_effort_after_ticks": 1,
            "publish_pending_speakers": False,
            "centroid_correction_enabled": False,
            "centroid_correction_min_gain": 0.05,
            "centroid_correction_min_cosine": 0.5,
            "centroid_correction_min_duration_sec": 1.5,
        },
    }


def _new_session_id() -> str:
    """A fresh, sortable directory name for one session's files."""

    return f"session-{next(_SESSION_COUNTER):05d}"


def _prune_sessions(root: Path, keep: int, protected: Collection[str] = ()) -> list[str]:
    """Delete all but the newest ``keep`` session directories; best effort.

    ``protected`` (running) sessions count toward ``keep`` and are never
    removed. Returns the names removed.
    """

    if keep <= 0:
        return []
    shielded = set(protected)
    try:
        candidates = sorted(
            entry
            for entry in root.iterdir()
            if entry.is_dir() and entry.name not in shielded
        )
    except OSError:
        return []
    room = keep - len(shielded)
    stale = candidates[: max(0, len(candidates) - max(0, room))]
    for entry in stale:
        shutil.rmtree(entry, ignore_errors=True)
    return [entry.name for entry in stale]


def _once(action: Callable[[], None]) -> Callable[[], None]:
    """``action``, run at most once however many threads call it."""

    gate = threading.Lock()

    def run() -> None:
        if gate.acquire(blocking=False):
            action()

    return run


class _Sessions:
    """Which session directories are in use; released from worker threads."""

    def __init__(self, root: Path, keep: int) -> None:
        self.root = root
        self.keep = keep
        self._lock = threading.Lock()
        self._in_use: collections.Counter[str] = collections.Counter()

    def hold(self, names: Collection[str]) -> Callable[[], None]:
        """Mark ``names`` in use; the returned release is safe to call twice."""

        held = list(dict.fromkeys(names))
        with self._lock:
            self._in_use.update(held)

        def release() -> None:
            with self._lock:
                self._in_use.subtract(held)
                for name in held:
                    if self._in_use[name] <= 0:
                        del self._in_use[name]

        return _once(release)

    def prune(self, current: str) -> None:
        with self._lock:
            protected = {current, *self._in_use}
        _prune_sessions(self.root, self.keep, protected)


def create_app(
    defaults: argparse.Namespace,
    *,
    live_session_factory: Callable[..., Any] | None = None,
):
    """The FastAPI app. ``live_session_factory`` builds the model session."""

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    root = Path(tempfile.mkdtemp(prefix="lego-stream-demo-")).resolve()
    slots = threading.BoundedSemaphore(max(1, int(defaults.max_concurrent)))
    sessions = _Sessions(root, int(defaults.keep_sessions))

    @asynccontextmanager
    async def lifespan(_app: Any):
        try:
            yield
        finally:
            shutil.rmtree(root, ignore_errors=True)

    app = FastAPI(title="LEGO-Stream demo", lifespan=lifespan)
    # Serves ``pcm-worklet.js`` with the JavaScript type ``addModule`` requires.
    app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_ROOT / "index.html").read_text(encoding="utf-8")

    def _decode_control(payload: str) -> Any:
        try:
            return json.loads(payload)
        except ValueError:
            raise LiveProtocolError(
                "a live control message must be valid JSON"
            ) from None

    async def _receive(socket: Any) -> tuple[str, Any]:
        """One inbound frame as ``("text"|"bytes", payload)``; raises on disconnect."""

        event = await socket.receive()
        if event["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(event.get("code") or 1000)
        if event.get("bytes") is not None:
            return "bytes", event["bytes"]
        if event.get("text") is not None:
            return "text", event["text"]
        raise LiveProtocolError("the live socket received an empty frame")

    async def _forward(queue: asyncio.Queue, socket: Any) -> None:
        """The socket's only writer, so outbound frames cannot interleave."""

        while True:
            message = await queue.get()
            if message is _STOP:
                return
            # Internal; never sent to the page.
            if message.get("type") == "closed":
                return
            try:
                await socket.send_json(message)
            except Exception:
                # The browser left; ``finally`` cancels the session.
                return

    async def _settle(task: asyncio.Task) -> None:
        try:
            await task
        except Exception:
            pass

    async def _reject(socket: Any, exc: BaseException) -> None:
        """Say why, then close with a policy violation."""

        try:
            await socket.send_json(error_frame(exc))
        except Exception:
            return
        try:
            await socket.close(code=CLOSE_POLICY_VIOLATION)
        except Exception:
            pass

    async def _pump(session: LiveSession, socket: Any, queue: asyncio.Queue) -> None:
        """Read client control frames until ``eos`` or ``cancel``."""

        pending: tuple[int, int] | None = None
        expected = 0
        while True:
            kind, payload = await _receive(socket)
            if kind == "bytes":
                if pending is None:
                    raise LiveProtocolError(
                        "a binary frame arrived with no chunk header before it"
                    )
                sequence, byte_length = pending
                pending = None
                if len(payload) != byte_length:
                    raise LiveProtocolError(
                        f"chunk header announced {byte_length} bytes but the "
                        f"frame carried {len(payload)}"
                    )
                held = await asyncio.get_running_loop().run_in_executor(
                    None, session.append, payload
                )
                queue.put_nowait(
                    chunk_accepted(sequence=sequence, duration_sec=held)
                )
                continue

            message = _decode_control(payload)
            control = control_type(message)
            if control == "start":
                raise LiveProtocolError("the live session has already started")
            if control == "chunk":
                if pending is not None:
                    raise LiveProtocolError(
                        "a chunk header arrived before the previous chunk's audio"
                    )
                sequence, byte_length = parse_chunk_header(message)
                if sequence != expected:
                    raise LiveProtocolError(
                        f"chunk {sequence} arrived out of order; expected {expected}"
                    )
                expected += 1
                pending = (sequence, byte_length)
                continue
            if control == "eos":
                session.seal()
                return
            session.cancel("the client canceled the live session")
            return

    @app.websocket("/api/live")
    async def live(websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        session: LiveSession | None = None
        forwarder: asyncio.Task | None = None
        finish: Callable[[], None] | None = None

        def emit(message: dict[str, Any]) -> None:
            # From the worker thread.
            loop.call_soon_threadsafe(queue.put_nowait, message)

        try:
            kind, payload = await _receive(websocket)
            if kind != "text":
                raise LiveProtocolError(
                    "the first live message must be a start message"
                )
            step_sec = parse_start(
                _decode_control(payload), step_sec=defaults.step_sec
            )

            if not slots.acquire(blocking=False):
                await _reject(websocket, LiveProtocolError(BUSY_MESSAGE))
                return
            session_id = _new_session_id()
            output_dir = root / session_id
            # So no other session prunes it while it runs.
            release_dirs = sessions.hold([session_id])

            def release() -> None:
                release_dirs()
                slots.release()

            # Run by the worker after its last tick, or below if none was built.
            finish = _once(release)
            sessions.prune(session_id)
            config = build_config(
                session_id,
                endpoint=defaults.endpoint,
                served_model_name=defaults.served_model_name,
                speaker_model=str(Path(defaults.speaker_model).expanduser().resolve()),
                speakerlab_root=str(
                    Path(defaults.speakerlab_root).expanduser().resolve()
                ),
                device=defaults.device,
                context_sec=defaults.context_sec,
                step_sec=defaults.step_sec,
            )
            session = await loop.run_in_executor(
                None,
                lambda: LiveSession(
                    config,
                    output_dir,
                    emit=emit,
                    session_factory=live_session_factory,
                    max_backlog_sec=defaults.max_backlog_sec,
                    on_finish=finish,
                ),
            )
            forwarder = loop.create_task(_forward(queue, websocket))
            queue.put_nowait(session_started(step_sec=step_sec))

            await _pump(session, websocket, queue)
            # Until ``done``; the slot is free before the socket closes.
            await forwarder
            try:
                await websocket.close()
            except Exception:
                pass
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            queue.put_nowait(_STOP)
            if forwarder is not None:
                await _settle(forwarder)
            await _reject(websocket, exc)
        finally:
            if forwarder is not None:
                queue.put_nowait(_STOP)

            async def cleanup() -> None:
                if session is not None:
                    # Cancel and join; the worker frees the slot.
                    await loop.run_in_executor(
                        None, lambda: session.close(timeout=_LIVE_JOIN_TIMEOUT_SEC)
                    )
                elif finish is not None:
                    finish()
                if forwarder is not None:
                    await _settle(forwarder)

            # Shielded: a vanished client cancels this coroutine, and a dropped
            # cleanup would hold the slot until the server restarts.
            await asyncio.shield(cleanup())

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo.server")
    parser.add_argument("--speaker-model", required=True)
    parser.add_argument("--speakerlab-root", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19300/v1")
    parser.add_argument("--served-model-name", default="moss-transcribe-diarize")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=2.0)
    parser.add_argument(
        "--keep-sessions",
        type=int,
        default=DEFAULT_KEEP_SESSIONS,
        help="finished transcripts kept under the temp root; 0 keeps all",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="sessions allowed at once; the GPU is shared, not multiplied",
    )
    parser.add_argument(
        "--max-backlog-sec",
        type=float,
        default=DEFAULT_MAX_BACKLOG_SEC,
        help="fail the live session once the ticker falls this far behind the "
             "newest captured audio, instead of dropping samples",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="localhost by default; there is no authentication",
    )
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        print(
            f"WARNING: binding {args.host} exposes an unauthenticated demo "
            "that runs inference on whatever audio anyone streams to it.",
        )
    uvicorn.run(create_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
