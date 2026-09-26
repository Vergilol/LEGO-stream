"""Causal embedding access for streaming speaker assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.identity.identity_evidence import EmbeddingViewSet
from ..core.types import Segment
from .extractor import _as_processor_waveform, _load_audio_waveform, _load_processor


_PROCESSOR_CACHE: dict[tuple[object, ...], Any] = {}


def _shared_processor(
    *,
    speakerlab_root: Path,
    model_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    chunk_duration_sec: float,
    chunk_step_sec: float,
    checkpoint_name: str,
) -> Any:
    key = (
        str(speakerlab_root),
        str(model_dir),
        str(model_name),
        str(device),
        int(batch_size),
        float(chunk_duration_sec),
        float(chunk_step_sec),
        str(checkpoint_name),
    )
    processor = _PROCESSOR_CACHE.get(key)
    if processor is None:
        processor_type = _load_processor(speakerlab_root)
        processor = processor_type(
            model_name=model_name,
            sv_ckpt_dir=str(model_dir),
            device=device,
            batch_size=int(batch_size),
            chunk_duration=float(chunk_duration_sec),
            chunk_step=float(chunk_step_sec),
            checkpoint_name=str(checkpoint_name),
        )
        _PROCESSOR_CACHE[key] = processor
    return processor


@dataclass
class CausalSegmentStore:
    """Causal segment-level embedding access without dense overlap windows."""

    audio_path: Path
    speakerlab_root: Path
    model_dir: Path
    model_name: str = "eres2netv2"
    device: str = "cuda:0"
    batch_size: int = 64
    chunk_duration_sec: float = 1.5
    chunk_step_sec: float = 0.75
    checkpoint_name: str = "pretrained_eres2netv2w24s4ep4.ckpt"
    sample_rate: int = 16000
    configured_duration_sec: float | None = None
    audio_source: Any = None
    _waveform: Any = field(init=False, repr=False)
    _processor: Any = field(init=False, repr=False)
    _cache: dict[tuple[float, float], np.ndarray] = field(
        init=False, repr=False, default_factory=dict
    )
    _embedding_calls: int = field(init=False, repr=False, default=0)
    _embedding_available: int = field(init=False, repr=False, default=0)
    _view_cache: dict[tuple[object, ...], EmbeddingViewSet] = field(
        init=False, repr=False, default_factory=dict
    )
    _view_embedding_calls: int = field(init=False, repr=False, default=0)
    _view_embedding_available: int = field(init=False, repr=False, default=0)
    _audio_duration_sec: float = field(init=False, repr=False, default=0.0)
    _embedding_input_sequence: list[dict[str, object]] = field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        audio_path = Path(self.audio_path).expanduser().resolve()
        speakerlab_root = Path(self.speakerlab_root).expanduser().resolve()
        model_dir = Path(self.model_dir).expanduser().resolve()
        if self.audio_source is None and not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        checkpoint = model_dir / self.checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        if self.audio_source is None:
            waveform, sample_rate, source_duration_sec, _loader = _load_audio_waveform(
                audio_path,
                target_sample_rate=self.sample_rate,
            )
            if self.configured_duration_sec is not None:
                configured_duration_sec = float(self.configured_duration_sec)
                if not math.isfinite(configured_duration_sec) or configured_duration_sec <= 0.0:
                    raise ValueError("configured_duration_sec must be finite and positive")
                if configured_duration_sec > source_duration_sec + 1e-6:
                    raise ValueError(
                        "configured_duration_sec exceeds the decoded audio duration: "
                        f"{configured_duration_sec} > {source_duration_sec}"
                    )
                waveform = waveform[:, : int(round(configured_duration_sec * sample_rate))]
        else:
            sample_rate = int(getattr(self.audio_source, "sample_rate", 0))
            if sample_rate != int(self.sample_rate):
                raise ValueError(
                    "online embedding source sample rate does not match the store"
                )
            waveform = None
        processor = _shared_processor(
            speakerlab_root=speakerlab_root,
            model_dir=model_dir,
            model_name=self.model_name,
            device=self.device,
            batch_size=int(self.batch_size),
            chunk_duration_sec=float(self.chunk_duration_sec),
            chunk_step_sec=float(self.chunk_step_sec),
            checkpoint_name=self.checkpoint_name,
        )
        # Load model weights before the first streaming tick.  The processor
        # itself is cached, but its model is lazy-loaded by the 3D-Speaker
        # adapter; deferring this to the first committed segment would charge
        # model startup to an online tick and create a false RTF spike.
        _ = processor.embedding_model
        object.__setattr__(self, "audio_path", audio_path)
        object.__setattr__(self, "speakerlab_root", speakerlab_root)
        object.__setattr__(self, "model_dir", model_dir)
        object.__setattr__(self, "_waveform", waveform)
        object.__setattr__(self, "_processor", processor)
        object.__setattr__(
            self,
            "_audio_duration_sec",
            (
                float(waveform.shape[1]) / float(sample_rate)
                if waveform is not None
                else float(getattr(self.audio_source, "available_until_sec", 0.0))
            ),
        )

    @property
    def embedding_call_count(self) -> int:
        return self._embedding_calls

    @property
    def embedding_available_count(self) -> int:
        return self._embedding_available

    @property
    def embedding_cache_size(self) -> int:
        return len(self._cache)

    @property
    def view_embedding_call_count(self) -> int:
        return self._view_embedding_calls

    @property
    def view_embedding_available_count(self) -> int:
        return self._view_embedding_available

    @property
    def view_embedding_cache_size(self) -> int:
        return len(self._view_cache)

    def summary(self) -> dict[str, object]:
        sequence = getattr(self, "_embedding_input_sequence", [])
        sequence_payload = json.dumps(
            sequence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "embedding_mode": "committed_segment",
            "audio_path": str(self.audio_path),
            "speakerlab_root": str(self.speakerlab_root),
            "model_dir": str(self.model_dir),
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": int(self.batch_size),
            "chunk_duration_sec": float(self.chunk_duration_sec),
            "chunk_step_sec": float(self.chunk_step_sec),
            "audio_duration_sec": float(
                getattr(
                    self.audio_source,
                    "available_until_sec",
                    self._audio_duration_sec,
                )
            ),
            "embedding_call_count": self.embedding_call_count,
            "embedding_available_count": self.embedding_available_count,
            "embedding_cache_size": self.embedding_cache_size,
            "view_embedding_call_count": self.view_embedding_call_count,
            "view_embedding_available_count": self.view_embedding_available_count,
            "view_embedding_cache_size": self.view_embedding_cache_size,
            "embedding_input_sequence_count": len(
                sequence
            ),
            "embedding_input_sequence_sha256": hashlib.sha256(
                sequence_payload
            ).hexdigest(),
        }

    def _view_intervals(self, segment: Segment) -> tuple[tuple[float, float], ...]:
        start = float(segment.start)
        end = float(segment.end)
        duration = end - start
        native = float(self.chunk_duration_sec)
        if duration <= 0.0:
            raise ValueError("embedding view segment must have positive duration")
        if duration < native - 1e-9:
            return ((start, end),)
        count = min(4, max(1, int(math.floor(duration / native + 1e-9))))
        latest_start = end - native
        starts = np.linspace(start, latest_start, num=count, dtype=np.float64)
        return tuple(
            (float(value), float(value + native))
            for value in starts.tolist()
        )

    def _extract_intervals(self, intervals: list[tuple[float, float]]) -> Any:
        if self.audio_source is None:
            return self._processor.extract(self._waveform, intervals)
        base = min(start for start, _end in intervals)
        end = max(end for _start, end in intervals)
        mono = np.asarray(
            self.audio_source.read(base, end),
            dtype=np.float32,
        )
        waveform = _as_processor_waveform(
            np.ascontiguousarray(mono[None, :], dtype=np.float32)
        )
        local_intervals = [
            (float(start - base), float(end - base))
            for start, end in intervals
        ]
        return self._processor.extract(waveform, local_intervals)

    def embedding_views(
        self,
        segment: Segment,
        available_until: float,
        *,
        source_key: str,
    ) -> Optional[EmbeddingViewSet]:
        """Return at most four equal-duration causal views for one observation."""

        cutoff = float(available_until)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("available_until must be finite and non-negative")
        if float(segment.end) > cutoff + 1e-9:
            raise ValueError("segment end exceeds available_until")
        intervals = self._view_intervals(segment)
        canonical_intervals = tuple(
            (round(start, 6), round(end, 6))
            for start, end in intervals
        )
        cache_key: tuple[object, ...] = (str(source_key), canonical_intervals)
        cached = self._view_cache.get(cache_key)
        if cached is not None:
            return cached
        self._view_embedding_calls += len(intervals)
        sequence = getattr(self, "_embedding_input_sequence", None)
        if sequence is None:
            sequence = []
            object.__setattr__(self, "_embedding_input_sequence", sequence)
        sequence.append(
            {
                "kind": "views",
                "source_key": str(source_key),
                "intervals": [list(item) for item in canonical_intervals],
            }
        )
        result = self._extract_intervals(list(intervals))
        embeddings = np.asarray(result.embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(intervals):
            return None
        normalized: list[np.ndarray] = []
        for vector in embeddings:
            value = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(value))
            if value.ndim != 1 or norm <= 1e-12 or not np.all(np.isfinite(value)):
                return None
            normalized.append(value / norm)
        self._view_embedding_available += len(normalized)
        view_set = EmbeddingViewSet.create(
            source_key=str(source_key),
            views=normalized,
            intervals=intervals,
            native_chunk_duration_sec=float(self.chunk_duration_sec),
            circle_padded=(
                float(segment.duration) + 1e-9 < float(self.chunk_duration_sec)
            ),
        )
        self._view_cache[cache_key] = view_set
        return view_set

    def embedding(self, segment: Segment, available_until: float) -> Optional[np.ndarray]:
        cutoff = float(available_until)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("available_until must be finite and non-negative")
        if float(segment.end) > cutoff + 1e-9:
            raise ValueError("segment end exceeds available_until")
        interval = (float(segment.start), float(segment.end))
        key = (round(interval[0], 6), round(interval[1], 6))
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        self._embedding_calls += 1
        sequence = getattr(self, "_embedding_input_sequence", None)
        if sequence is None:
            sequence = []
            object.__setattr__(self, "_embedding_input_sequence", sequence)
        sequence.append(
            {
                "kind": "embedding",
                "intervals": [[key[0], key[1]]],
            }
        )
        result = self._extract_intervals([interval])
        embeddings = np.asarray(result.embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != 1:
            return None
        vector = np.asarray(embeddings[0], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or norm <= 1e-12 or not np.all(np.isfinite(vector)):
            return None
        normalized = vector / norm
        self._embedding_available += 1
        self._cache[key] = normalized
        return normalized.copy()


__all__ = ["CausalSegmentStore"]
