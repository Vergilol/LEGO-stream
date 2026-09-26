from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ..core.audio_io import load_audio_frames


def _load_processor(speakerlab_root: Path) -> type[_LocalSpeakerEmbeddingProcessor]:
    """Return a processor class bound to a 3D-Speaker checkout.

    ``speakerlab_root`` must contain the ``speakerlab`` package -- that is, a
    clone of https://github.com/modelscope/3D-Speaker. The speaker encoder is
    built directly from that package against a local checkpoint; no diarization
    entrypoint is imported, which keeps the optional pyannote and ModelScope
    stacks out of the streaming path.
    """

    speakerlab_root = Path(speakerlab_root).expanduser().resolve()
    required = speakerlab_root / "speakerlab"
    if not required.is_dir():
        raise FileNotFoundError(
            "speakerlab package not found under "
            f"{speakerlab_root}; pass --speakerlab-root pointing at a "
            "3D-Speaker checkout"
        )
    loaded = sys.modules.get("speakerlab")
    loaded_paths = tuple(getattr(loaded, "__path__", ())) if loaded is not None else ()
    if loaded_paths and not any(
        _is_relative_to(Path(item).resolve(), required) for item in loaded_paths
    ):
        raise RuntimeError(
            "speakerlab is already imported from a different checkout: "
            + ", ".join(str(item) for item in loaded_paths)
        )
    return _processor_type_for(speakerlab_root)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class _EmbeddingResult:
    embeddings: np.ndarray
    chunks: list[tuple[float, float]]
    model_name: str = "eres2netv2"
    embedding_dim: int = 192


def _speaker_model_config(model_name: str) -> dict[str, Any]:
    if model_name == "eres2netv2":
        return {
            "embedding_model": {
                "obj": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
                "args": {
                    "feat_dim": 80,
                    "embedding_size": 192,
                    "m_channels": 64,
                    "baseWidth": 24,
                    "scale": 4,
                    "expansion": 4,
                },
            },
            "feature_extractor": {
                "obj": "speakerlab.process.processor.FBank",
                "args": {
                    "n_mels": 80,
                    "sample_rate": 16000,
                    "mean_nor": True,
                },
            },
        }
    if model_name == "campplus":
        return {
            "embedding_model": {
                "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
                "args": {
                    "feat_dim": 80,
                    "embedding_size": 192,
                },
            },
            "feature_extractor": {
                "obj": "speakerlab.process.processor.FBank",
                "args": {
                    "n_mels": 80,
                    "sample_rate": 16000,
                    "mean_nor": True,
                },
            },
        }
    raise ValueError(f"unsupported speaker embedding model: {model_name}")


def _load_state_dict(checkpoint: Path) -> dict[str, Any]:
    import torch

    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint does not contain a state dict: {checkpoint}")
    return {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
    }


def _circle_pad(waveform: Any, target_len: int) -> Any:
    length = int(waveform.shape[0])
    if length <= 0:
        raise ValueError("cannot pad an empty waveform")
    if length >= target_len:
        return waveform
    import torch

    repeats = int(math.ceil(target_len / length))
    return torch.narrow(
        torch.cat([waveform] * repeats, dim=0),
        0,
        0,
        int(target_len),
    )


class _LocalSpeakerEmbeddingProcessor:
    """Minimal local ERes2Net/CAM++ processor without diarization imports."""

    _speakerlab_root: Path

    def __init__(
        self,
        model_name: str = "eres2netv2",
        sv_ckpt_dir: str | None = None,
        sv_exp_dir: str | None = None,
        sv_embedding_size: int | None = None,
        model_cache_dir: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        chunk_duration: float = 1.5,
        chunk_step: float = 0.75,
        checkpoint_name: str | None = None,
    ) -> None:
        del sv_exp_dir, model_cache_dir
        self.model_name = str(model_name)
        self.sv_ckpt_dir = sv_ckpt_dir
        self.sv_embedding_size = sv_embedding_size
        self.device = device or "cpu"
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.chunk_duration = float(chunk_duration)
        self.chunk_step = float(chunk_step)
        self.checkpoint_name = checkpoint_name
        self._embedding_model: Any = None
        self._feature_extractor: Any = None
        self.sample_rate = 16000

    def _checkpoint_path(self) -> Path:
        if not self.sv_ckpt_dir:
            raise FileNotFoundError("a speaker embedding checkpoint is required")
        path = Path(self.sv_ckpt_dir).expanduser().resolve()
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(path)
        names = (
            self.checkpoint_name,
            "pretrained_eres2netv2w24s4ep4.ckpt",
            "embedding_model.ckpt",
        )
        for name in names:
            if name and (path / name).is_file():
                return path / name
        raise FileNotFoundError(
            f"no supported checkpoint found under {path}; tried {tuple(n for n in names if n)}"
        )

    def _load_model(self) -> tuple[Any, Any]:
        if self._embedding_model is not None:
            return self._embedding_model, self._feature_extractor

        speakerlab_root = Path(self._speakerlab_root).resolve()
        if not speakerlab_root.is_dir():
            raise FileNotFoundError(f"3D-Speaker repository is missing: {speakerlab_root}")
        root_text = str(speakerlab_root)
        sys.path[:] = [item for item in sys.path if item != root_text]
        sys.path.insert(0, root_text)

        import torch
        from speakerlab.utils.builder import build
        from speakerlab.utils.config import Config

        config_dict = _speaker_model_config(self.model_name)
        if self.sv_embedding_size is not None:
            config_dict["embedding_model"]["args"]["embedding_size"] = int(
                self.sv_embedding_size
            )
        config = Config(config_dict)
        feature_extractor = build("feature_extractor", config)
        embedding_model = build("embedding_model", config)
        embedding_model.load_state_dict(
            _load_state_dict(self._checkpoint_path()),
            strict=False,
        )
        embedding_model.eval()
        embedding_model.to(torch.device(self.device))
        self._embedding_model = embedding_model
        self._feature_extractor = feature_extractor
        return embedding_model, feature_extractor

    @property
    def embedding_model(self) -> Any:
        self._load_model()
        return self._embedding_model

    @property
    def feature_extractor(self) -> Any:
        self._load_model()
        return self._feature_extractor

    def extract(
        self,
        wav_data: Any,
        chunks: list[tuple[float, float]],
    ) -> _EmbeddingResult:
        import torch

        if not chunks:
            return _EmbeddingResult(
                embeddings=np.empty((0, 192), dtype=np.float32),
                chunks=[],
                model_name=self.model_name,
            )
        waveform = torch.as_tensor(wav_data, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2 or int(waveform.shape[0]) == 0:
            raise ValueError("wav_data must have shape [channels, samples]")
        waveform = waveform[:1].cpu()
        intervals = [(float(start), float(end)) for start, end in chunks]
        if any(
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end <= start
            for start, end in intervals
        ):
            raise ValueError("embedding chunks must be finite positive intervals")
        wavs = []
        for start, end in intervals:
            start_frame = int(start * self.sample_rate)
            end_frame = int(end * self.sample_rate)
            wavs.append(waveform[0, start_frame:end_frame])
        if any(int(item.shape[0]) == 0 for item in wavs):
            raise ValueError("embedding chunks must contain audio")
        # The pad floor is ONE NATIVE WINDOW, read from `self.chunk_duration` so a
        # config that moves `embedding.chunk_duration_sec` off 1.5 stays in step.
        max_len = max(
            max(int(item.shape[0]) for item in wavs),
            int(self.chunk_duration * self.sample_rate),
        )
        batches = [
            torch.stack(
                [
                    _circle_pad(item, max_len)
                    for item in wavs[index : index + self.batch_size]
                ]
            )
            .unsqueeze(1)
            for index in range(0, len(wavs), self.batch_size)
        ]
        embeddings: list[Any] = []
        with torch.no_grad():
            for batch in batches:
                features = torch.stack(
                    [self.feature_extractor(item) for item in batch],
                    dim=0,
                ).to(self.device)
                embeddings.append(self.embedding_model(features).detach().cpu())
        matrix = torch.cat(embeddings, dim=0).numpy().astype(np.float32, copy=False)
        return _EmbeddingResult(
            embeddings=matrix,
            chunks=list(chunks),
            model_name=self.model_name,
            embedding_dim=(
                int(matrix.shape[1])
                if matrix.ndim == 2 and matrix.shape[0]
                else 192
            ),
        )


_PROCESSOR_TYPES: dict[Path, type[_LocalSpeakerEmbeddingProcessor]] = {}


def _processor_type_for(speakerlab_root: Path) -> type[_LocalSpeakerEmbeddingProcessor]:
    root = Path(speakerlab_root).expanduser().resolve()
    cached = _PROCESSOR_TYPES.get(root)
    if cached is not None:
        return cached
    processor_type = type(
        "SpeakerEmbeddingProcessor",
        (_LocalSpeakerEmbeddingProcessor,),
        {"__module__": __name__, "_speakerlab_root": root},
    )
    _PROCESSOR_TYPES[root] = processor_type
    return processor_type


def _mean_mono(waveform: Any) -> Any:
    if getattr(waveform, "ndim", None) != 2:
        raise RuntimeError("audio loader returned a waveform with invalid dimensions")
    if int(waveform.shape[1]) == 0:
        raise RuntimeError("audio loader returned an empty waveform")
    if int(waveform.shape[0]) <= 1:
        return waveform
    # Unit fixtures hand this a numpy array rather than a tensor.
    if isinstance(waveform, np.ndarray):
        return waveform.mean(axis=0, keepdims=True)
    return waveform.mean(dim=0, keepdim=True)


def _resample_numpy(
    waveform: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate == target_sample_rate:
        return np.asarray(waveform, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "audio decoding needs scipy.signal.resample_poly when "
            "sample_rate != 16000"
        ) from exc
    divisor = math.gcd(int(source_sample_rate), int(target_sample_rate))
    resampled = resample_poly(
        np.asarray(waveform, dtype=np.float32),
        int(target_sample_rate) // divisor,
        int(source_sample_rate) // divisor,
        axis=1,
    )
    return np.asarray(resampled, dtype=np.float32)


def _as_processor_waveform(waveform: np.ndarray) -> Any:
    try:
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    except (ImportError, AttributeError):
        # A numpy-only fake torch module: the unit fixtures supply one.
        pass
    return waveform


def _load_audio_waveform(
    audio_path: Path,
    *,
    target_sample_rate: int = 16000,
) -> tuple[Any, int, float, str]:
    frames, source_sample_rate, loader = load_audio_frames(audio_path)
    source_sample_rate = int(source_sample_rate)
    if frames.ndim != 2 or frames.shape[0] == 0:
        raise RuntimeError("audio decoder returned an empty waveform")
    if source_sample_rate <= 0:
        raise RuntimeError("audio decoder returned an invalid sample rate")
    waveform = _mean_mono(np.ascontiguousarray(frames.T, dtype=np.float32))
    source_duration_sec = float(waveform.shape[1]) / source_sample_rate
    waveform = _resample_numpy(
        waveform,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
    )
    return (
        _as_processor_waveform(waveform),
        target_sample_rate,
        source_duration_sec,
        loader,
    )


__all__ = ["_load_audio_waveform", "_load_processor"]
