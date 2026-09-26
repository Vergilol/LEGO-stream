"""Audio decoding shared by streaming ASR and global speaker embeddings."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


def _read_pcm_wav(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frame_count = int(handle.getnframes())
        compression = handle.getcomptype()
        raw = handle.readframes(frame_count)
    if channels <= 0 or sample_rate <= 0 or compression != "NONE":
        raise ValueError("unsupported WAV metadata")
    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        values = (values - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")
    if values.size != frame_count * channels:
        raise ValueError("WAV frame payload is truncated")
    return values.reshape(frame_count, channels), sample_rate


def load_audio_frames(audio_path: Path) -> tuple[np.ndarray, int, str]:
    """Return float32 audio as ``[frames, channels]`` and the decoder name."""

    path = Path(audio_path)
    try:
        values, sample_rate = _read_pcm_wav(path)
        return values, sample_rate, "wave"
    except (OSError, EOFError, ValueError, wave.Error):
        pass

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        if isinstance(waveform, np.ndarray):
            values = np.asarray(waveform, dtype=np.float32).T
        else:
            values = waveform.detach().cpu().numpy().T.astype(
                np.float32, copy=False
            )
        return values, int(sample_rate), "torchaudio"
    except Exception as torchaudio_error:
        try:
            import soundfile as sf

            values, sample_rate = sf.read(
                str(path), always_2d=True, dtype="float32"
            )
            return np.asarray(values, dtype=np.float32), int(sample_rate), "soundfile"
        except Exception as soundfile_error:
            raise RuntimeError(
                "audio read failed with WAV, torchaudio, and soundfile; "
                f"torchaudio_error={torchaudio_error!r}; "
                f"soundfile_error={soundfile_error!r}"
            ) from soundfile_error


__all__ = ["load_audio_frames"]
