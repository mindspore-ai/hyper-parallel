# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Audio decoding for the Qwen3-Omni data pipeline.

Decoding goes through PyAV; resampling uses ``scipy.signal.resample_poly``
(the target environment ships ``av`` and ``scipy`` but not ``librosa``/
``soundfile``/``audioread``, and installs are forbidden). PyAV decodes every
container (wav is a container too), so the whole load chain stays
dependency-free.

Output convention: float32 mono at ``sample_rate=16000``, matching
``librosa.load(sr=16000, mono=True)`` — the scale the Whisper feature
extractor expects.
"""

import io
import logging
import math
from typing import ByteString, List, Optional, Tuple, Union

import av
import numpy as np
import torch
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

_INT_PCM_NORMALIZERS = {
    np.dtype(np.int16): 32768.0,
    np.dtype(np.int32): 2147483648.0,
}


def resample_audio(
    audio: np.ndarray, orig_rate: int, target_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Rational-polyphase resample to ``target_rate``, matching librosa's default.

    ``librosa.load`` resamples with a polyphase filter whose ratio is the
    reduced fraction ``target/orig``; ``scipy.signal.resample_poly`` with the
    same reduced fraction (up=target//g, down=orig//g) is its scipy equivalent.
    Length matches ``ceil(len * target / orig)`` — librosa trims to
    ``ceil(len * target / orig)`` as well, so a ±0/1 sample tail difference at
    most (checked in the audio_utils tests).
    """
    if orig_rate == target_rate:
        return audio
    g = math.gcd(int(orig_rate), int(target_rate))
    up = int(target_rate) // g
    down = int(orig_rate) // g
    return resample_poly(audio, up, down).astype(np.float32)


def _to_float32(aframes: np.ndarray) -> np.ndarray:
    """Normalize decoded PCM to float32 in [-1, 1), matching librosa.load.

    ``frame.to_ndarray()`` returns integer-scale values for integer-PCM
    sources (int16 wav → int16 values ±32767; float wav → already [-1, 1]).
    librosa always returns [-1, 1] floats, and the Whisper
    feature extractor expects that scale, so integer input must be divided by
    its full-scale value (32768 for s16 — librosa uses 2**15 so ±32767 →
    just under 1.0).
    """
    if aframes.dtype in _INT_PCM_NORMALIZERS:
        aframes = aframes.astype(np.float32) / _INT_PCM_NORMALIZERS[aframes.dtype]
    return aframes.astype(np.float32)


def load_audio_from_bytes(audio_bytes: bytes, sample_rate: int = TARGET_SAMPLE_RATE, **kwargs) -> np.ndarray:
    """Decode raw audio-container bytes (wav/flac/mp3/aac/ogg…) to mono float32.

    Uses PyAV for both demux and decode (a dependency-free replacement for
    ``librosa.load``). Mono conversion = channel mean, identical to
    ``librosa.load(mono=True)``.
    """
    container = av.open(io.BytesIO(audio_bytes))
    try:
        audio_stream = container.streams.audio[0]
        frames = []
        for frame in container.decode(audio_stream):
            # to_ndarray → (channels, samples)
            frames.append(frame.to_ndarray())
        if not frames:
            return np.zeros(0, dtype=np.float32)
        aframes = np.concatenate(frames, axis=1)
        if aframes.shape[0] > 1:
            aframes = np.mean(aframes, axis=0)
        else:
            aframes = aframes[0]
    finally:
        container.close()
    return resample_audio(_to_float32(aframes), audio_stream.rate, sample_rate)


AudioInput = Union[
    np.ndarray,
    ByteString,
    str,
]


def load_audio_from_path(audio_path: str, sample_rate: int = TARGET_SAMPLE_RATE, **kwargs) -> np.ndarray:
    """Load an audio file (local path) to mono float32 at 16 kHz."""
    with open(audio_path, "rb") as f:
        return load_audio_from_bytes(f.read(), sample_rate)


def load_audio(audio: AudioInput, **kwargs) -> np.ndarray:
    if isinstance(audio, str):
        return load_audio_from_path(audio, **kwargs)
    elif isinstance(audio, (bytes, bytearray, memoryview)):
        return load_audio_from_bytes(bytes(audio), **kwargs)
    else:
        raise NotImplementedError(f"unsupported audio input type: {type(audio)}")


def fetch_audios(audios: List[AudioInput], **kwargs) -> List[np.ndarray]:
    audios = [load_audio(audio, **kwargs) for audio in audios]
    return audios


def extract_audio_from_video(
    video_input: Union[str, bytes], max_duration_seconds: Optional[float] = None
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Extract audio from video file using PyAV.

    Decode side only (the encoder helpers ``_write_audio``/``_resample_audio``
    are generation-side and out of scope). Returns mono float32 audio at the
    stream's native rate — the caller resamples to 16 kHz via
    ``resample_audio``. Integer-PCM output is normalized to [-1, 1)
    (raw to_ndarray passthrough would be correct for AAC/fltp mp4s but wrong
    for integer-PCM containers — see ``_to_float32``).
    """
    audio, audio_fps = None, None

    try:
        container_input = io.BytesIO(video_input) if isinstance(video_input, bytes) else video_input
        container = av.open(container_input)

        if len(container.streams.audio) > 0:
            audio_stream = container.streams.audio[0]
            audio_fps = audio_stream.rate

            # Prevent OOM: limit audio buffer size
            if max_duration_seconds is None:
                video_duration = container.duration / av.time_base if container.duration else 60.0
                max_duration_seconds = video_duration + 1.0

            max_samples = int(max_duration_seconds * audio_fps)

            audio_frames_list = []
            current_samples = 0

            for frame in container.decode(audio_stream):
                frame_np = frame.to_ndarray()
                audio_frames_list.append(frame_np)
                current_samples += frame_np.shape[1]
                if current_samples >= max_samples:
                    break

            if len(audio_frames_list) > 0:
                aframes = np.concatenate(audio_frames_list, axis=1)
                if aframes.shape[0] > 1:
                    aframes = np.mean(aframes, axis=0)
                else:
                    aframes = aframes[0]
                # Integer-PCM video tracks decode at integer scale; normalize to
                # [-1, 1) like librosa (the processor's expected scale).
                # AAC/fltp is already float — no-op for those.
                audio = _to_float32(aframes)

        container.close()

    except Exception as e:
        logger.warning("Failed to extract audio from video: %s", e)
        audio = None
        audio_fps = None

    return audio, audio_fps


def load_audio_from_video(
    video_path: str, sample_rate: int = TARGET_SAMPLE_RATE, **kwargs
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Extract + resample a video's embedded audio track to 16 kHz mono.

    Companion to ``extract_audio_from_video`` for the audio-in-video path;
    returns ``(audio_or_None, native_fps_or_None)`` with the audio already at
    ``sample_rate``.
    """
    audio, audio_fps = extract_audio_from_video(video_path)
    if audio is not None:
        audio = resample_audio(audio.astype(np.float32), audio_fps, sample_rate)
    return audio, audio_fps


__all__ = [
    "TARGET_SAMPLE_RATE",
    "resample_audio",
    "load_audio_from_path",
    "load_audio_from_bytes",
    "load_audio",
    "fetch_audios",
    "extract_audio_from_video",
    "load_audio_from_video",
]
