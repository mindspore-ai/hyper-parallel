# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Multimodal (audio/video) data utilities.

``audio_utils`` decodes and resamples audio via PyAV + scipy;
``video_utils`` does video decode + temporal sampling via PyAV.
"""

from .audio_utils import (
    TARGET_SAMPLE_RATE,
    extract_audio_from_video,
    fetch_audios,
    load_audio,
    load_audio_from_bytes,
    load_audio_from_path,
    load_audio_from_video,
    resample_audio,
)
from .video_utils import (
    calculate_frame_indices,
    fetch_videos,
    fetch_videos_metadata,
    load_video,
    load_video_from_path,
    smart_audio_nframes,
    smart_resize,
    smart_video_nframes,
)

__all__ = [
    "TARGET_SAMPLE_RATE",
    "calculate_frame_indices",
    "extract_audio_from_video",
    "fetch_audios",
    "fetch_videos",
    "fetch_videos_metadata",
    "load_audio",
    "load_audio_from_bytes",
    "load_audio_from_path",
    "load_audio_from_video",
    "load_video",
    "load_video_from_path",
    "resample_audio",
    "smart_audio_nframes",
    "smart_resize",
    "smart_video_nframes",
]
