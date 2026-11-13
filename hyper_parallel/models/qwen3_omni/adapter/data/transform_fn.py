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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen3-Omni sample transform: OmniDataTransform over the shared VLM encoder.

The family's recipes pair this transform with
``hyper_parallel.data.vlm.build_vlm_dataset``. The shared ``VLMChatTransform``
already renders conversations through the Qwen3-Omni processor (text/image/
video/audio, packed audio features); this module adapts it to the
:class:`~hyper_parallel.data.omni.omni_transform.OmniDataTransform` contract
the trainer validates, and anchors relative media paths against the JSON
directory the records were loaded from.
"""

import os
from typing import Any

from hyper_parallel.data.omni.omni_transform import OmniDataTransform
from hyper_parallel.data.vlm.build_data_transform import build_vlm_data_transform


class Qwen3OmniTransform(OmniDataTransform):
    """Wrap the shared VLM sample transform as an OmniDataTransform (eager encode)."""

    def __init__(
            self, *, max_seq_len: int, processor: Any, training_mode: str = "sft",
            media_root: str | None = None, **transform_options: Any,
    ) -> None:
        """Store the omni contract fields and build the shared VLM encoder.

        Args:
            max_seq_len: Padding budget; over-long samples run at full length.
            processor: Qwen3-Omni processor used to render and encode
                conversations.
            training_mode: Omni training mode; only ``"sft"`` is supported —
                the shared VLM transform has no pretrain rendering path.
            media_root: Optional directory that relative media paths in the
                per-modality record keys (``images`` / ``audios`` /
                ``videos``) are resolved against.
            **transform_options: Passed through to the shared VLM transform
                (e.g. ``video_fps``).
        """
        OmniDataTransform.__init__(
            self, max_seq_len=max_seq_len, processor=processor, training_mode=training_mode,
        )
        if training_mode != "sft":
            raise ValueError(f"Qwen3OmniTransform supports sft only, got {training_mode!r}")
        if media_root is not None and not isinstance(media_root, str):
            raise TypeError(f"media_root must be a str, got {type(media_root).__name__}")
        self.media_root = media_root
        self.chat_transform = build_vlm_data_transform(
            processor=processor, max_seq_len=max_seq_len, **transform_options,
        )

    def _resolve_media_paths(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Resolve relative media paths in the top-level per-modality keys.

        ``VLMDataset._resolve_record_paths`` resolves the images/videos keys
        but not audios, and content-list media is only resolved when the
        record carries ONLINE_SOURCE_PATH_KEY (which the VLM online source
        does not set). Re-anchor every media key here against the JSON
        directory the records were loaded from.
        """
        media_root = self.media_root
        if media_root is None:
            return sample
        resolved = dict(sample)
        for key in ("images", "audios", "videos"):
            paths = resolved.get(key)
            if isinstance(paths, list):
                resolved[key] = [
                    os.path.join(media_root, path) if isinstance(path, str) and not path.startswith(
                        ("http://", "https://", "/", "data:")
                    ) else path
                    for path in paths
                ]
        return resolved

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Encode one record through the shared VLM transform.

        prepare_messages deep-copies and resolves media paths; the per-modality
        record keys (images/audios/videos) pass through untouched so
        VLMChatTransform._normalize_messages still consumes them.

        The prepared messages go into a shallow copy of the record, never back
        into the caller's dict: VLMDataset hands out its stored records by
        reference (_TransformDataset.__getitem__), and writing the expanded
        content-list into the shared record makes the second epoch's
        prepare_messages re-encode already-expanded image blocks.
        """
        prepared = self.prepare_messages(sample)
        record = self._resolve_media_paths(sample)
        record = {**record, "messages": prepared}
        return self.chat_transform(record)

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        """Callable contract for ``build_vlm_dataset``'s per-record apply.

        The family recipes pair this transform with
        ``hyper_parallel.data.vlm.build_vlm_dataset``, whose _TransformDataset
        calls ``transform(record)`` directly; OmniDataTransform itself only
        exposes the encode_sample/preencode/postencode hooks used by the
        online-mapping path. Both contracts meet here.
        """
        return self.encode_sample(record)


def build_qwen3_omni_transform(
        *,
        processor: Any = None,
        max_seq_len: int = 256,
        training_mode: str = "sft",
        **transform_options: Any,
) -> Qwen3OmniTransform:
    """Build the Qwen3-Omni sample transform.

    Args:
        processor: Qwen3-Omni processor used to render and encode
            conversations.
        max_seq_len: Padding budget; over-long samples run at full length.
        training_mode: Omni training mode; only ``"sft"`` is supported.
        **transform_options: Passed through to the shared VLM transform
            (e.g. ``video_fps``); ``media_root`` optionally anchors relative
            media paths.

    Returns:
        The configured :class:`Qwen3OmniTransform`.

    Raises:
        ValueError: If ``processor`` is not provided.
    """
    if processor is None:
        raise ValueError("processor is required for the Qwen3-Omni data transform")
    return Qwen3OmniTransform(
        max_seq_len=max_seq_len, processor=processor, training_mode=training_mode,
        **transform_options,
    )


__all__ = ["Qwen3OmniTransform", "build_qwen3_omni_transform"]
