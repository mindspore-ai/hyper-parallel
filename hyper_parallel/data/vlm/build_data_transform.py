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
"""Build the VLM multimodal sample transform."""

import re
from typing import Any, Optional

import torch

from hyper_parallel.data.multimodal import fetch_audios, fetch_videos_metadata
from hyper_parallel.data.constants import IGNORE_INDEX

_SEQ_FIELDS = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")


def _get_exact_token_id(tokenizer: Any, token: str) -> int:
    """Resolve a special-token id strictly.

    Returns the vocab id of ``token`` without accepting the unk id as a match,
    so a missing special token fails loudly instead of silently matching every
    literal ``<unk>`` in the sequence.
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is not None and token_id != unk_token_id:
        return token_id
    raise ValueError(f"Cannot find token ({token}) in tokenizer vocab.")


class VLMChatTransform:
    """Encode one multimodal conversation into one padded model sample.

    ``messages`` follows the Qwen3-VL content-list format
    (``{"type": "image", "url": ...}`` / ``{"type": "text", "text": ...}``),
    and ``<image>``/``<audio>`` string placeholders are also accepted. Every
    ``assistant`` turn's content plus its closing ``<|im_end|>``/newline is the
    training target; all other positions are masked with -100.

    Audio/video content uses a two-step flow — render text with
    ``apply_chat_template(tokenize=False)``, pre-decode the media ourselves,
    then call the processor directly. HF 4.57.1's
    ``apply_chat_template(tokenize=True)`` audio path calls ``load_audio``
    which requires torchcodec/librosa/soundfile — none installed in the target
    env — so the direct processor call is the only working route.

    Args:
        processor: Qwen3-VL/Qwen3-Omni processor (tokenizer + image/video/audio processors).
        max_seq_len: Padding budget; over-long samples run at full length.
    """

    def __init__(self, processor: Any, *, max_seq_len: int = 256, video_fps: float = 2.0) -> None:
        """Store the processor, target sequence length, and video sampling fps."""
        self.processor = processor
        self.max_seq_len = max_seq_len
        # DEFAULT_FPS = 2.0; frame_factor=2 matches the thinker's
        # temporal_patch_size so the HF video processor needs no padding.
        self.video_fps = video_fps
        self._image_token_id = self._resolve_token_id("image_token")
        self._video_token_id = self._resolve_token_id("video_token")
        self._audio_token_id = self._resolve_token_id("audio_token")
        # Anchor the label mask on the ``assistant`` special token instead of
        # the prompt prefix length. The ``user`` token closes each assistant
        # span.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self._assistant_token_id = _get_exact_token_id(tokenizer, "assistant")
        self._user_token_id = _get_exact_token_id(tokenizer, "user")

    def _resolve_token_id(self, token_attr: str) -> Optional[int]:
        """Resolve a placeholder token id (e.g. the image pad token) from the processor.

        Qwen3-VL-style processors emit ``mm_token_type_ids`` directly; others
        (e.g. the Qwen3-Omni processor) do not, so we mark modality positions
        ourselves from ``input_ids``. Prefer an explicit ``*_token_id``
        attribute, then fall back to converting the ``*_token`` string via the
        tokenizer.
        """
        token_id = getattr(self.processor, f"{token_attr}_id", None)
        if isinstance(token_id, int):
            return token_id
        token = getattr(self.processor, token_attr, None)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if token is not None and tokenizer is not None:
            resolved = tokenizer.convert_tokens_to_ids(token)
            if isinstance(resolved, int) and resolved >= 0:
                return resolved
        return None

    def _build_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mark modality placeholder positions: image=1, video=2 (audio=3, reserved).

        image=1 / video=2 follow the standard convention; audio=3 marks audio
        runs for downstream consumers. The Qwen3-Omni thinker selects
        modality embeddings from ``input_ids`` directly (HF
        ``get_placeholder_mask``), not from this field, so the audio value is
        informational until a consumer needs it.
        """
        mm = torch.zeros_like(input_ids)
        if self._image_token_id is not None:
            mm[input_ids == self._image_token_id] = 1
        if self._video_token_id is not None:
            mm[input_ids == self._video_token_id] = 2
        if self._audio_token_id is not None:
            mm[input_ids == self._audio_token_id] = 3
        return mm

    def _normalize_messages(
        self, messages: Any, images: Any = None, audios: Any = None, videos: Any = None
    ) -> Any:
        """Split ``<image>``/``<video>``/``<audio>`` string placeholders into content-list parts."""
        if images is None:
            images = []
        if audios is None:
            audios = []
        if videos is None:
            videos = []
        image_queue = list(images)
        audio_queue = list(audios)
        video_queue = list(videos)
        normalized = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                parts = []
                for token in re.split(r"(<image>|<video>|<audio>)", content):
                    if token in ("<image>", "<video>", "<audio>"):
                        queue = (
                            image_queue if token == "<image>"
                            else video_queue if token == "<video>"
                            else audio_queue
                        )
                        if not queue:
                            raise ValueError(
                                f"message content contains a {token} placeholder but no matching media is available"
                            )
                        media_type = token[1:-1]
                        parts.append({"type": media_type, "url": queue.pop(0)})
                    elif token:
                        parts.append({"type": "text", "text": token})
                content = parts
            normalized.append({"role": message.get("role"), "content": content})
        return normalized

    def _collect_media(
        self, messages: Any, images: Any, audios: Any, videos: Any
    ) -> tuple[list, list, list, bool]:
        """Collect per-conversation media in content order.

        Returns ``(image_paths, audio_arrays, video_tensors, use_av)``. Audio
        is pre-decoded to 16 kHz mono float32 (``fetch_audios``, PyAV+scipy);
        videos are decoded + temporally sampled to (T, C, H, W) float tensors
        (``fetch_videos_metadata``, PyAV; spatial resize stays with the HF
        video processor, whose ``do_sample_frames=False`` skips its internal
        frame sampling since frames are pre-sampled); images keep their
        path/url form for the processor's own image loader.

        ``use_av``: every video in the conversation carries an embedded audio
        track, so the processor interleaves video and audio tokens on the
        shared M-RoPE time axis (HF 4.57.1 ``use_audio_in_video=True`` per-video
        expansion). In that mode the returned ``audio_arrays`` is ordered by
        conversation content position — video parts contribute their embedded
        track, standalone audio parts their own array — matching the order the
        processor's ``audio_lengths`` iterator consumes them. Any video
        without a track falls back to the visual-only expansion.
        """
        image_paths = []
        audio_refs = []
        video_refs = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image":
                    for key in ("image", "url", "path"):
                        if key in part:
                            image_paths.append(part[key])
                            break
                elif part.get("type") == "audio":
                    for key in ("audio", "url", "path"):
                        if key in part:
                            audio_refs.append(part[key])
                            break
                elif part.get("type") == "video":
                    for key in ("video", "url", "path"):
                        if key in part:
                            video_refs.append(part[key])
                            break

        # Probe tracks first; only interleave when every video has audio.
        video_tensors, video_audios, _, _ = [], [], None, None
        if video_refs:
            fetched = fetch_videos_metadata(
                video_refs, use_audio_in_video=True, fps=self.video_fps, frame_factor=2
            )
            video_tensors, fetched_audios = fetched[0], fetched[2]
            if all(a is not None for a in fetched_audios):
                # Conversation-order audio binding (V video_audios_iter):
                # walk the content again, interleaving per-position sources.
                # Standalone audio refs are decoded up-front (they may inter-
                # leave between videos, so decode all before binding).
                fetched_standalone = fetch_audios(audio_refs) if audio_refs else []
                audio_arrays = []
                video_audios_iter = iter(fetched_audios)
                standalone_iter = iter(fetched_standalone)
                for message in messages:
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if part.get("type") == "video":
                            audio_arrays.append(next(video_audios_iter))
                        elif part.get("type") == "audio":
                            audio_arrays.append(next(standalone_iter))
                return image_paths, audio_arrays, video_tensors, True
            # Fall back to visual-only expansion for trackless videos.
            video_tensors, _, _, _ = fetch_videos_metadata(
                video_refs, use_audio_in_video=False, fps=self.video_fps, frame_factor=2
            )

        audio_arrays = fetch_audios(audio_refs) if audio_refs else []
        return image_paths, audio_arrays, video_tensors, False

    def _render(self, messages: Any, *, add_generation_prompt: bool) -> str:
        """Render one conversation to text (tokenize=False)."""
        chat_template = getattr(self.processor, "chat_template", None)
        if not chat_template:
            tokenizer = getattr(self.processor, "tokenizer", None)
            chat_template = getattr(tokenizer, "chat_template", None)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
            chat_template=chat_template,
        )

    def _encode(self, messages: Any, *, add_generation_prompt: bool) -> dict[str, Any]:
        """Render and encode one conversation with the processor.

        Image-only conversations use the single-call path
        (``apply_chat_template(tokenize=True, return_dict=True)``) so the
        image baseline stays bit-comparable. Conversations with audio/video
        take the two-step flow (render text, then direct processor call with
        pre-decoded media) — required because HF 4.57.1's tokenizing
        ``apply_chat_template`` cannot load audio in this env.
        """
        has_audio = any(
            part.get("type") == "audio"
            for message in messages
            for part in (message.get("content") or [])
            if isinstance(message.get("content"), list)
        )
        has_video = any(
            part.get("type") == "video"
            for message in messages
            for part in (message.get("content") or [])
            if isinstance(message.get("content"), list)
        )
        if not has_audio and not has_video:
            chat_template = getattr(self.processor, "chat_template", None)
            if not chat_template:
                tokenizer = getattr(self.processor, "tokenizer", None)
                chat_template = getattr(tokenizer, "chat_template", None)
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
                chat_template=chat_template,
            )

        text = self._render(messages, add_generation_prompt=add_generation_prompt)
        image_paths, audio_arrays, video_tensors, use_av = self._collect_media(messages, None, None, None)
        images = image_paths if image_paths else None
        audio = audio_arrays if audio_arrays else None
        videos = video_tensors if video_tensors else None
        self._last_use_av = use_av
        return self.processor(
            text=[text],
            images=[images] if images else None,
            audio=audio,
            videos=[videos] if videos else None,
            fps=self.video_fps,
            use_audio_in_video=use_av,
            return_tensors="pt",
            padding=True,
        )

    def _truncate_and_pad(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pad sequence fields up to ``max_seq_len`` without truncating long samples.

        Never truncate a sample — ``max_seq_len`` only sizes the padding
        budget, and over-long samples run at full length. Padding is kept so
        the collator
        still receives fixed-length tensors for in-budget samples.
        """
        max_len = self.max_seq_len
        for field in _SEQ_FIELDS:
            value = sample[field]
            if value.shape[0] < max_len:
                pad_value = IGNORE_INDEX if field == "labels" else 0
                sample[field] = torch.cat([
                    value,
                    torch.full((max_len - value.shape[0],), pad_value, dtype=value.dtype),
                ])
        return sample

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask everything except assistant turns.

        For every ``assistant`` special-token position ``assis_i`` the span
        ``[assis_i + 2, next_user_start - 1)`` — assistant content plus the
        closing ``<|im_end|>`` and newline — is unmasked; everything else
        (system / user turns, modality tokens, special tokens) stays
        IGNORE_INDEX.
        """
        labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
        user_start_index = torch.where(input_ids == self._user_token_id)[0].tolist()
        assistant_start_index = torch.where(input_ids == self._assistant_token_id)[0].tolist()
        user_start_index.append(len(input_ids) + 1)
        user_i = 0
        for assis_i in assistant_start_index:
            while user_start_index[user_i] < assis_i:
                user_i += 1
            labels[assis_i + 2 : user_start_index[user_i] - 1] = input_ids[
                assis_i + 2 : user_start_index[user_i] - 1
            ]
        return labels

    def __call__(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode one record into a padded model sample."""
        messages = self._normalize_messages(
            record["messages"],
            record.get("images"),
            record.get("audios"),
            record.get("videos"),
        )
        full = self._encode(messages, add_generation_prompt=False)

        input_ids = torch.tensor(full["input_ids"][0], dtype=torch.long)
        labels = self._build_labels(input_ids)

        if "mm_token_type_ids" in full:
            mm_token_type_ids = torch.tensor(full["mm_token_type_ids"][0], dtype=torch.long)
        else:
            # Processors that do not emit mm_token_type_ids (e.g. Qwen3-Omni):
            # derive it from the modality placeholder positions.
            mm_token_type_ids = self._build_mm_token_type_ids(input_ids)

        sample = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(full["attention_mask"][0], dtype=torch.long),
            "mm_token_type_ids": mm_token_type_ids,
            "labels": labels,
            "pixel_values": full.get("pixel_values"),
            "image_grid_thw": full.get("image_grid_thw"),
            # Video: pass through the HF video processor's packed patches +
            # grid (+ per-grid seconds for position-id computation).
            "pixel_values_videos": full.get("pixel_values_videos"),
            "video_grid_thw": full.get("video_grid_thw"),
        }
        second_per_grid = full.get("video_second_per_grid")
        if second_per_grid is not None:
            # The processor emits a per-video float list; the thinker's
            # ``get_rope_index`` consumes it as a tensor (HF forward kwarg).
            sample["video_second_per_grid"] = (
                second_per_grid
                if torch.is_tensor(second_per_grid)
                else torch.tensor(second_per_grid, dtype=torch.float)
            )

        # AV interleave: tell the thinker's M-RoPE which interleave mode the
        # processor used (native 4.57.1 forward kwarg; the processor itself
        # does not echo the flag back). Only meaningful when this sample
        # actually interleaved audio into video.
        if getattr(self, "_last_use_av", False):
            sample["use_audio_in_video"] = True

        # Audio: pop the padded mel bank, compute per-audio lengths, and pack
        # valid frames flat. The thinker consumes
        # (T_total, n_mels) ``input_features`` + ``audio_feature_lengths``
        # (HF ``get_audio_features`` splits by lengths after the conv tower).
        feature_attention_mask = full.get("feature_attention_mask")
        if feature_attention_mask is not None:
            input_features = full["input_features"]
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            valid_mask = audio_feature_lengths != 0
            packed = input_features[valid_mask].permute(0, 2, 1)[
                feature_attention_mask[valid_mask].bool()
            ]
            sample["input_features"] = packed
            sample["audio_feature_lengths"] = audio_feature_lengths
        elif full.get("input_features") is not None:
            # Single-audio unpadded path: (1, n_mels, T) → (T, n_mels).
            sample["input_features"] = full["input_features"][0].permute(1, 0)
            sample["audio_feature_lengths"] = torch.tensor([full["input_features"].shape[-1]])

        return self._truncate_and_pad(sample)


def build_vlm_data_transform(
        *,
        processor: Any = None,
        max_seq_len: int = 256,
        **transform_options: Any,
) -> VLMChatTransform:
    """Build the VLM sample transform.

    Args:
        processor: Qwen3-VL processor used to render and encode conversations.
        max_seq_len: Padding budget; over-long samples run at full length.
        **transform_options: Reserved model-specific transform options;
            ``video_fps`` (default 2.0) sets the video temporal sampling
            rate.

    Returns:
        The configured :class:`VLMChatTransform`.

    Raises:
        ValueError: If ``processor`` is not provided.
    """
    if processor is None:
        raise ValueError("processor is required for the VLM data transform")
    video_fps = float(transform_options.pop("video_fps", 2.0))
    if transform_options:
        raise ValueError(f"unknown transform options: {sorted(transform_options)}")
    return VLMChatTransform(processor, max_seq_len=max_seq_len, video_fps=video_fps)


__all__ = ["VLMChatTransform", "build_vlm_data_transform"]
