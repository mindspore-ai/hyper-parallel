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
"""DeepSeek-V4.1 transform backed by its native processor."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.omni.omni_transform import OmniDataTransform
from hyper_parallel.models.deepseek_v41.adapter.data.image_processor import (
    IMAGE,
    IMAGE_NEW_LINE,
    TEXT,
    ImageInput,
)
from hyper_parallel.models.deepseek_v41.adapter.data.processor import (
    DeepseekV41Processor,
)

logger = logging.getLogger(__name__)


class DeepseekV41OmniTransform(OmniDataTransform):
    """Encode OpenAI-style image conversations for DeepSeek-V4.1 SFT."""

    def __init__(
            self,
            processor: DeepseekV41Processor,
            *,
            max_seq_len: int = 4096,
            thinking_mode: str = "chat",
            drop_thinking: bool = True,
    ) -> None:
        """Bind the native processor to the generic Omni lifecycle."""
        if not isinstance(processor, DeepseekV41Processor):
            raise TypeError("DeepseekV41OmniTransform requires DeepseekV41Processor")
        self.thinking_mode = thinking_mode
        self.drop_thinking = drop_thinking
        super().__init__(
            max_seq_len=max_seq_len,
            processor=processor,
        )

    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Encode one conversation into the common Omni sample contract."""
        messages = self.prepare_messages(sample)
        logger.info(f'[deepseek] encode_sample {messages}')
        if messages[-1].get("role") != "assistant":
            raise ValueError("DeepSeek-V4.1 SFT sample must end with an assistant message")

        prompt, media = self.processor.chat_template(
            messages,
            thinking_mode=self.thinking_mode,
            drop_thinking=self.drop_thinking,
            return_multi_modal_data=True,
        )
        logger.info(f'[deepseek] prompt {prompt}, media {media}, media["images"] {media["images"]}')
        input_ids, token_types, image_inputs = self.processor.image_processor(
            prompt,
            media["images"],
            self.processor.tokenizer,
            self.processor,
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_types = torch.tensor(token_types, dtype=torch.long)
        normalized_image_inputs = [] if image_inputs is None else image_inputs
        assistant_start = self._get_assistant_start(messages, normalized_image_inputs)
        labels = self._build_labels(input_ids, token_types, assistant_start)
        # Packed boundaries must align with the V4.1 ratio-two KV compressor.
        # Padding before packing leaves image starts local to their source sample.
        if input_ids.numel() % 2:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            input_ids = torch.cat((input_ids, input_ids.new_tensor([0 if pad_token_id is None else pad_token_id])))
            labels = torch.cat((labels, labels.new_tensor([IGNORE_INDEX])))
            token_types = torch.cat((token_types, token_types.new_tensor([TEXT])))
        if input_ids.numel() > self.max_seq_len:
            raise ValueError(
                f"DeepSeek-V4.1 sample length {input_ids.numel()} exceeds max_seq_len={self.max_seq_len}"
            )
        encoded_sample = {
            "input_ids": input_ids,
            "labels": labels,
            "token_types": token_types,
        }
        image_fields = self._build_image_fields(normalized_image_inputs)
        encoded_sample.update(image_fields)
        return encoded_sample

    def encode_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Convert packing metadata into the V4.1 model batch contract."""
        encoded_batch = dict(batch)
        if "pixel_values" not in encoded_batch:
            return encoded_batch

        required_fields = (
            "image_patch_lengths",
            "image_vit_grid_hw",
            "image_llm_grid_hw",
            "image_token_starts",
        )
        missing_fields = []
        for field in required_fields:
            if field not in encoded_batch:
                missing_fields.append(field)
        if missing_fields:
            raise ValueError(
                "DeepSeek-V4.1 image batch is missing fields: "
                + ", ".join(missing_fields)
            )

        patch_lengths = encoded_batch.pop("image_patch_lengths")
        image_batch_indices = self._build_image_batch_indices(patch_lengths)
        flat_patch_lengths = patch_lengths.reshape(-1)
        zero = flat_patch_lengths.new_zeros(1)
        encoded_batch["image_patch_offsets"] = torch.cat(
            (zero, flat_patch_lengths.cumsum(dim=0))
        )
        encoded_batch["image_batch_indices"] = image_batch_indices
        encoded_batch["image_vit_grid_hw"] = encoded_batch["image_vit_grid_hw"].reshape(-1, 2)
        encoded_batch["image_llm_grid_hw"] = encoded_batch["image_llm_grid_hw"].reshape(-1, 2)
        encoded_batch["image_token_starts"] = encoded_batch["image_token_starts"].reshape(-1)
        return encoded_batch

    @staticmethod
    def _build_image_batch_indices(patch_lengths: torch.Tensor) -> torch.Tensor:
        """Map collated image metadata rows to their packed batch rows."""
        if patch_lengths.ndim == 1:
            image_batch_indices = patch_lengths.new_zeros(patch_lengths.numel())
            return image_batch_indices
        if patch_lengths.ndim != 2:
            raise ValueError("image_patch_lengths must have shape [images] or [batch, images]")

        batch_size, images_per_batch = patch_lengths.shape
        image_batch_indices = torch.arange(
            batch_size,
            dtype=patch_lengths.dtype,
            device=patch_lengths.device,
        ).repeat_interleave(images_per_batch)
        return image_batch_indices

    def _get_assistant_start(
            self,
            messages: Sequence[Mapping[str, Any]],
            image_inputs: Sequence[ImageInput],
    ) -> int:
        """Return the first supervised token after the final assistant header."""
        prompt, _ = self.processor.chat_template(
            list(messages[:-1]),
            thinking_mode=self.thinking_mode,
            drop_thinking=self.drop_thinking,
            return_multi_modal_data=True,
        )
        prompt_tokens = self.processor.tokenizer.encode(prompt)
        placeholder_count = 0
        for token in prompt_tokens:
            if token == self.processor.image_token_id:
                placeholder_count += 1

        if placeholder_count > len(image_inputs):
            raise ValueError("Assistant prefix contains more images than the encoded sample")

        assistant_start = len(prompt_tokens)
        for image_input in image_inputs[:placeholder_count]:
            assistant_start += image_input.types.numel() - 1
        return assistant_start

    @staticmethod
    def _build_labels(
            input_ids: torch.Tensor,
            token_types: torch.Tensor,
            assistant_start: int,
    ) -> torch.Tensor:
        """Keep final-assistant text as the only supervised token range."""
        if not 0 <= assistant_start < input_ids.numel():
            raise ValueError("DeepSeek-V4.1 assistant response must contain at least one token")

        labels = input_ids.clone()
        labels[:assistant_start] = IGNORE_INDEX
        labels[token_types != TEXT] = IGNORE_INDEX
        return labels

    @classmethod
    def _build_image_fields(
            cls,
            image_inputs: Sequence[ImageInput],
    ) -> dict[str, torch.Tensor]:
        """Flatten official ImageInput objects into packing-ready tensors."""
        if not image_inputs:
            image_fields = {}
            return image_fields

        patches = []
        patch_lengths = []
        vit_grid_hw = []
        llm_grid_hw = []
        token_starts = []
        for image_input in image_inputs:
            if not isinstance(image_input, ImageInput):
                raise TypeError("DeepSeek-V4.1 image_processor must return ImageInput objects")

            patches.append(image_input.patches)
            patch_lengths.append(image_input.patches.shape[0])
            vit_grid_hw.append((image_input.n_vit_h, image_input.n_vit_w))
            llm_grid_hw.append(cls._get_llm_grid(image_input.types))
            token_starts.append(image_input.start)

        image_fields = {
            "pixel_values": torch.cat(patches, dim=0),
            "image_patch_lengths": torch.tensor(patch_lengths, dtype=torch.long),
            "image_vit_grid_hw": torch.tensor(vit_grid_hw, dtype=torch.long),
            "image_llm_grid_hw": torch.tensor(llm_grid_hw, dtype=torch.long),
            "image_token_starts": torch.tensor(token_starts, dtype=torch.long),
        }
        return image_fields

    @staticmethod
    def _get_llm_grid(token_types: torch.Tensor) -> tuple[int, int]:
        """Recover the LLM image grid encoded by the official token types."""
        grid_height = int((token_types == IMAGE_NEW_LINE).sum())
        image_token_count = int((token_types == IMAGE).sum())
        if grid_height <= 0 or image_token_count % grid_height != 0:
            raise ValueError("DeepSeek-V4.1 image token types do not describe a rectangular grid")

        grid_width = image_token_count // grid_height
        image_grid = (grid_height, grid_width)
        return image_grid


def build_deepseek_v41_omni_transform(
        *,
        processor: DeepseekV41Processor,
        max_seq_len: int = 4096,
        thinking_mode: str = "chat",
        drop_thinking: bool = True,
) -> DeepseekV41OmniTransform:
    """Build the native DeepSeek-V4.1 transform from a prebuilt processor."""
    data_transform = DeepseekV41OmniTransform(
        processor,
        max_seq_len=max_seq_len,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
    )
    return data_transform


__all__ = ["DeepseekV41OmniTransform", "build_deepseek_v41_omni_transform"]
