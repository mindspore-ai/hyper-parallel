# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss token counting for trainer-side aggregation."""

from typing import Union

import torch

from hyper_parallel.data.constants import IGNORE_INDEX


def count_loss_token(
    batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    # FIXME: VeOmni version -> AutoModels version
    """Calculate the total number of text_tokens/image_tokens/** for loss in a global batch, or one micro batch."""
    if isinstance(batches, dict):
        batches = [batches]
    token_len: dict[str, torch.Tensor] = {}

    def _count(obj):
        if isinstance(obj, dict) and not obj.get("padding_flag", False):
            # Hugging Face causal LM loss predicts labels from position one.
            labels = obj.get("shift_labels")
            if labels is None:
                labels = obj["labels"][..., 1:]
            foundation_tokens = torch.sum(labels != IGNORE_INDEX)
            if "foundation_tokens" in token_len:
                foundation_tokens = token_len["foundation_tokens"] + foundation_tokens
            token_len["foundation_tokens"] = foundation_tokens  # text tokens

            for key in obj.keys():
                if key.endswith("_labels") and key != "shift_labels":
                    token_name = key.split("_labels")[0]
                    token_len[f"{token_name}_tokens"] = torch.sum(obj[key] != IGNORE_INDEX)  # image generation tokens

            if "image_output_mask" in obj:
                image_decoder_tokens = torch.sum(obj["image_output_mask"])
                if "image_decoder_tokens" in token_len:
                    image_decoder_tokens = token_len["image_decoder_tokens"] + image_decoder_tokens
                token_len["image_decoder_tokens"] = image_decoder_tokens  # image generation tokens
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _count(item)
        else:
            raise TypeError(f"Unsupported batch type: {type(obj)}")

    _count(batches)
    foundation_tokens = token_len.setdefault("foundation_tokens", torch.tensor(0))
    token_len.setdefault("image_decoder_tokens", foundation_tokens.new_zeros(()))
    return token_len


__all__ = ["count_loss_token"]
