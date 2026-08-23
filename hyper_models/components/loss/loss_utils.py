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
from typing import Union

import torch
import torch.distributed as dist

from ..distributed.infrastructure import MeshContext
from ..utils.constants import IGNORE_INDEX
from .dist_utils import all_reduce


def count_loss_token(
    batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    # FIXME: VeOmni version -> HyperModels version
    """Calculate the total number of text_tokens/image_tokens/** for loss in a global batch, or one micro batch."""
    if isinstance(batches, dict):
        batches = [batches]
    token_len: dict[str, torch.Tensor] = {}

    def _count(obj):
        if isinstance(obj, dict) and not obj.get("padding_flag", False):
            # Hugging Face causal LM loss predicts labels from position one.
            foundation_tokens = torch.sum(obj["labels"][..., 1:] != IGNORE_INDEX)
            if "foundation_tokens" in token_len:
                foundation_tokens = token_len["foundation_tokens"] + foundation_tokens
            token_len["foundation_tokens"] = foundation_tokens  # text tokens

            for key in obj.keys():
                if key.endswith("_labels"):
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


def mean_global_loss(
    losses: Union[dict[str, torch.Tensor], torch.Tensor],
    current_token_counts: dict[str, torch.Tensor],
    step_token_counts: dict[str, torch.Tensor],
    device_mesh: MeshContext,
) -> dict[str, torch.Tensor]:
    # FIXME: VeOmni version -> HyperModels version
    """Calculate the global mean loss using explicit mesh information.

    FSDP divides gradients over its flattened DP+CP domain, so each local loss
    is weighted by valid tokens and multiplied by ``dp_size * cp_size``.

    Args:
        losses: A loss tensor or mapping of named loss tensors.
        current_token_counts: Token counts for the current micro batch.
        step_token_counts: Token counts for the local optimizer step.
        device_mesh: Trainer mesh context containing parallel sizes and groups.

    Returns:
        Token-weighted loss tensors keyed by loss name.

    Raises:
        ValueError: If sequence parallelism is enabled without a TP device mesh.
    """
    loss_dict = {}
    dp_cp_mesh = device_mesh.dp_cp_mesh
    dp_cp_group = dp_cp_mesh.get_group() if dp_cp_mesh is not None else None
    sequence_parallel = device_mesh.sequence_parallel
    sequence_parallel_group = None
    sequence_parallel_size = 1
    if sequence_parallel:
        if device_mesh.device_mesh is None or "tp" not in device_mesh.device_mesh.mesh_dim_names:
            raise ValueError("Sequence parallelism requires a DeviceMesh with a 'tp' dimension.")
        sequence_parallel_group = device_mesh.device_mesh["tp"].get_group()
        sequence_parallel_size = dist.get_world_size(group=sequence_parallel_group)

    if isinstance(losses, torch.Tensor):  # text loss only
        losses = {"foundation_loss": losses}

    for key, cur_loss in losses.items():
        loss_name = key.split("_loss", maxsplit=1)[0]  # foundation/image_decoder/**

        cur_token_len = current_token_counts[f"{loss_name}_tokens"]
        if sequence_parallel:
            cur_token_len = all_reduce(cur_token_len.item(), op="sum", group=sequence_parallel_group)

        all_reduced_len = all_reduce(
            step_token_counts[f"{loss_name}_tokens"].item(),
            op="sum",
            group=dp_cp_group,
        )

        if all_reduced_len != 0:
            local_weighted_loss = cur_loss * cur_token_len
            backward_loss = local_weighted_loss / all_reduced_len * device_mesh.dp_size * device_mesh.cp_size
            global_weighted_loss = all_reduce(
                local_weighted_loss.detach().item(),
                op="sum",
                group=dp_cp_group,
            )
            global_mean = cur_loss.new_tensor(global_weighted_loss / all_reduced_len)
            cur_loss = backward_loss + global_mean - backward_loss.detach()
        else:
            if not torch.allclose(cur_loss, torch.zeros_like(cur_loss)):
                raise ValueError(
                    f"The all_reduced_len for {loss_name}_tokens is 0, but the cur_loss is not 0: {cur_loss}"
                )

        if sequence_parallel:
            cur_loss = cur_loss / sequence_parallel_size

        loss_dict[key] = cur_loss

    return loss_dict
