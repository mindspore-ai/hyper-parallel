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
"""Opt-in Qwen3.6 image Encoder-DP/CP bridge, separate from the text adapter."""

from __future__ import annotations

import importlib
from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import
from transformers.modeling_outputs import BaseModelOutputWithPooling

from hyper_parallel import HSDPModule
from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head
from hyper_parallel.distributed.encoder_parallel.attention import current_encoder_attention_context
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderParallelAdapterSpec, EncoderSegment, EncoderWorkItem,
)
from hyper_parallel.models.replacement import module_replacement


class QwenEncoderAttention(nn.Module):
    """Keep native parameter names; exchange sequence/head axes only when active."""

    def __init__(self, source: nn.Module, module_fqn: str) -> None:
        """Retain the Torch/HF module state and its inactive forward implementation."""
        super().__init__()
        self.__dict__.update(source.__dict__)
        self._modules = source._modules.copy()
        self._parameters = source._parameters.copy()
        self._buffers = source._buffers.copy()
        self.source_forward = type(source).forward
        self.module_fqn = module_fqn

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                position_embeddings: tuple | None = None, **kwargs: Any) -> torch.Tensor:
        """Apply independent, padding-masked image attention over Encoder-CP."""
        context = current_encoder_attention_context()
        if context is None:
            return self.source_forward(self, hidden_states, cu_seqlens=cu_seqlens,
                                       position_embeddings=position_embeddings, **kwargs)
        context.record(self.module_fqn)
        context.validate_runtime_head_counts(module_fqn=self.module_fqn, query_heads=self.num_heads,
                                             key_heads=self.num_heads, value_heads=self.num_heads)
        query, key, value = self.qkv(hidden_states).reshape(-1, 3, self.num_heads, self.head_dim).unbind(1)
        cosine, sine = (entry.float().unsqueeze(-2) for entry in position_embeddings)

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            """Apply Qwen's FP32 vision rotary calculation and restore Q/K dtype."""
            left, right = tensor.float().chunk(2, dim=-1)
            return (tensor.float() * cosine + torch.cat((-right, left), dim=-1) * sine).to(tensor.dtype)

        query, key = rotate(query), rotate(key)
        outputs = []
        for segment in context.segments:
            start, length = segment.packed_offset, segment.local_padded_length
            parts = [tensor[start:start + length] for tensor in (query, key, value)]
            if context.cp_mesh.size() > 1:
                parts = [ulysses_seq_to_head(tensor, seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
                         for tensor in parts]
            valid = parts[0].shape[0] if segment.is_dummy else segment.item.sequence_length
            mask = torch.arange(parts[0].shape[0], device=hidden_states.device) < valid
            result = functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                *(tensor.transpose(0, 1).unsqueeze(0) for tensor in parts),
                attn_mask=mask.reshape(1, 1, 1, -1), dropout_p=0.0, is_causal=False, scale=self.scaling,
            ).squeeze(0).transpose(0, 1).contiguous()
            if context.cp_mesh.size() > 1:
                result = ulysses_head_to_seq(result, seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
            outputs.append(result.reshape(length, -1))
        return self.proj(torch.cat(outputs, dim=0))


@module_replacement
def replace_vision_attention(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any]) -> nn.Module:
    """Replace the visual attention without renaming or duplicating weights."""
    del context
    return QwenEncoderAttention(module, module_fqn)


class QwenVisionBridge:
    """Bind the native patch embedding, vision blocks and spatial merger."""

    input_fields = ("pixel_values", "image_grid_thw")

    def __init__(self, model: nn.Module) -> None:
        """Bind the native visual subtree without registering duplicate parameters."""
        self.model = model
        self.vision = model.model.visual
        self.merge_size = self.vision.spatial_merge_size
        self.image_token_id = model.config.image_token_id
        self.feature_context = ContextVar("qwen_encoder_features", default=None)

    @property
    def head_counts(self) -> dict[str, int]:
        """Return native visual head counts, independent of decoder TP."""
        return dict.fromkeys(("query", "key", "value"), self.vision.config.num_heads)

    def validate_training(self, options: Mapping[str, Any]) -> None:
        """Reject modes whose HF feature insertion contract is not yet adapted."""
        if isinstance(self.vision, HSDPModule) or isinstance(self.model.model, HSDPModule):
            raise NotImplementedError("Wrap visual children, not model.visual/model as standalone FSDP units")
        if options.get("micro_batch_size") != 1:
            raise ValueError("Qwen encoder parallel currently requires micro_batch_size=1")
        if options.get("sequence_parallel") or options.get("loss_parallel"):
            raise NotImplementedError("Qwen image insertion requires SP=false and loss_parallel=false")
        if options.get("cp_size", 1) != 1 and not getattr(self.model, "_hp_vlm_cp_enabled", False):
            raise NotImplementedError(
                "Decoder CP requires QwenVLMParallelBatch to own image insertion and loss shifting"
            )
        if options.get("activation_checkpoint", "off") not in (None, "off"):
            raise NotImplementedError("Qwen encoder parallel currently requires activation_checkpoint.mode=off")
        if getattr(self.vision, "gradient_checkpointing", False):
            raise NotImplementedError("Qwen encoder parallel does not support vision gradient checkpointing")

    def parse_images(self, batch: Mapping[str, Any]) -> tuple[EncoderWorkItem, ...]:
        """Validate image/token correspondence before launching visual collectives."""
        tokens = batch.get("vlm_global_input_ids", batch["input_ids"])
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("Qwen encoder parallel requires a single unpacked sample")
        boundaries = batch.get("cu_seq_lens")
        if boundaries is not None and boundaries.numel() > 2:
            raise NotImplementedError("Qwen encoder parallel does not support multi-sample packing")
        if batch.get("pixel_values_videos") is not None or batch.get("video_grid_thw") is not None:
            raise NotImplementedError("Qwen encoder parallel currently supports images, not videos")
        positions = (tokens[0] == self.image_token_id).nonzero().flatten().tolist()
        pixels, grids = (batch.get(name) for name in self.input_fields)
        if pixels is None and grids is None:
            if positions:
                raise ValueError("Image tokens are present without image tensors")
            return ()
        if not torch.is_tensor(pixels) or not torch.is_tensor(grids) or pixels.ndim != 2:
            raise ValueError("Qwen image tensors must contain flattened patches and grid_thw")
        if grids.ndim != 2 or grids.shape[1] != 3 or grids.dtype not in (torch.int32, torch.int64):
            raise ValueError("image_grid_thw must be an integer [images, 3] tensor")
        items, offset, token_offset = [], 0, 0
        for index, (temporal, height, width) in enumerate(grids.tolist()):
            if temporal != 1 or min(height, width) <= 0 or height % self.merge_size or width % self.merge_size:
                raise ValueError("Images require T=1 and positive H/W divisible by spatial_merge_size")
            length = height * width
            rows = length // self.merge_size ** 2
            span = positions[token_offset:token_offset + rows]
            if len(span) != rows or span != list(range(span[0], span[0] + rows)):
                raise ValueError("Image token span is truncated or does not match image_grid_thw")
            items.append(EncoderWorkItem(index, 0, length, rows, span[0], rows, offset, offset + length,
                                         {"grid": (temporal, height, width)}))
            offset += length
            token_offset += rows
        if offset != pixels.shape[0] or token_offset != len(positions):
            raise ValueError("Image payload/token count does not match image_grid_thw")
        return tuple(items)

    def _local_inputs(self, batch: Mapping[str, Any], segments: tuple[EncoderSegment, ...]) -> tuple:
        """Reuse native HF position formulas and select local patches before embedding."""
        helpers = importlib.import_module("transformers.vision_utils")
        config = self.vision.config
        parameter = next(self.vision.parameters())
        device, dtype = parameter.device, parameter.dtype
        patch_width = config.in_channels * config.temporal_patch_size * config.patch_size ** 2
        patches, indices, weights, positions = [], [], [], []
        for segment in segments:
            length, valid = segment.local_padded_length, segment.local_valid_length
            patch = torch.zeros(length, patch_width, device=device, dtype=dtype)
            index = torch.zeros(4, length, device=device, dtype=torch.long)
            weight = torch.zeros(4, length, device=device)
            position = torch.zeros(length, 2, device=device, dtype=torch.long)
            if not segment.is_dummy:
                grid = torch.tensor([segment.item.metadata["grid"]], device=device, dtype=torch.long)
                full_index, full_weight = helpers.get_vision_bilinear_indices_and_weights(
                    grid, self.vision.num_grid_per_side, self.merge_size)
                full_position = helpers.get_vision_position_ids(grid, self.merge_size)
                start = segment.local_start
                payload = segment.item.payload_start + start
                patch[:valid] = batch["pixel_values"][payload:payload + valid].to(dtype)
                index[:, :valid] = full_index[:, start:start + valid]
                weight[:, :valid] = full_weight[:, start:start + valid]
                position[:valid] = full_position[start:start + valid]
            patches.append(patch)
            indices.append(index)
            weights.append(weight)
            positions.append(position)
        return torch.cat(patches), torch.cat(indices, 1), torch.cat(weights, 1), torch.cat(positions)

    def encode_local(self, batch: Mapping[str, Any], segments: tuple[EncoderSegment, ...]) -> torch.Tensor:
        """Run each FSDP visual child once, including ranks with only dummy work."""
        patches, indices, weights, positions = self._local_inputs(batch, segments)
        hidden = self.vision.patch_embed(patches)
        positional = (self.vision.pos_embed(indices) * weights.unsqueeze(-1)).sum(0)
        hidden = hidden + positional.to(hidden.dtype)
        rotary = self.vision.rotary_pos_emb(positions).reshape(hidden.shape[0], -1)
        embedding = torch.cat((rotary, rotary), -1)
        position_embeddings = (embedding.cos(), embedding.sin())
        lengths = [0] + [segment.local_padded_length for segment in segments]
        cu_seqlens = torch.tensor(lengths, device=hidden.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        for block in self.vision.blocks:
            hidden = block(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        return hidden

    def project(self, hidden: torch.Tensor, item: EncoderWorkItem | None) -> torch.Tensor:
        """Apply native spatial merging after gathering an image's complete grid."""
        if item is None:
            hidden = hidden[:1].expand(self.merge_size ** 2, -1) * 0.0
        return self.vision.merger(hidden)

    def install_forward(self) -> None:
        """Inject routed image features without replacing the HF decoder forward."""
        if getattr(self.model, "_hp_vlm_cp_enabled", False):
            # The CP forward consumes the same plan but inserts only locally owned image rows.
            return
        original_forward = self.model.forward
        original_visual = self.vision.forward

        @wraps(original_visual)
        def visual_forward(*args: Any, **kwargs: Any) -> Any:
            """Use routed features only within the active outer forward."""
            features = self.feature_context.get()
            if features is None:
                return original_visual(*args, **kwargs)
            return BaseModelOutputWithPooling(pooler_output=features)

        @wraps(original_forward)
        def forward(*args: Any, **kwargs: Any) -> Any:
            """Run encoder work under the outer FSDP call and retain dummy gradients."""
            if "vlm_cp_inputs" in kwargs:
                return original_forward(*args, **kwargs)
            plan = kwargs.pop("encoder_parallel_plan", None)
            if plan is None:
                return original_forward(*args, **kwargs)
            records, anchor = plan.encode_and_route(kwargs)
            features = torch.cat([feature for _, feature in records]) if records else None
            token = self.feature_context.set(features)
            try:
                output = original_forward(*args, **kwargs)
            finally:
                self.feature_context.reset(token)
            if anchor is not None:
                if getattr(output, "loss", None) is None:
                    raise ValueError("Qwen encoder training requires a ModelOutput with loss")
                output.loss = output.loss + anchor
            return output

        self.vision.forward = visual_forward
        self.model.forward = forward


def get_encoder_parallel_spec() -> EncoderParallelAdapterSpec:
    """Return the Qwen3.6 VLM-only visual ownership contract."""
    return EncoderParallelAdapterSpec(("model.visual",), "model.visual.blocks.*.attn",
                                      QwenEncoderAttention, QwenVisionBridge)
