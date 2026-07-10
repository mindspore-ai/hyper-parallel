# Copyright 2026 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Qwen3.5-MoE vision-language composite (``Qwen3_5MoeForConditionalGeneration``).

The model combines a 27-block ViT vision tower (``self.model.visual``) with the
Qwen3.5-MoE text backbone (``self.model.language_model``), injecting image
features via ``masked_scatter`` at image-token positions and using interleaved
3-D mRoPE positions for the spatial layout.

Composition:

- **Text backbone** — the :class:`Qwen3_5MoeTextModel`
  (GatedDeltaNet hybrid + 256-expert MoE) is used unchanged.
- **Vision tower** — the Qwen3.5 ViT is the Qwen3-VL ViT with DeepStack
  removed, so :class:`Qwen3VLMoeVisionModel` is reused with
  ``deepstack_visual_indexes=[]`` (its ``deepstack_merger_list`` becomes an
  empty ``ModuleList`` → no stray params, no checkpoint mismatch). The
  vision tower's NPU-ULP determinism guards stay in that module.
- **Fusion + mRoPE** — :meth:`get_rope_index` / :meth:`get_vision_position_ids`
  keep the checkpoint runtime's 3-D position math, including the video
  (``t > 1``) ordering.

State-dict keys match the checkpoint namespace directly: ``model.visual.*``,
``model.language_model.*``, ``lm_head.weight``.
"""
# pylint: disable=C0103  # Qwen class-name convention (Qwen3_5*)
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.qwen3_5_moe.model import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextModel,
    _normalize_qwen3_5_position_ids,
    _prepare_qwen3_5_attention_masks,
    moe_aux_loss,
)
from hyper_parallel.models.qwen3_5_moe.mtp import Qwen3_5MoeMTP, mtp_loss
from hyper_parallel.models.qwen3_vl_vision import (
    Qwen3VLMoeVisionConfig,
    Qwen3VLMoeVisionModel,
)


def _vl_vision_config() -> Qwen3VLMoeVisionConfig:
    """Qwen3.5-MoE vision defaults: the Qwen3-VL ViT with DeepStack disabled."""
    return Qwen3VLMoeVisionConfig(
        deepstack_visual_indexes=[],
        _attn_implementation="eager",
    )


@dataclass
class Qwen3_5MoeVLConfig:
    """Composite config for Qwen3.5-MoE conditional generation (text + vision).

    ``text_config`` holds the Qwen3.5-MoE text knobs; ``vision_config`` the
    27-block ViT. Multimodal token ids mirror the upstream ``config.json``.
    """

    text_config: Qwen3_5MoeConfig = field(default_factory=Qwen3_5MoeConfig)
    vision_config: Qwen3VLMoeVisionConfig = field(default_factory=_vl_vision_config)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vl: bool = True


class Qwen3_5MoeVLModel(nn.Module):
    """Composite backbone: ``self.visual`` (ViT) + ``self.language_model`` (text).

    The forward fuses image features into the text embeddings, computes 3-D
    mRoPE positions, then runs the Qwen3.5-MoE text layers. It returns the
    final hidden states (the :class:`Qwen3_5MoeVLForConditionalGeneration`
    wrapper applies ``lm_head`` and the loss).
    """

    def __init__(self, config: Qwen3_5MoeVLConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLMoeVisionModel(config.vision_config)
        self.language_model = Qwen3_5MoeTextModel(config.text_config)
        self.rope_deltas: Optional[torch.Tensor] = None
        self.visual_injection_input = nn.Identity()
        self.visual_injection_output = nn.Identity()

    @property
    def layers(self):
        return self.language_model.layers

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute 3-D (T, H, W) vision positions for one image/video.

        The repeat patterns are order-sensitive and must not be changed.
        """
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        position_height = torch.arange(llm_grid_h, device=device) + start_position

        position_width = position_width.repeat(llm_grid_h * llm_grid_t)
        position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
        position_temporal = (
            position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
        )
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build ``(3, B, S)`` mRoPE positions + deltas.

        Walks contiguous runs of ``mm_token_type_ids`` (0=text, 1=image,
        2=video), laying out text runs as monotonic positions and vision runs
        via :meth:`get_vision_position_ids`.
        """
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0,
            )
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        rope_deltas = []
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                keep = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[keep]
                input_token_type = input_token_type[keep]

            input_type_group = []
            for _key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1],
            ):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((_key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    llm_pos_ids_list.append(
                        self.get_vision_position_ids(
                            current_pos, grid_thw, 1, spatial_merge_size,
                            device=input_ids.device,
                        )
                    )
                    current_pos += max(grid_thw[1], grid_thw[2]).item() // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            rope_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        rope_deltas = torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, rope_deltas

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run the ViT and split merged features per image."""
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // (self.visual.spatial_merge_size ** 2)
        ).tolist()
        return torch.split(vision_output.pooler_output, split_sizes)

    def get_video_features(
        self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run the (shared) ViT on video frames and split features per video.

        Qwen3.5 has no separate video encoder — video frames go through the same
        ViT as images (HF ``get_video_features`` just calls ``get_image_features``
        on the video grid; ``video_grid_thw`` carries the temporal ``t``).
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, S, H]`` bool mask of image-token slots; validates token count."""
        special_image_mask = input_ids == self.config.image_token_id
        n_image_tokens = int(special_image_mask.sum())
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={tuple(image_features.shape)}"
            )
        return special_image_mask

    def get_video_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        video_features: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, S, H]`` bool mask of video-token slots; validates token count."""
        special_video_mask = input_ids == self.config.video_token_id
        n_video_tokens = int(special_video_mask.sum())
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens={n_video_tokens}, features={tuple(video_features.shape)}"
            )
        return special_video_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_prenorm: bool = False,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Fuse image features, build 3-D positions, run the text backbone.

        The fused embeddings flow into the text backbone unchanged, matching
        the upstream Transformers residual dtype.
        """
        # pylint: disable=W0613  # interface conformance
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None or pixel_values_videos is not None:
            inputs_embeds = self.visual_injection_input(inputs_embeds)
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_features, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype,
                )
                image_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds, image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if pixel_values_videos is not None:
                video_features = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_features, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype,
                )
                video_mask = self.get_video_placeholder_mask(
                    input_ids, inputs_embeds, video_features=video_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            inputs_embeds = self.visual_injection_output(inputs_embeds)

        if position_ids is None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                mm_token_type_ids[input_ids == self.config.image_token_id] = 1
                mm_token_type_ids[input_ids == self.config.video_token_id] = 2
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_prenorm=return_prenorm,
        )


class Qwen3_5MoeVLForConditionalGeneration(nn.Module):
    """Qwen3.5-MoE multimodal entry point.

    Submodule layout follows the checkpoint namespace::

        model.visual.*               (Qwen3VLMoeVisionModel, deepstack off)
        model.language_model.*        (Qwen3_5MoeTextModel)
        lm_head.weight
    """

    def __init__(self, config: Qwen3_5MoeVLConfig):
        super().__init__()
        self.config = config

        text_config = config.text_config
        rope_dim = int(text_config.head_dim * text_config.partial_rotary_factor)
        if sum(text_config.mrope_section) * 2 != rope_dim:
            raise ValueError(
                f"sum(mrope_section)*2 ({sum(text_config.mrope_section) * 2}) "
                f"must equal rope_dim ({rope_dim} = head_dim * "
                f"partial_rotary_factor)"
            )
        self.model = Qwen3_5MoeVLModel(config)
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False,
        )
        self.mtp = (
            Qwen3_5MoeMTP(text_config, self.model.language_model.rotary_emb)
            if text_config.mtp_loss_weight > 0
            else None
        )
        if text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    @property
    def layers(self):
        return self.model.language_model.layers

    def tie_weights(self) -> None:
        # Re-tie after ``to_empty`` — fresh per-Parameter storage breaks the
        # ``__init__``-time tie.
        if self.config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass."""
        # pylint: disable=W0613  # interface conformance
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            return_prenorm=self.mtp is not None,
        )
        if self.mtp is not None:
            hidden_states, prenorm_hidden = out
        else:
            hidden_states = out
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        if self.mtp is not None:
            next_token_embeds = self.model.language_model.embed_tokens(
                F.pad(input_ids, (0, 1))[..., 1:]
            )
            mtp_hidden = self.mtp(
                prenorm_hidden,
                next_token_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            mtp_logits = self.lm_head(mtp_hidden.to(self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            # Right-pad labels with -100 (instead of slicing logits) so the
            # autograd graph flows through the full ``logits`` tensor.
            logits_fp = logits.float()
            targets = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            num_items_in_batch = kwargs.get("num_items_in_batch")
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = F.cross_entropy(
                logits_fp.view(-1, logits_fp.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            if num_items_in_batch is not None:
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
            text_config = self.config.text_config
            if text_config.output_router_logits:
                aux = moe_aux_loss(
                    self.model.language_model.layers, text_config, attention_mask,
                )
                if aux is not None:
                    loss = loss + aux.to(loss.device)
            if self.mtp is not None:
                loss = loss + text_config.mtp_loss_weight * mtp_loss(mtp_logits, input_ids)
        return {"loss": loss, "logits": logits}


class Qwen3_5MoeVLStageModule(nn.Module):
    """One pipeline-parallel stage of Qwen3.5-MoE VL.

    Stage 0 holds the frozen visual tower and token embedding, injects image
    features, and computes the 3-D mRoPE positions. Later stages receive
    ``(hidden_states, position_ids)``. The last stage owns ``norm`` and
    ``lm_head`` and returns sum-reduced cross-entropy on pre-shifted targets.

    Args:
        layers: This stage's decoder layers.
        config: Composite model config (stage 0 only, else ``None``).
        visual: Vision tower (stage 0 only, else ``None``).
        embed_tokens: Token embedding (stage 0 only, else ``None``).
        norm: Final RMSNorm (last stage only, else ``None``).
        lm_head: Output projection (last stage only, else ``None``).
    """

    def __init__(
        self, layers, config=None, visual=None, embed_tokens=None,
        norm=None, lm_head=None,
    ):
        super().__init__()
        self.config = config
        self.visual = visual
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def get_vision_position_ids(self, *args, **kwargs):
        """Delegate to the composite vision-position helper."""
        return Qwen3_5MoeVLModel.get_vision_position_ids(self, *args, **kwargs)

    @staticmethod
    def _check_media_rows(values: torch.Tensor, grid_thw: torch.Tensor, name: str) -> None:
        """Validate PP micro-batch media rows are aligned with grid metadata."""
        expected_rows = int(grid_thw.prod(-1).sum())
        if expected_rows == values.shape[0]:
            return
        raise NotImplementedError(
            f"Qwen3.5-MoE VL PP micro-batching split {name} "
            f"({values.shape[0]} rows) and {name.replace('pixel_values', 'grid_thw')} "
            f"({expected_rows} grid rows) onto mismatched boundaries. "
            "pp_micro_batch_num>1 requires a sample-uniform VL batch; use "
            "pp_micro_batch_num=1, or pad the batch so every sample has the same "
            "media/patch count."
        )

    def _inject_image_embeds(self, input_ids, inputs_embeds, pixel_values, image_grid_thw):
        """Inject image features into token embeddings."""
        if pixel_values is None:
            return inputs_embeds
        self._check_media_rows(pixel_values, image_grid_thw, "pixel_values")
        image_features = Qwen3_5MoeVLModel.get_image_features(
            self, pixel_values, image_grid_thw,
        )
        image_embeds = torch.cat(image_features, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype,
        )
        image_mask = Qwen3_5MoeVLModel.get_placeholder_mask(
            self, input_ids, inputs_embeds, image_features=image_embeds,
        )
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def _inject_video_embeds(self, input_ids, inputs_embeds, pixel_values_videos, video_grid_thw):
        """Inject video features into token embeddings."""
        if pixel_values_videos is None:
            return inputs_embeds
        self._check_media_rows(pixel_values_videos, video_grid_thw, "pixel_values_videos")
        video_features = Qwen3_5MoeVLModel.get_image_features(
            self, pixel_values_videos, video_grid_thw,
        )
        video_embeds = torch.cat(video_features, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype,
        )
        video_mask = Qwen3_5MoeVLModel.get_video_placeholder_mask(
            self, input_ids, inputs_embeds, video_features=video_embeds,
        )
        return inputs_embeds.masked_scatter(video_mask, video_embeds)

    def _first_stage_hidden(
        self,
        input_ids,
        position_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        pixel_values_videos,
        video_grid_thw,
        mm_token_type_ids,
    ):
        """Build first-stage hidden states and optional mRoPE positions."""
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self._inject_image_embeds(
            input_ids, inputs_embeds, pixel_values, image_grid_thw,
        )
        inputs_embeds = self._inject_video_embeds(
            input_ids, inputs_embeds, pixel_values_videos, video_grid_thw,
        )
        if position_ids is None and (image_grid_thw is not None or video_grid_thw is not None):
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                mm_token_type_ids[input_ids == self.config.image_token_id] = 1
                mm_token_type_ids[input_ids == self.config.video_token_id] = 2
            position_ids, _ = Qwen3_5MoeVLModel.get_rope_index(
                self,
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        return inputs_embeds, position_ids

    def _run_decoder_layers(self, hidden_states, position_ids, attention_mask):
        """Run this stage's decoder layers."""
        seq_len = hidden_states.shape[1]
        linear_attention_mask, causal_attention_mask = _prepare_qwen3_5_attention_masks(
            attention_mask, seq_len, hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=causal_attention_mask,
                linear_attention_mask=linear_attention_mask,
            )
        return hidden_states

    def _final_stage_output(self, hidden_states, targets):
        """Return logits or sum-reduced CE from the last PP stage."""
        if self.norm is None or self.lm_head is None:
            return None
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        if targets is None:
            return logits
        logits_fp = logits.float()
        return F.cross_entropy(
            logits_fp.view(-1, logits_fp.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ):
        """Run this stage and carry 3-D mRoPE positions across stage boundaries."""
        if self.embed_tokens is not None:
            input_ids = hidden_states
            inputs_embeds, position_ids = self._first_stage_hidden(
                input_ids,
                position_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
                mm_token_type_ids,
            )
            hidden_states = inputs_embeds

        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        position_ids = _normalize_qwen3_5_position_ids(
            position_ids, bsz, seq_len, hidden_states.device,
        )
        hidden_states = self._run_decoder_layers(hidden_states, position_ids, attention_mask)
        final_output = self._final_stage_output(hidden_states, targets)
        if final_output is None:
            return hidden_states, position_ids
        return final_output


__all__ = [
    "Qwen3_5MoeVLConfig",
    "Qwen3_5MoeVLModel",
    "Qwen3_5MoeVLForConditionalGeneration",
    "Qwen3_5MoeVLStageModule",
]
