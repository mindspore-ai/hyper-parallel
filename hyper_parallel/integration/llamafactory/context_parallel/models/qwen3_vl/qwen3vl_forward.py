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
"""Qwen3VL MoE multimodal context-parallel runtime patching."""
import sys

import torch
from torch import nn
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel as _Qwen3VLMoeModel,
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeTextAttention as _Qwen3VLMoeTextAttention,
    create_causal_mask,
    is_torchdynamo_compiling,
)

from hyper_parallel import ContextParallel, parallelize_module
from hyper_parallel.core.dtensor.dtensor import DTensor

from ...inputs import get_cp_group, get_cp_group_ranks, get_cp_rank


_QWEN3VL_MOE_TEXT_ATTENTION_CP_CLASSES: dict[type, type] = {}


def is_qwen3vl_moe_model(model: nn.Module) -> bool:
    """Return whether the model tree contains the Qwen3VL-MoE base model."""
    return any(isinstance(module, _Qwen3VLMoeModel) for module in model.modules())


class _ContextParallelFaCore(nn.Module):
    """Runtime flash-attention core wrapped by ``ContextParallel.apply``."""

    def __init__(
        self,
        attention_interface,
        cp_rank: int,
        cp_size: int,
        cp_group=None,
        cp_group_ranks: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.attention_interface = attention_interface
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.cp_group = cp_group
        self.cp_group_ranks = cp_group_ranks

    @staticmethod
    def _to_local_tensor(tensor: torch.Tensor | DTensor) -> torch.Tensor:
        """Convert a DTensor input to its local tensor view."""
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _run_flash_attention(
        self,
        owner_module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        global_position_ids: torch.Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
        cu_seq_lens_q: torch.Tensor | None,
        cu_seq_lens_k: torch.Tensor | None,
        max_length_q: int | None,
        max_length_k: int | None,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        """Run model-native FA and let the backend choose mask/varlen semantics."""
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(device=query.device)
        fa_kwargs = dict(kwargs)
        if isinstance(global_position_ids, torch.Tensor) and "position_ids" not in fa_kwargs:
            fa_kwargs["position_ids"] = global_position_ids.to(device=query.device)
        if cu_seq_lens_q is not None:
            fa_kwargs["cu_seq_lens_q"] = cu_seq_lens_q
        if cu_seq_lens_k is not None:
            fa_kwargs["cu_seq_lens_k"] = cu_seq_lens_k
        if max_length_q is not None:
            fa_kwargs["max_length_q"] = max_length_q
        if max_length_k is not None:
            fa_kwargs["max_length_k"] = max_length_k

        attn_output, _ = self.attention_interface(
            owner_module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout_p,
            scaling=scale,
            is_causal=is_causal,
            **fa_kwargs,
        )
        return attn_output.transpose(1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor | DTensor,
        key: torch.Tensor | DTensor,
        value: torch.Tensor | DTensor,
        *,
        owner_module: nn.Module,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        dropout_p: float = 0.0,
        scale: float | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        cu_seq_lens_k: torch.Tensor | None = None,
        max_length_q: int | None = None,
        max_length_k: int | None = None,
        global_position_ids: torch.Tensor | None = None,
        rotary_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        apply_rotary_pos_emb_func=None,
        **kwargs,
    ) -> torch.Tensor | DTensor:
        """Run FA after ContextParallel has redistributed Q/K/V."""
        local_query = self._to_local_tensor(query)
        local_key = self._to_local_tensor(key)
        local_value = self._to_local_tensor(value)
        if rotary_position_embeddings is not None and apply_rotary_pos_emb_func is not None:
            cos, sin = rotary_position_embeddings
            if cos.shape[-2] != local_query.shape[2] or sin.shape[-2] != local_query.shape[2]:
                raise ValueError(
                    "Context Parallel RoPE position embedding length does not match the CP attention sequence length: "
                    f"cos={cos.shape[-2]}, sin={sin.shape[-2]}, query={local_query.shape[2]}."
                )
            cos = cos.to(device=local_query.device, dtype=local_query.dtype)
            sin = sin.to(device=local_query.device, dtype=local_query.dtype)
            local_query, local_key = apply_rotary_pos_emb_func(local_query, local_key, cos, sin)
        return self._run_flash_attention(
            owner_module,
            local_query,
            local_key,
            local_value,
            global_position_ids,
            is_causal,
            dropout_p,
            scale,
            cu_seq_lens_q,
            cu_seq_lens_k,
            max_length_q,
            max_length_k,
            attention_mask,
            **kwargs,
        )


def _select_local_dense_embeds_by_mask(
    dense_embeds: torch.Tensor,
    global_mask: torch.Tensor,
    seq_start: int,
    seq_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map Qwen3VL mask-ordered visual embeddings to the local CP token range."""
    global_mask = global_mask.to(torch.bool)
    batch_size, seq_len = global_mask.shape
    local_mask = global_mask[:, seq_start:seq_end].contiguous()
    if not local_mask.any():
        return local_mask, dense_embeds[:0]

    local_positions = torch.arange(seq_start, seq_end, device=global_mask.device).unsqueeze(0).expand(batch_size, -1)
    batch_offsets = (torch.arange(batch_size, device=global_mask.device) * seq_len).unsqueeze(1)
    global_positions = (batch_offsets + local_positions)[local_mask]
    dense_indices = global_mask.reshape(-1).to(torch.int64).cumsum(0)[global_positions] - 1
    return local_mask, dense_embeds.index_select(0, dense_indices.to(dense_embeds.device))


def _build_qwen3vl_global_attention_mask(
    model,
    attention_mask,
    text_position_ids: torch.Tensor | None,
    global_inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor | None,
    past_key_values,
):
    """Build the full-sequence CP attention mask using Qwen3VL's native helper."""
    global_seq_len = global_inputs_embeds.shape[1]
    if isinstance(cache_position, torch.Tensor) and cache_position.numel() == global_seq_len:
        global_cache_position = cache_position.to(device=global_inputs_embeds.device)
    else:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        global_cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + global_seq_len,
            device=global_inputs_embeds.device,
        )

    return create_causal_mask(
        config=model.language_model.config,
        input_embeds=global_inputs_embeds,
        attention_mask=attention_mask,
        cache_position=global_cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )


def _select_local_visual_feature_stream(
    *,
    dense_embeds: list[torch.Tensor],
    deepstack_embeds: list[torch.Tensor],
    global_mask: torch.Tensor,
    seq_start: int,
    seq_end: int,
    feature_name: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Select the visual features that correspond to this rank's local text shard."""
    dense_embeds = torch.cat(dense_embeds, dim=0)
    if int(global_mask.sum().item()) != dense_embeds.shape[0]:
        raise ValueError(
            f"{feature_name} features and tokens do not match before CP sharding: "
            f"tokens: {int(global_mask.sum().item())}, features {dense_embeds.shape[0]}"
        )

    local_mask, local_dense_embeds = _select_local_dense_embeds_by_mask(dense_embeds, global_mask, seq_start, seq_end)
    local_deepstack_embeds = [
        _select_local_dense_embeds_by_mask(
            embed,
            global_mask,
            seq_start,
            seq_end,
        )[1]
        for embed in deepstack_embeds
    ]
    if int(local_mask.sum().item()) != local_dense_embeds.shape[0]:
        raise ValueError(
            f"{feature_name} features and local tokens do not match after CP sharding: "
            f"tokens: {int(local_mask.sum().item())}, features {local_dense_embeds.shape[0]}"
        )
    return [local_dense_embeds], local_deepstack_embeds


def _slice_global_position_ids_for_local_cp(position_ids: torch.Tensor | None, seq_start: int, seq_end: int):
    """Slice full-sequence position ids to the local CP shard consumed by the text model."""
    if not isinstance(position_ids, torch.Tensor) or position_ids.size(-1) < seq_end:
        return position_ids
    return position_ids[..., seq_start:seq_end].contiguous()


class Qwen3VLMoeModelForCP(_Qwen3VLMoeModel):
    """Qwen3VL MoE model with CP-aware multimodal feature alignment."""

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        """Run Qwen3VL MoE forward with CP-aware visual feature and text input alignment."""
        global_input_ids = kwargs.pop("_hp_cp_global_input_ids", None)
        global_position_ids = kwargs.pop("_hp_cp_global_position_ids", None)
        local_seq_start = kwargs.pop("_hp_cp_local_seq_start", None)
        local_seq_end = kwargs.pop("_hp_cp_local_seq_end", None)
        if global_input_ids is None or local_seq_start is None or local_seq_end is None:
            # Non-CP calls should keep the exact upstream Qwen3VL MoE behavior.
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Keep this branch close to Qwen3VL's native forward: CP needs full
        # sequence mROPE ids first, then derives the local text-model slice.
        if isinstance(global_position_ids, torch.Tensor):
            position_ids = global_position_ids
        elif not (isinstance(position_ids, torch.Tensor) and position_ids.size(-1) == global_input_ids.size(-1)):
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and global_input_ids.shape[1] != 1
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            rope_deltas = getattr(self, "rope_deltas", None)
            if (prefill_compiled_stage or prefill_noncompiled_stage) or rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    global_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = global_input_ids.shape
                delta = (
                    (cache_position[0] + rope_deltas).to(global_input_ids.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=global_input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)
        local_position_ids = _slice_global_position_ids_for_local_cp(
            position_ids,
            local_seq_start,
            local_seq_end,
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        # Keep raw visual inputs global, then select only the dense visual
        # embeddings whose placeholders fall inside this rank's local text shard.
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds, deepstack_image_embeds = _select_local_visual_feature_stream(
                dense_embeds=image_embeds,
                deepstack_embeds=deepstack_image_embeds,
                global_mask=global_input_ids == self.config.image_token_id,
                seq_start=local_seq_start,
                seq_end=local_seq_end,
                feature_name="Image",
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Video features follow the same mask-aware local selection as images.
        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds, deepstack_video_embeds = _select_local_visual_feature_stream(
                dense_embeds=video_embeds,
                deepstack_embeds=deepstack_video_embeds,
                global_mask=global_input_ids == self.config.video_token_id,
                seq_start=local_seq_start,
                seq_end=local_seq_end,
                feature_name="Video",
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        # Rebuild the visual masks/deepstack tensors exactly as the native
        # forward does, but using local visual masks and local dense embeddings.
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Build full-sequence metadata for CP attention. The language model runs
        # on local hidden states, while TextAttention consumes these global
        # position/mask tensors after ContextParallel all-to-all redistribution.
        global_attention_carrier = inputs_embeds.new_empty(
            inputs_embeds.shape[0],
            global_input_ids.shape[1],
            inputs_embeds.shape[-1],
        )
        # Same split as Qwen3VLMoeTextModel.forward: text positions feed the
        # causal mask, while position_ids itself continues as mROPE ids.
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        global_attention_mask = _build_qwen3vl_global_attention_mask(
            self,
            attention_mask,
            text_position_ids,
            global_attention_carrier,
            cache_position,
            past_key_values,
        )
        global_position_embeddings = self.language_model.rotary_emb(
            global_attention_carrier,
            position_ids,
        )

        # Feed local tensors to the native text model, and pass global metadata
        # through kwargs so only the patched TextAttention consumes it.
        outputs = self.language_model(
            input_ids=None,
            position_ids=local_position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            _hp_cp_global_position_ids=text_position_ids,
            _hp_cp_global_position_embeddings=global_position_embeddings,
            _hp_cp_global_attention_mask=global_attention_mask,
            **kwargs,
        )

        return Qwen3VLMoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


def _apply_qwen3vl_moe_visual_embeds_patch(module: nn.Module) -> bool:
    """Switch a Qwen3VL MoE model instance to the CP-aware subclass."""
    if getattr(module, "_hp_cp_visual_embeds_enabled", False):
        return False
    if not isinstance(module, _Qwen3VLMoeModel):
        return False
    if not all(hasattr(module.config, attr) for attr in ("image_token_id", "video_token_id")):
        return False

    # Keep the existing initialized parameters/modules, but route future calls to
    # the CP subclass instead of binding a standalone runtime forward function.
    module.__class__ = Qwen3VLMoeModelForCP
    module._hp_cp_visual_embeds_enabled = True  # pylint: disable=protected-access
    return True


def _enable_qwen3vl_moe_visual_embeds_patch(model: nn.Module, hp_args) -> None:
    """Enable the Qwen3VL-MoE visual-embeds-to-input-embeds CP patch."""
    if getattr(hp_args, "cp_size", 1) <= 1:
        return
    for module in model.modules():
        _apply_qwen3vl_moe_visual_embeds_patch(module)


def _resolve_cp_mesh(mesh):
    """Return a 1-D CP mesh slice when available."""
    if mesh is None:
        return None
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None) or ()
    if "cp" in mesh_dim_names:
        return mesh["cp"]
    if getattr(mesh, "ndim", 0) == 1:
        return mesh
    return None


def _resolve_qwen3vl_moe_apply_rotary_pos_emb(module: nn.Module):
    """Resolve the native RoPE helper used by this Qwen3VL-MoE TextAttention instance."""
    forward = getattr(module, "_hp_original_forward", module.forward)
    forward_fn = getattr(forward, "__func__", forward)
    globals_dict = getattr(forward_fn, "__globals__", {})
    module_impl = sys.modules.get(module.__class__.__module__)

    apply_rotary_pos_emb_func = globals_dict.get("apply_rotary_pos_emb")
    if apply_rotary_pos_emb_func is None:
        apply_rotary_pos_emb_func = getattr(module_impl, "apply_rotary_pos_emb", None)
    if apply_rotary_pos_emb_func is None:
        raise ValueError(f"{module.__class__.__name__} does not expose apply_rotary_pos_emb.")
    return apply_rotary_pos_emb_func


def _resolve_qwen3vl_moe_attention_interface(module: nn.Module):
    """Resolve the native flash-attention implementation configured for Qwen3VL-MoE TextAttention."""
    forward = getattr(module, "_hp_original_forward", module.forward)
    forward_fn = getattr(forward, "__func__", forward)
    globals_dict = getattr(forward_fn, "__globals__", {})
    module_impl = sys.modules.get(module.__class__.__module__)

    attn_impl = getattr(getattr(module, "config", None), "_attn_implementation", "eager")
    if not str(attn_impl).startswith("flash_attention"):
        raise ValueError(
            "Context Parallel attention currently supports only flash_attention backends. "
            f"Got attention implementation {attn_impl!r}; please enable flash_attention_2 for CP."
        )

    all_attention_functions = globals_dict.get("ALL_ATTENTION_FUNCTIONS")
    if all_attention_functions is None:
        all_attention_functions = getattr(module_impl, "ALL_ATTENTION_FUNCTIONS", None)
    if all_attention_functions is None or attn_impl not in all_attention_functions:
        raise ValueError(f"{module.__class__.__name__} does not expose flash attention implementation {attn_impl!r}.")
    return all_attention_functions[attn_impl]


def _resolve_qwen3vl_moe_fa_dependencies(module: nn.Module):
    """Resolve native RoPE and FA dependencies from the module's original forward."""
    return _resolve_qwen3vl_moe_apply_rotary_pos_emb(module), _resolve_qwen3vl_moe_attention_interface(module)


class Qwen3VLMoeTextAttentionForCP:
    """Qwen3VL-MoE TextAttention forward with CP inserted at the attention core."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Run Qwen3VL-MoE text attention with CP applied around FA."""
        if past_key_values is not None:
            raise ValueError("Context parallel runtime attention does not support KV cache in trainer integration.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        global_position_ids = kwargs.pop("_hp_cp_global_position_ids", None)
        global_position_embeddings = kwargs.pop("_hp_cp_global_position_embeddings", None)
        global_attention_mask = kwargs.pop("_hp_cp_global_attention_mask", None)
        if global_position_embeddings is None:
            raise ValueError(
                "Context Parallel attention requires global position embeddings so RoPE can be applied "
                "after Q/K/V redistribution."
            )

        # ContextParallel redistributes sequence/head shards before FA; RoPE is
        # intentionally delayed until after that redistribution.
        attn_output = self._hp_context_parallel_core_attn(
            query_states,
            key_states,
            value_states,
            owner_module=self,
            attention_mask=global_attention_mask if isinstance(global_attention_mask, torch.Tensor) else attention_mask,
            is_causal=getattr(self, "is_causal", True),
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=getattr(self, "scaling", None),
            global_position_ids=global_position_ids,
            rotary_position_embeddings=global_position_embeddings,
            apply_rotary_pos_emb_func=self._hp_apply_rotary_pos_emb,
            **kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def _get_qwen3vl_moe_text_attention_cp_class(base_cls: type) -> type:
    """Return a CP subclass for the model-native Qwen3VL-MoE TextAttention cell."""
    cached_cls = _QWEN3VL_MOE_TEXT_ATTENTION_CP_CLASSES.get(base_cls)
    if cached_cls is not None:
        return cached_cls

    cp_cls = type(
        f"{base_cls.__name__}ForCP",
        (Qwen3VLMoeTextAttentionForCP, base_cls),
        {
            "__doc__": "Qwen3VL-MoE TextAttention with ContextParallel applied to the FA core.",
            "__module__": base_cls.__module__,
        },
    )
    _QWEN3VL_MOE_TEXT_ATTENTION_CP_CLASSES[base_cls] = cp_cls
    return cp_cls


def _supports_qwen3vl_moe_text_attention(module: nn.Module) -> bool:
    """Return whether a module is the Qwen3VL-MoE text attention currently supported by CP."""
    required_attrs = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "head_dim", "config")
    if not all(hasattr(module, attr) for attr in required_attrs):
        return False
    return isinstance(module, _Qwen3VLMoeTextAttention)


def _apply_qwen3vl_moe_attention_patch(
    module: nn.Module,
    cp_mesh,
    cp_rank: int,
    cp_size: int,
    cp_group=None,
    cp_group_ranks: tuple[int, ...] | None = None,
) -> bool:
    """Attach a CP-aware core to a Qwen3VL-MoE text attention module."""
    if getattr(module, "_hp_cp_attention_enabled", False):
        return False
    if not _supports_qwen3vl_moe_text_attention(module):
        return False

    original_cls = module.__class__
    module._hp_original_class = original_cls  # pylint: disable=protected-access
    module._hp_original_forward = module.forward  # pylint: disable=protected-access
    apply_rotary_pos_emb_func, attention_interface = _resolve_qwen3vl_moe_fa_dependencies(module)
    module._hp_apply_rotary_pos_emb = apply_rotary_pos_emb_func  # pylint: disable=protected-access
    module.add_module(
        "_hp_context_parallel_core_attn",
        _ContextParallelFaCore(
            attention_interface=attention_interface,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_group=cp_group,
            cp_group_ranks=cp_group_ranks,
        ),
    )
    parallelize_module(
        module,
        cp_mesh,
        {"_hp_context_parallel_core_attn": ContextParallel(seq_dim=2, head_dim=1, ulysses_degree=None)},
    )
    module.__class__ = _get_qwen3vl_moe_text_attention_cp_class(original_cls)
    module._hp_cp_attention_enabled = True  # pylint: disable=protected-access
    return True


def _enable_qwen3vl_moe_attention_patch(model: nn.Module, mesh, hp_args) -> None:
    """Enable Qwen3VL-MoE TextAttention CP at runtime without modifying model source files."""
    cp_size = getattr(hp_args, "cp_size", 1)
    if cp_size <= 1:
        return

    cp_mesh = _resolve_cp_mesh(mesh)
    if cp_mesh is None:
        raise ValueError("Context parallel requires a CP mesh slice, but none was available.")

    cp_rank = get_cp_rank(hp_args)
    cp_group = get_cp_group(hp_args)
    cp_group_ranks = get_cp_group_ranks(hp_args)
    text_candidates = 0
    enabled = 0
    for _, module in model.named_modules():
        text_candidates += int(_supports_qwen3vl_moe_text_attention(module))
        enabled += int(
            _apply_qwen3vl_moe_attention_patch(
                module,
                cp_mesh,
                cp_rank,
                cp_size,
                cp_group=cp_group,
                cp_group_ranks=cp_group_ranks,
            )
        )

    if enabled == 0 and text_candidates > 0:
        raise ValueError(
            "Context parallel did not find any runtime-adaptable attention modules. "
            "Expected Qwen3VLMoeTextAttention with q_proj/k_proj/v_proj/o_proj and rotary attention."
        )
