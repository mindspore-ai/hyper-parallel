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
"""Transformers Qwen3 adapter for the Hyper-vLLM runtime."""

from collections.abc import Iterable
from typing import Any, Optional, Union

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    apply_rotary_pos_emb,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from hyper_parallel.auto_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import (
    validate_model_compatibility,
)
from hyper_parallel import DeviceMesh, distribute_tensor, mark_created_groups


def _join_prefix(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _config_value(config: object, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _validate_adapter_config(vllm_config: VllmConfig) -> None:
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    parallel_config = vllm_config.parallel_config

    if getattr(hf_config, "model_type", None) != "qwen3":
        raise ValueError("HyperQwen3ForCausalLM requires model_type='qwen3'")
    if model_config.dtype != torch.bfloat16:
        raise ValueError("HyperQwen3ForCausalLM currently supports only bfloat16")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("HyperQwen3ForCausalLM currently supports pipeline_parallel_size=1")
    if _config_value(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3ForCausalLM currently supports prefill_context_parallel_size=1")
    if _config_value(parallel_config, "decode_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3ForCausalLM currently supports decode_context_parallel_size=1")
    if vllm_config.quant_config is not None:
        raise ValueError("HyperQwen3ForCausalLM requires an unquantized checkpoint")
    if not bool(_config_value(hf_config, "is_causal", True)):
        raise ValueError("HyperQwen3ForCausalLM requires causal attention")
    if float(_config_value(hf_config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("HyperQwen3ForCausalLM does not support attention dropout")


class _VLLMQwen3Attention(nn.Module):
    """Keep Transformers Qwen3 projections around vLLM paged attention."""

    def __init__(
        self,
        attention: Qwen3Attention,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        """Replace Qwen3 attention compute with a vLLM paged-attention leaf."""
        super().__init__()
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.head_dim = attention.head_dim
        self.num_heads = attention.config.num_attention_heads
        self.num_key_value_heads = attention.config.num_key_value_heads
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.scaling = attention.scaling
        self.layer_idx = attention.layer_idx
        self.attention = Attention(
            num_heads=self.num_heads // tp_size,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_key_value_heads // tp_size,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=attention.sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        """Run Transformers projections and RoPE with vLLM-owned KV state."""
        del kwargs
        if attention_mask is not None:
            raise ValueError("vLLM manages causal masks; explicit attention_mask is unsupported")
        if past_key_values is not None:
            raise ValueError("vLLM owns KV cache state; Transformers past_key_values is unsupported")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("HyperQwen3ForCausalLM expects packed hidden states with shape [1,T,H]")

        batch_size, num_tokens, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query = query.transpose(1, 2).reshape(num_tokens, -1)
        key = key.transpose(1, 2).reshape(num_tokens, -1)
        value = value.transpose(1, 2).reshape(num_tokens, -1)
        output = self.attention(query, key, value)
        output = output.view(batch_size, num_tokens, -1)
        return self.o_proj(output), None


def _normalize_positions(positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if positions.ndim == 1 and positions.shape[0] == num_tokens:
        return positions.unsqueeze(0)
    if positions.ndim == 2 and positions.shape == (1, num_tokens):
        return positions
    raise ValueError("Qwen3 positions must have shape [T] or [1,T]")


def _map_weight_name(name: str) -> Optional[str]:
    if name.endswith("rotary_emb.inv_freq"):
        return None
    return name


def _device_mesh_from_vllm_tp() -> DeviceMesh:
    """Build a Hyper mesh view over vLLM's existing TP process group."""
    tp_group = get_tp_group()
    process_group = tp_group.device_group
    mark_created_groups(process_group)
    mesh = DeviceMesh.from_group(
        process_group,
        device_type="npu",
        mesh_dim_names=("tp",),
    )
    if mesh.get_group() is not process_group:
        raise RuntimeError("Hyper Qwen3 TP mesh did not retain vLLM's process group")
    if tuple(mesh.rank_list) != tuple(tp_group.ranks):
        raise RuntimeError(
            f"Hyper TP mesh ranks {mesh.rank_list} differ from vLLM TP ranks "
            f"{tuple(tp_group.ranks)}"
        )
    return mesh


def _load_parameter(
    parameter: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    tp_mesh: Optional[DeviceMesh] = None,
    placements: Optional[tuple[object, ...]] = None,
) -> None:
    """Load a full Actor/checkpoint tensor into a replicated or TP-local parameter."""
    if tp_mesh is None or placements is None:
        default_weight_loader(parameter, loaded_weight)
        return
    local_weight = distribute_tensor(
        loaded_weight,
        tp_mesh,
        placements,
        src_data_rank=None,
    ).to_local()
    if tuple(parameter.shape) != tuple(local_weight.shape):
        raise ValueError(
            "Qwen3 TP checkpoint shard shape mismatch: "
            f"parameter={tuple(parameter.shape)}, weight={tuple(local_weight.shape)}"
        )
    with torch.no_grad():
        parameter.copy_(
            local_weight.to(device=parameter.device, dtype=parameter.dtype)
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "inputs_embeds": 0,
    }
)


class HyperQwen3ForCausalLM(Qwen3ForCausalLM):
    """Run the Transformers Qwen3 model with Hyper-vLLM runtime leaves."""

    is_text_generation_model = True
    supports_pp = False
    supports_multimodal = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Build and TP-shard the Transformers Qwen3 model for Hyper-vLLM."""
        _validate_adapter_config(vllm_config)
        super().__init__(vllm_config.model_config.hf_config)
        with torch.device("cpu"):
            canonical_rotary = type(self.model.rotary_emb)(self.config)
        rotary_device = self.model.rotary_emb.inv_freq.device
        self.model.rotary_emb.inv_freq = canonical_rotary.inv_freq.to(rotary_device)
        self.model.rotary_emb.original_inv_freq = canonical_rotary.original_inv_freq.to(
            rotary_device
        )
        for layer_idx, layer in enumerate(self.model.layers):
            layer_prefix = _join_prefix(prefix, f"model.layers.{layer_idx}.self_attn")
            layer.self_attn = _VLLMQwen3Attention(
                layer.self_attn,
                vllm_config=vllm_config,
                prefix=layer_prefix,
            )
        self._tp_mesh: Optional[DeviceMesh] = None
        self._tp_placements: dict[str, tuple[object, ...]] = {}
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        if tp_size > 1:
            validate_model_compatibility(self, tp_size=tp_size)
            self._tp_mesh = _device_mesh_from_vllm_tp()
            plan = ShardingPlanner().plan(
                self,
                self._tp_mesh,
                tp_size=tp_size,
                cp_size=1,
                ep_size=1,
                sequence_parallel=False,
                loss_parallel=False,
            )
            _, tp_layout_info = apply_sharding_plan(
                self,
                plan,
                self._tp_mesh,
                validate_mode=False,
            )
            if tp_layout_info is None:
                raise RuntimeError("Qwen3 TP sharding returned no source-layout metadata")
            self._tp_placements = {
                name: tuple(placements)
                for name, (placements, _source_mesh) in tp_layout_info.items()
            }
            if self.config.tie_word_embeddings:
                embedding_placements = self._tp_placements.get(
                    "model.embed_tokens.weight"
                )
                lm_head_placements = self._tp_placements.get("lm_head.weight")
                if embedding_placements is not None:
                    self._tp_placements.setdefault(
                        "lm_head.weight", embedding_placements
                    )
                elif lm_head_placements is not None:
                    self._tp_placements.setdefault(
                        "model.embed_tokens.weight", lm_head_placements
                    )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply the Transformers Qwen3 token embedding."""
        return self.model.embed_tokens(input_ids)

    def get_input_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[nn.Module, torch.Tensor]:
        """Preserve the HF accessor and support vLLM's legacy embedding call."""
        if input_ids is None:
            return self.model.embed_tokens
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run packed tokens through Transformers layers and vLLM attention."""
        del kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        if hidden_states.ndim != 2:
            raise ValueError("packed input embeddings must have shape [T,H]")

        position_ids = _normalize_positions(positions, hidden_states.shape[0])
        hidden_states = hidden_states.unsqueeze(0)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
        hidden_states = self.model.norm(hidden_states).squeeze(0)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute full-vocabulary logits with the Transformers LM head."""
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strictly load streamed Qwen3 checkpoint or online Actor weights."""
        parameters = dict(self.named_parameters())
        loaded_parameters = set()
        tied_embedding_weight = None

        for source_name, loaded_weight in weights:
            target_name = _map_weight_name(source_name)
            if target_name is None:
                continue
            if (
                target_name == "lm_head.weight"
                and target_name not in parameters
                and self.config.tie_word_embeddings
            ):
                continue
            if target_name not in parameters:
                raise ValueError(
                    f"Unexpected Qwen3 checkpoint parameter '{source_name}'"
                )
            if target_name in loaded_parameters:
                raise ValueError(f"Duplicate Qwen3 checkpoint parameter '{source_name}'")
            _load_parameter(
                parameters[target_name],
                loaded_weight,
                tp_mesh=self._tp_mesh,
                placements=self._tp_placements.get(target_name),
            )
            loaded_parameters.add(target_name)
            if target_name == "model.embed_tokens.weight":
                tied_embedding_weight = loaded_weight

        if (
            self.config.tie_word_embeddings
            and "lm_head.weight" in parameters
            and "lm_head.weight" not in loaded_parameters
            and tied_embedding_weight is not None
        ):
            _load_parameter(
                parameters["lm_head.weight"],
                tied_embedding_weight,
                tp_mesh=self._tp_mesh,
                placements=self._tp_placements.get("lm_head.weight"),
            )
            loaded_parameters.add("lm_head.weight")

        missing_parameters = set(parameters).difference(loaded_parameters)
        if missing_parameters:
            raise ValueError(
                "Incomplete Qwen3 checkpoint; missing parameters: "
                + ", ".join(sorted(missing_parameters))
            )
        return loaded_parameters


__all__ = ["HyperQwen3ForCausalLM"]
