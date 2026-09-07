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
"""Transformers DeepSeek-V3 adapter with narrowly scoped vLLM leaves."""

from collections.abc import Iterable
from copy import deepcopy
import re
from typing import Any, Optional, Union

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

from rl.roles.rollout.vllm_moe import HyperExpertPlacement, HyperLocalFusedExperts, routed_forward
from rl.roles.rollout.vllm_moe_parallel import build_moe_tp_plan, pad_moe_tp_tokens
from rl.roles.rollout.vllm_qwen3 import _device_mesh_from_vllm_tp, _load_parameter

from hyper_parallel.auto_models.components.distributed import apply_sharding_plan
from hyper_parallel.auto_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.core.dtensor.layout import Layout, infer_slice_area_by_layout
from hyper_parallel.core.dtensor.placement_types import Replicate


_EXPERT_PROJECTION_PATTERN = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)\."
    r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _join_prefix(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _config_value(config: object, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _validate_adapter_config(vllm_config: VllmConfig) -> None:
    """Validate the supported Moonlight checkpoint and parallel contract."""
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    parallel_config = vllm_config.parallel_config
    if getattr(hf_config, "model_type", None) != "deepseek_v3":
        raise ValueError("HyperDeepseekV3ForCausalLM requires model_type='deepseek_v3'")
    if model_config.dtype != torch.bfloat16:
        raise ValueError("HyperDeepseekV3ForCausalLM currently supports only bfloat16")
    if getattr(hf_config, "q_lora_rank", None) is not None:
        raise ValueError("Moonlight requires the q_lora_rank=None MLA path")
    tp_size = int(parallel_config.tensor_parallel_size)
    if tp_size != 1 and not (
        tp_size == 2 and int(parallel_config.data_parallel_size) == 2
        and bool(parallel_config.enable_expert_parallel)
    ):
        raise ValueError("HyperDeepseekV3ForCausalLM supports TP1 or colocated DP2/TP2/EP4")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("HyperDeepseekV3ForCausalLM currently supports PP1")
    if _config_value(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("HyperDeepseekV3ForCausalLM currently supports prefill CP1")
    if _config_value(parallel_config, "decode_context_parallel_size", 1) != 1:
        raise ValueError("HyperDeepseekV3ForCausalLM currently supports decode CP1")
    if bool(_config_value(parallel_config, "enable_eplb", False)):
        raise ValueError("HyperDeepseekV3ForCausalLM requires EPLB disabled")
    if vllm_config.quant_config is not None:
        raise ValueError("HyperDeepseekV3ForCausalLM requires an unquantized checkpoint")
    if not bool(model_config.use_mla):
        raise ValueError("HyperDeepseekV3ForCausalLM requires vLLM absorbed MLA")
    if float(_config_value(hf_config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("HyperDeepseekV3ForCausalLM does not support attention dropout")


class _VLLMDeepseekV3MLA(DeepseekV2MLAAttention):
    """Retain vLLM's absorbed MLA solely for its latent paged-KV contract.

    The leaf includes the projections and RoPE because vLLM derives absorbed
    decode weights from them during post-load processing. Splitting those
    tensors away from the cache-aware leaf would make refit and decode state
    disagree.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        """Build the cache-aware MLA leaf from the checkpoint configuration."""
        config = deepcopy(vllm_config.model_config.hf_config)
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=int(config.hidden_size),
            num_heads=int(config.num_attention_heads),
            qk_nope_head_dim=int(config.qk_nope_head_dim),
            qk_rope_head_dim=int(config.qk_rope_head_dim),
            v_head_dim=int(config.v_head_dim),
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=int(config.kv_lora_rank),
            max_position_embeddings=int(
                getattr(config, "max_position_embeddings", 8192)
            ),
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[object] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        """Adapt the Transformers decoder call to vLLM packed-token MLA."""
        del position_embeddings
        if attention_mask is not None:
            raise ValueError("vLLM MLA manages causal masks; explicit masks are unsupported")
        if past_key_values is not None:
            raise ValueError("vLLM MLA owns KV state; Transformers caches are unsupported")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("DeepSeek-v3 MLA expects packed hidden states with shape [1,T,H]")
        positions = kwargs.pop("position_ids", None)
        kwargs.pop("use_cache", None)
        if kwargs:
            raise ValueError(f"Unsupported DeepSeek-v3 MLA arguments: {sorted(kwargs)}")
        if positions is None:
            raise ValueError("DeepSeek-v3 MLA requires explicit packed-token positions")
        positions = _normalize_positions(positions, hidden_states.shape[1]).squeeze(0)
        output = super().forward(
            positions=positions,
            hidden_states=hidden_states.squeeze(0),
            llama_4_scaling=None,
        )
        return output.unsqueeze(0), None


class _VLLMDeepseekV3RoutedExperts(nn.Module):
    """Keep Transformers MoE semantics around an Ascend FusedMoE leaf.

    Transformers owns the router linear, correction-bias buffer and shared
    experts.  The common vLLM leaf owns only local expert storage/execution.
    In EP1 every rollout DP worker holds the complete expert set and this
    module performs no collective.
    """

    def __init__(
        self,
        moe: DeepseekV3MoE,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        """Preserve the HF router and shared experts around local fused storage."""
        super().__init__()
        config = moe.config
        self.config = config
        self.gate = moe.gate
        self.shared_experts = moe.shared_experts
        placement = HyperExpertPlacement.from_config(vllm_config, int(config.n_routed_experts))
        self.hyper_ep_mesh = placement.mesh
        self.experts = HyperLocalFusedExperts(
            local_expert_count=placement.local_count,
            global_expert_start=placement.global_start,
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            params_dtype=vllm_config.model_config.dtype,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the HF router/shared expert and the vLLM routed-expert leaf."""
        routed = routed_forward(
            self,
            hidden_states,
            router_fn=MOE_ROUTER_ADAPTERS["deepseekv3"],
        )
        return routed + self.shared_experts(hidden_states)


def _normalize_positions(positions: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if positions.ndim == 1 and positions.shape[0] == num_tokens:
        return positions.unsqueeze(0)
    if positions.ndim == 2 and positions.shape == (1, num_tokens):
        return positions
    raise ValueError("DeepSeek-v3 positions must have shape [T] or [1,T]")


def _checkpoint_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return learned parameters plus the persistent router correction bias."""
    tensors = dict(model.named_parameters())
    for name, buffer in model.named_buffers():
        if not name.endswith(".mlp.gate.e_score_correction_bias"):
            continue
        if name in tensors:
            raise RuntimeError(f"Duplicate DeepSeek-v3 checkpoint tensor {name!r}")
        tensors[name] = buffer
    return tensors


def _map_weight_name(name: str) -> Optional[str]:
    if name.endswith("rotary_emb.inv_freq"):
        return None
    return name


def _validate_public_mla_layout(model, placements, global_shapes, tp_mesh) -> None:
    """Require the cache-owned MLA storage to match the public Trainer TP slices."""
    parameters = dict(model.named_parameters())
    for name, axes in placements.items():
        shape = global_shapes[name]
        layout = Layout.from_device_mesh(tp_mesh)(list(axes))
        layout.placement_to_tensor_map(len(shape))
        region = infer_slice_area_by_layout(layout, tp_mesh.get_local_rank(), shape)
        expected = tuple(end - start for start, end in region)
        actual = tuple(parameters[name].shape)
        if actual != expected:
            raise ValueError(f"MLA leaf violates public TP storage contract: {name}: {actual} != {expected}")


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "inputs_embeds": 0,
    }
)
class HyperDeepseekV3ForCausalLM(DeepseekV3ForCausalLM):
    """Run the Transformers DeepSeek-V3 outer model with two vLLM leaves."""

    is_text_generation_model = True
    supports_pp = False
    supports_multimodal = False
    hyper_component_ownership = {
        "outer_model": "Transformers DeepseekV3ForCausalLM",
        "decoder_residual_norm": "Transformers DeepseekV3DecoderLayer",
        "embedding_dense_mlp_router_shared_lm_head": "Transformers DeepSeek-V3",
        "absorbed_mla": (
            "vLLM: latent paged KV cache, prefill/decode split, and derived "
            "absorbed-weight refresh"
        ),
        "routed_experts": (
            "common local vLLM FusedMoE: Ascend fused/grouped expert kernel "
            "and stable physical w13/w2 storage; no internal communication"
        ),
        "expert_dispatch": "Hyper ep_routed_forward over vLLM-owned EP group; HF router/shared branch",
        "tensor_parallel": (
            "Hyper public planner/apply for outer model; cache-owned MLA projection handles and local-head "
            "derived weights stay together, with physical slices checked against the public Trainer plan"
        ),
        "shared_expert_boundary": "Hyper public SP gather/scatter nested inside the common MoE TP bridge",
        "scheduler_cache_lifecycle": "vLLM: request scheduling and paged-cache ownership",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Build the HF outer model without materializing replaced experts."""
        _validate_adapter_config(vllm_config)
        config = deepcopy(vllm_config.model_config.hf_config)
        # The plugin alias selects this adapter, not a new public model family.
        # Keep canonical identity for the Trainer's existing MLA planner rules.
        config.architectures = ["DeepseekV3ForCausalLM"]
        with torch.device("meta"):
            super().__init__(config)
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        tp_plan = None
        self._tp_mesh = None
        self._tp_placements = {}
        mla_placements = {}
        mla_shapes = {}
        if tp_size > 1:
            mla_shapes = {name: tuple(parameter.shape) for name, parameter in self.named_parameters()
                          if ".self_attn." in name}
            group = get_ep_group()
            metadata_mesh = Layout(
                (group.world_size // tp_size, tp_size), ("dp", "tp"),
                rank_list=group.ranks, init_backend=False,
            ).mesh
            tp_plan = build_moe_tp_plan(self, metadata_mesh, tp_size=tp_size, ep_size=group.world_size)
            for name, spec in list(tp_plan.modules.items()):
                if ".self_attn" not in name:
                    continue
                mla_placements.update({
                    f"{name}.{parameter}": (axes.get("tp", Replicate()),)
                    for parameter, axes in spec.params.items()
                })
                # The absorbed cache-aware leaf materializes its own TP slices.
                # Keep the public layout for synchronization, never shard twice.
                del tp_plan.modules[name]
        for layer in self.model.layers:
            layer.self_attn = None
            if isinstance(layer.mlp, DeepseekV3MoE):
                layer.mlp.experts = None
        self.to_empty(device=vllm_config.device_config.device)

        for layer_idx, layer in enumerate(self.model.layers):
            attention_prefix = _join_prefix(
                prefix,
                f"model.layers.{layer_idx}.self_attn",
            )
            layer.self_attn = _VLLMDeepseekV3MLA(
                vllm_config=vllm_config,
                prefix=attention_prefix,
            )
            if isinstance(layer.mlp, DeepseekV3MoE):
                correction_bias = layer.mlp.gate.e_score_correction_bias
                layer.mlp.gate.e_score_correction_bias = torch.empty(
                    correction_bias.shape,
                    dtype=torch.float32,
                    device=vllm_config.device_config.device,
                )
                moe_prefix = _join_prefix(prefix, f"model.layers.{layer_idx}.mlp")
                layer.mlp = _VLLMDeepseekV3RoutedExperts(
                    layer.mlp,
                    vllm_config=vllm_config,
                    prefix=moe_prefix,
                )
        if tp_plan is not None:
            self._tp_mesh = _device_mesh_from_vllm_tp()
            _validate_public_mla_layout(self, mla_placements, mla_shapes, self._tp_mesh)
            _, layout_info = apply_sharding_plan(self, tp_plan, self._tp_mesh, validate_mode=False)
            if layout_info is None:
                raise RuntimeError("Hyper MLA outer TP application returned no source layouts")
            self._tp_placements = {name: tuple(axes) for name, (axes, _) in layout_info.items()}
            self._tp_placements.update(mla_placements)
            for index, layer in enumerate(self.model.layers):
                if isinstance(layer.mlp, _VLLMDeepseekV3RoutedExperts):
                    base = f"model.layers.{index}.mlp.gate"
                    self._tp_placements[f"{base}.e_score_correction_bias"] = self._tp_placements[f"{base}.weight"]
                    pad_moe_tp_tokens(layer.mlp, tp_size)
        self._expert_loaders = {
            f"model.layers.{layer_idx}.mlp.experts": layer.mlp.experts
            for layer_idx, layer in enumerate(self.model.layers)
            if isinstance(layer.mlp, _VLLMDeepseekV3RoutedExperts)
        }

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply the Transformers DeepSeek-V3 token embedding."""
        return self.model.embed_tokens(input_ids)

    def get_input_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[nn.Module, torch.Tensor]:
        """Preserve the HF accessor and support vLLM's embedding call."""
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
        """Run packed tokens through HF decoder layers and vLLM MLA leaves."""
        intermediate_tensors = kwargs.pop("intermediate_tensors", None)
        if intermediate_tensors is not None:
            raise ValueError("HyperDeepseekV3ForCausalLM does not support pipeline state")
        if kwargs:
            raise ValueError(f"Unsupported DeepSeek-v3 model arguments: {sorted(kwargs)}")
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
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=None,
            )
        return self.model.norm(hidden_states).squeeze(0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute full-vocabulary logits with the Transformers LM head."""
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strictly load checkpoint tensors into HF or vLLM-owned leaves."""
        tensors = _checkpoint_tensors(self)
        expected_tensors = {
            name
            for name in tensors
            if not name.endswith((".experts.w13_weight", ".experts.w2_weight"))
        }
        expected_tensors.update(
            f"{base}.{projection}"
            for base in self._expert_loaders
            for projection in ("gate_up_proj", "down_proj")
        )
        checkpoint_expert_components: dict[str, set[tuple[int, str]]] = {}
        loaded_sources = set()
        loaded_tensors = set()
        tied_embedding_weight = None

        for source_name, loaded_weight in weights:
            target_name = _map_weight_name(source_name)
            if target_name is None:
                continue
            if source_name in loaded_sources:
                raise ValueError(f"Duplicate DeepSeek-v3 checkpoint tensor {source_name!r}")
            loaded_sources.add(source_name)

            packed_projection = next(
                (
                    (base, projection)
                    for base in self._expert_loaders
                    for projection in ("gate_up_proj", "down_proj")
                    if source_name == f"{base}.{projection}"
                ),
                None,
            )
            if packed_projection is not None:
                expert_base, projection = packed_projection
                if source_name in loaded_tensors or source_name in checkpoint_expert_components:
                    raise ValueError(f"Duplicate DeepSeek-v3 checkpoint tensor {source_name!r}")
                experts = self._expert_loaders[expert_base]
                if loaded_weight.shape[0] != int(self.config.n_routed_experts):
                    raise ValueError(
                        "Packed DeepSeek-v3 weights must retain the complete expert axis: "
                        f"name={source_name!r}, expected={self.config.n_routed_experts}, "
                        f"actual={loaded_weight.shape[0]}"
                    )
                (experts.load_gate_up if projection == "gate_up_proj" else experts.load_down)(
                    loaded_weight
                )
                loaded_tensors.add(source_name)
                continue

            expert_match = _EXPERT_PROJECTION_PATTERN.match(source_name)
            if expert_match is not None:
                expert_base, global_expert_text, projection = expert_match.groups()
                global_expert = int(global_expert_text)
                if not 0 <= global_expert < int(self.config.n_routed_experts):
                    raise ValueError(
                        f"DeepSeek-v3 checkpoint expert index is out of range: {source_name!r}"
                    )
                experts = self._expert_loaders.get(expert_base)
                if experts is None:
                    raise ValueError(f"Unexpected DeepSeek-v3 checkpoint tensor {source_name!r}")
                logical_name = (
                    f"{expert_base}.down_proj"
                    if projection == "down_proj"
                    else f"{expert_base}.gate_up_proj"
                )
                if logical_name in loaded_tensors:
                    raise ValueError(
                        f"Duplicate DeepSeek-v3 packed/per-expert tensor {source_name!r}"
                    )
                component = (global_expert, projection)
                components = checkpoint_expert_components.setdefault(logical_name, set())
                if component in components:
                    raise ValueError(f"Duplicate DeepSeek-v3 checkpoint tensor {source_name!r}")
                components.add(component)
                experts.load_expert_projection(loaded_weight, projection, global_expert)
                continue

            if (
                target_name == "lm_head.weight"
                and target_name not in tensors
                and self.config.tie_word_embeddings
            ):
                continue
            parameter = tensors.get(target_name)
            if parameter is None:
                raise ValueError(f"Unexpected DeepSeek-v3 checkpoint tensor {source_name!r}")
            if ".self_attn." in target_name:
                weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
                weight_loader(parameter, loaded_weight)
            else:
                _load_parameter(
                    parameter, loaded_weight,
                    tp_mesh=getattr(self, "_tp_mesh", None),
                    placements=getattr(self, "_tp_placements", {}).get(target_name),
                )
            loaded_tensors.add(target_name)
            if target_name == "model.embed_tokens.weight":
                tied_embedding_weight = loaded_weight

        if (
            self.config.tie_word_embeddings
            and "lm_head.weight" in tensors
            and "lm_head.weight" not in loaded_tensors
            and tied_embedding_weight is not None
        ):
            _load_parameter(
                tensors["lm_head.weight"], tied_embedding_weight,
                tp_mesh=getattr(self, "_tp_mesh", None),
                placements=getattr(self, "_tp_placements", {}).get("lm_head.weight"),
            )
            loaded_tensors.add("lm_head.weight")

        for logical_name, components in checkpoint_expert_components.items():
            projections_per_expert = 1 if logical_name.endswith("down_proj") else 2
            expected = int(self.config.n_routed_experts) * projections_per_expert
            if len(components) != expected:
                raise ValueError(
                    "Incomplete DeepSeek-v3 per-expert checkpoint tensor: "
                    f"name={logical_name!r}, expected_components={expected}, "
                    f"actual_components={len(components)}"
                )
            loaded_tensors.add(logical_name)
        missing_tensors = expected_tensors.difference(loaded_tensors)
        if missing_tensors:
            raise ValueError(
                "Incomplete DeepSeek-v3 checkpoint; missing tensors: "
                + ", ".join(sorted(missing_tensors))
            )
        return loaded_tensors


__all__ = ["HyperDeepseekV3ForCausalLM"]
