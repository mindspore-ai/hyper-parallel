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
"""Official Qwen3-30B-A3B adapter with Hyper-owned expert dispatch."""

import re
from collections.abc import Iterable
from types import MethodType
from typing import Optional, Union

import torch  # pylint: disable=forbidden-backend-import
from rl.roles.model import (
    QWEN3_30B_A3B_CONFIG,
)
from rl.roles.rollout.vllm_moe import HyperExpertPlacement, HyperLocalFusedExperts, qwen3_combine, routed_forward
from rl.roles.rollout.vllm_moe_parallel import build_moe_tp_plan, pad_moe_tp_tokens
from rl.roles.rollout.vllm_qwen3 import _device_mesh_from_vllm_tp, _load_parameter
from rl.roles.rollout.vllm_qwen3_common import (
    Qwen3PagedAttention,
    config_value,
    join_prefix,
    normalize_positions,
)
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.models.interfaces import MixtureOfExperts

from hyper_parallel.auto_models.components.distributed import apply_sharding_plan
from hyper_parallel.auto_models.components.distributed.ep_utils import MOE_ROUTER_ADAPTERS
from hyper_parallel.core.dtensor.layout import Layout

_EXPERT_PROJECTION_PATTERN = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)\."
    r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _validate_adapter_config(vllm_config: VllmConfig) -> None:
    """Validate the supported Qwen3-30B-A3B checkpoint and parallel contract."""
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    parallel_config = vllm_config.parallel_config
    if getattr(hf_config, "model_type", None) != "qwen3_moe":
        raise ValueError("HyperQwen3MoeForCausalLM requires model_type='qwen3_moe'")
    mismatches = {
        name: (expected, getattr(hf_config, name, None))
        for name, expected in QWEN3_30B_A3B_CONFIG
        if getattr(hf_config, name, None) != expected
    }
    if mismatches:
        raise ValueError(
            "HyperQwen3MoeForCausalLM supports only the official Qwen3-30B-A3B "
            f"configuration; mismatches={mismatches}"
        )
    if bool(getattr(hf_config, "tie_word_embeddings", False)):
        raise ValueError("Official Qwen3-30B-A3B rollout requires untied embeddings")
    if model_config.dtype != torch.bfloat16:
        raise ValueError("HyperQwen3MoeForCausalLM currently supports only bfloat16")
    tp_ep = (
        parallel_config.tensor_parallel_size == 2
        and parallel_config.data_parallel_size == 2
        and config_value(parallel_config, "enable_expert_parallel", False)
    )
    if parallel_config.tensor_parallel_size != 1 and not tp_ep:
        raise ValueError("HyperQwen3MoeForCausalLM currently supports tensor_parallel_size=1")
    if bool(config_value(parallel_config, "enable_eplb", False)):
        raise ValueError("HyperQwen3MoeForCausalLM does not support EPLB")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("HyperQwen3MoeForCausalLM currently supports pipeline_parallel_size=1")
    if config_value(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3MoeForCausalLM supports prefill_context_parallel_size=1")
    if config_value(parallel_config, "decode_context_parallel_size", 1) != 1:
        raise ValueError("HyperQwen3MoeForCausalLM supports decode_context_parallel_size=1")
    if vllm_config.quant_config is not None:
        raise ValueError("HyperQwen3MoeForCausalLM requires an unquantized checkpoint")
    if float(config_value(hf_config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("HyperQwen3MoeForCausalLM does not support attention dropout")


def _ep_moe_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Keep the HF Qwen3 router while using the shared Hyper EP dispatcher."""
    return routed_forward(
        module, hidden_states, router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"], combine_fn=qwen3_combine
    )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "inputs_embeds": 0,
    }
)
class HyperQwen3MoeForCausalLM(Qwen3MoeForCausalLM, MixtureOfExperts):
    """Run official Qwen3-30B-A3B with shared EP dispatch and local experts."""

    is_text_generation_model = True
    supports_pp = False
    supports_multimodal = False
    hyper_component_ownership = {
        "outer_model": "Transformers Qwen3MoeForCausalLM",
        "decoder_residual_norm": "Transformers Qwen3MoeDecoderLayer",
        "embedding_router_lm_head": "Transformers Qwen3-MoE",
        "paged_attention": (
            "vLLM Attention: paged KV cache plus prefill/decode scheduler ABI"
        ),
        "routed_experts": (
            "common local vLLM FusedMoE: Ascend fused/grouped expert kernel "
            "and stable physical w13/w2 storage; no internal communication"
        ),
        "expert_dispatch": "Hyper ep_routed_forward over vLLM-owned EP group; Qwen3 HF combine",
        "tensor_parallel": "Hyper ShardingPlanner/apply_sharding_plan; common dense Qwen3 TP loader",
        "moe_token_boundary": "Hyper public TP redistribution; padding confined to the pointwise MoE region",
        "scheduler_cache_lifecycle": "vLLM: request scheduling and paged-cache ownership",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Build the HF outer model without materializing duplicate experts."""
        _validate_adapter_config(vllm_config)
        config = vllm_config.model_config.hf_config
        target_device = vllm_config.device_config.device
        with torch.device("meta"):
            super().__init__(config)
        sparse_layers = []
        for layer_idx, layer in enumerate(self.model.layers):
            attention_prefix = join_prefix(
                prefix,
                f"model.layers.{layer_idx}.self_attn",
            )
            layer.self_attn = Qwen3PagedAttention(
                layer.self_attn,
                vllm_config=vllm_config,
                prefix=attention_prefix,
                family="HyperQwen3MoeForCausalLM",
            )
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                sparse_layers.append((layer_idx, layer.mlp))
        if not sparse_layers:
            raise ValueError("Official Qwen3-30B-A3B contains no sparse MoE layers")
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        tp_plan = None
        self._tp_mesh = None
        self._tp_placements = {}
        if tp_size > 1:
            group = get_ep_group()
            metadata_mesh = Layout(
                (group.world_size // tp_size, tp_size), ("dp", "tp"),
                rank_list=group.ranks, init_backend=False,
            ).mesh
            tp_plan = build_moe_tp_plan(self, metadata_mesh, tp_size=tp_size, ep_size=group.world_size)
        for _, moe in sparse_layers:
            moe.experts = None
        self.to_empty(device=target_device)

        self._expert_loaders: dict[str, tuple[HyperLocalFusedExperts, str]] = {}
        self.moe_layers = []
        placement = HyperExpertPlacement.from_config(vllm_config, int(config.num_experts))
        for layer_idx, moe in sparse_layers:
            base = f"model.layers.{layer_idx}.mlp.experts"
            experts = HyperLocalFusedExperts(
                local_expert_count=placement.local_count,
                global_expert_start=placement.global_start,
                hidden_size=int(config.hidden_size),
                intermediate_size=int(config.moe_intermediate_size),
                params_dtype=vllm_config.model_config.dtype,
                prefix=join_prefix(prefix, base),
            )
            moe.experts = experts
            moe.hyper_ep_mesh = placement.mesh
            if placement.mesh is not None:
                moe.forward = MethodType(_ep_moe_forward, moe)
            self._expert_loaders[f"{base}.gate_up_proj"] = (experts, "gate_up")
            self._expert_loaders[f"{base}.down_proj"] = (experts, "down")
            self.moe_layers.append(experts)
        if tp_plan is not None:
            self._tp_mesh = _device_mesh_from_vllm_tp()
            _, layout_info = apply_sharding_plan(self, tp_plan, self._tp_mesh, validate_mode=False)
            if layout_info is None:
                raise RuntimeError("Hyper MoE TP application returned no dense source layout")
            self._tp_placements = {name: tuple(axes) for name, (axes, _) in layout_info.items()}
            for _, moe in sparse_layers:
                pad_moe_tp_tokens(moe, tp_size)
        self._expected_weight_names = {
            name
            for name in dict(self.named_parameters())
            if not name.endswith((".experts.w13_weight", ".experts.w2_weight"))
        }
        self._expected_weight_names.update(self._expert_loaders)
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_logical_experts = int(config.num_experts)
        self.num_physical_experts = self.num_logical_experts
        self.num_local_physical_experts = placement.local_count
        self.num_routed_experts = self.num_logical_experts
        self.num_shared_experts = 0
        self.num_redundant_experts = 0
        self.expert_weights: list[list[torch.Tensor]] = []

        with torch.device("cpu"):
            canonical_rotary = type(self.model.rotary_emb)(self.config)
        self.model.rotary_emb.inv_freq = canonical_rotary.inv_freq.to(target_device)
        self.model.rotary_emb.original_inv_freq = (
            canonical_rotary.original_inv_freq.to(target_device)
        )

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """Reject EPLB because expert ownership is fixed."""
        del expert_load_view, logical_to_physical_map, logical_replica_count
        raise NotImplementedError("Hyper Qwen3-MoE does not support EPLB")

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        """Reject dynamic expert metadata because placement is fixed."""
        del num_physical_experts, num_local_physical_experts
        raise NotImplementedError("Hyper Qwen3-MoE uses fixed expert placement")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply the Transformers Qwen3-MoE token embedding."""
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
        """Run packed tokens through Transformers layers and vLLM leaves."""
        intermediate_tensors = kwargs.pop("intermediate_tensors", None)
        if intermediate_tensors is not None:
            raise ValueError("HyperQwen3MoeForCausalLM does not support pipeline state")
        if kwargs:
            raise ValueError(f"Unsupported Qwen3-MoE model arguments: {sorted(kwargs)}")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        if hidden_states.ndim != 2:
            raise ValueError("packed input embeddings must have shape [T,H]")

        position_ids = normalize_positions(
            positions,
            hidden_states.shape[0],
            family="Qwen3-MoE",
        )
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
        return self.model.norm(hidden_states).squeeze(0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute full-vocabulary logits with the Transformers LM head."""
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strictly load official per-expert or packed Transformers weights."""
        parameters = dict(self.named_parameters())
        loaded_names = set()
        per_expert_components: dict[str, set[tuple[int, str]]] = {}
        for source_name, loaded_weight in weights:
            if source_name.endswith("rotary_emb.inv_freq"):
                continue
            match = _EXPERT_PROJECTION_PATTERN.match(source_name)
            if match is not None:
                expert_base, global_expert_text, projection = match.groups()
                global_expert = int(global_expert_text)
                if not 0 <= global_expert < self.num_logical_experts:
                    raise ValueError(
                        f"Qwen3-MoE checkpoint expert index is out of range: {source_name!r}"
                    )
                logical_name = (
                    f"{expert_base}.down_proj"
                    if projection == "down_proj"
                    else f"{expert_base}.gate_up_proj"
                )
                if logical_name in loaded_names:
                    raise ValueError(
                        f"Duplicate Qwen3-MoE packed/per-expert tensor {source_name!r}"
                    )
                loader = self._expert_loaders.get(logical_name)
                if loader is None:
                    raise ValueError(f"Unexpected Qwen3-MoE checkpoint parameter {source_name!r}")
                component = (global_expert, projection)
                components = per_expert_components.setdefault(logical_name, set())
                if component in components:
                    raise ValueError(f"Duplicate Qwen3-MoE checkpoint parameter {source_name!r}")
                components.add(component)
                loader[0].load_expert_projection(loaded_weight, projection, global_expert)
                continue

            loader = self._expert_loaders.get(source_name)
            if loader is not None:
                if source_name in loaded_names or source_name in per_expert_components:
                    raise ValueError(f"Duplicate Qwen3-MoE checkpoint parameter {source_name!r}")
                if loaded_weight.shape[0] != self.num_logical_experts:
                    raise ValueError(
                        "Packed Qwen3-MoE weights must retain the complete expert axis: "
                        f"name={source_name!r}, expected={self.num_logical_experts}, "
                        f"actual={loaded_weight.shape[0]}"
                    )
                experts, kind = loader
                (experts.load_gate_up if kind == "gate_up" else experts.load_down)(
                    loaded_weight
                )
                loaded_names.add(source_name)
                continue

            parameter = parameters.get(source_name)
            if parameter is None:
                raise ValueError(f"Unexpected Qwen3-MoE checkpoint parameter {source_name!r}")
            if source_name in loaded_names:
                raise ValueError(f"Duplicate Qwen3-MoE checkpoint parameter {source_name!r}")
            _load_parameter(
                parameter, loaded_weight, tp_mesh=getattr(self, "_tp_mesh", None),
                placements=getattr(self, "_tp_placements", {}).get(source_name),
            )
            loaded_names.add(source_name)

        for logical_name, components in per_expert_components.items():
            projections_per_expert = 1 if logical_name.endswith("down_proj") else 2
            expected = self.num_logical_experts * projections_per_expert
            if len(components) != expected:
                raise ValueError(
                    "Incomplete Qwen3-MoE per-expert checkpoint tensor: "
                    f"name={logical_name!r}, expected_components={expected}, "
                    f"actual_components={len(components)}"
                )
            loaded_names.add(logical_name)

        missing = self._expected_weight_names.difference(loaded_names)
        if missing:
            raise ValueError(
                "Incomplete Qwen3-MoE checkpoint; missing parameters: "
                + ", ".join(sorted(missing))
            )
        return loaded_names


__all__ = ["HyperQwen3MoeForCausalLM"]
