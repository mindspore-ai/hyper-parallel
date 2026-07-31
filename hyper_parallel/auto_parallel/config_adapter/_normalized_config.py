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
"""Shared type definitions for auto parallel strategy search configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


@dataclass
class NormalizedConfig:
    """Aggregate container for a parallel strategy search task.

    Holds all configuration sections as plain dicts for maximum
    compatibility with PR631's Config class (sapp_nd.nd.common.config).
    The ``model_spec`` field names follow the HuggingFace-style
    ``config_overrides`` convention (e.g. ``hidden_size``,
    ``num_hidden_layers``), matching the HP ``train.yaml`` format.

    **Required model_spec fields**: ``num_hidden_layers``, ``hidden_size``,
    ``num_attention_heads``, ``vocab_size``.

    **Optional model_spec fields**: ``intermediate_size``,
    ``num_key_value_heads``, ``max_position_embeddings``,
    ``local_batch_size``, ``compute_dtype``,
    ``softmax_compute_type``, ``moe_enabled``, ``num_experts``,
    ``num_experts_per_tok``, ``num_shared_experts``, ``moe_intermediate_size``,
    ``use_flash_attention``, ``use_seq_parallel``,
    ``optimizer_weight_shard_size``,
    ``enable_parallel_optimizer``,
    ``multiple_of``, ``ffn_dim_multiplier``,
    ``mtp_depth``, ``first_k_dense_replace``, ``kv_lora_rank``,
    ``q_lora_rank``, ``qk_rope_head_dim``, ``v_head_dim``,
    ``capacity_factor``, ``offset``,
    ``param_init_type``, ``recompute_slice_activation``.

    Args:
        model_spec: Model architecture parameters. Must contain at least
            ``num_hidden_layers``, ``hidden_size``, ``num_attention_heads``,
            ``vocab_size``.
        cluster_spec: Hardware cluster description.
        search_space: Parallel dimension candidate values, e.g.
            ``{"dp": [1,2,4], "tp": [1,2,4,8], "pp": [1,2], "cp": [1], "ep": [1]}``.
        constraint: User-imposed constraints (global_batch_size,
            memory_limit_gb, fixed_*_degree).
        estimator: Estimation algorithm parameters.  Optional key:
            ``cp_algo`` (``"colossalai_cp"`` | ``"ulysses_cp"`` | ``"hybrid_cp"``,
            default ``"colossalai_cp"``).
        pp_config: Pipeline-parallel specific configuration.
        resolved_strategy: Final resolved strategy, populated after search.
    """

    model_spec: Dict[str, Any] = field(default_factory=dict)
    cluster_spec: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, List[int]] = field(default_factory=dict)
    constraint: Dict[str, Any] = field(default_factory=dict)
    estimator: Dict[str, Any] = field(default_factory=dict)
    pp_config: Dict[str, Any] = field(default_factory=dict)
    resolved_strategy: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all config sections to a nested dictionary."""
        result: Dict[str, Any] = {
            "model_spec": dict(self.model_spec),
            "cluster_spec": dict(self.cluster_spec),
            "search_space": dict(self.search_space),
            "constraint": dict(self.constraint),
            "estimator": dict(self.estimator),
            "pp_config": dict(self.pp_config),
        }
        if self.resolved_strategy is not None:
            result["resolved_strategy"] = dict(self.resolved_strategy)
        return result


@dataclass
class ValidationError:
    """A single validation error or warning discovered during config validation.

    Args:
        field_path: Dot-separated path to the offending field
            (e.g. ``"model_spec.dim"``).
        message: Human-readable description of the problem.
        severity: Error severity level (``"error"`` or ``"warning"``).
    """

    field_path: str
    message: str
    severity: Literal["error", "warning"] = "error"


ValidationSeverity = Literal["error", "warning"]
