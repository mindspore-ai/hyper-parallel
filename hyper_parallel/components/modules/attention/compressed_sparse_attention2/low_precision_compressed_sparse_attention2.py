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
"""Low-precision CSA2 member binding KV/Indexer policies and allowlisted FP8 Linears."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.family import resolve_module_family_target
from hyper_parallel.components.modules.linear import Linear
from hyper_parallel.components.quantization.config import LowPrecisionConfig
from hyper_parallel.components.quantization.functional.indexer_qk import create_indexer_qk_policy
from hyper_parallel.components.quantization.functional.main_kv_cache_qat import MainKVCacheQATStrategy
from hyper_parallel.components.quantization.functional.swa_kv_qat import SWAKVQATStrategy
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from .compressed_sparse_attention2 import CompressedSparseAttention2Module, SharedCompressedAttentionState

# Explicit allowlist of 2D Dense GEMMs inside CSA2. Excludes o_a_proj (3D grouped
# torch.bmm weight container whose nn.Linear.forward is never called) and
# indexer.weights_proj (small scalar head-merge gate).
DEFAULT_LINEAR_ALLOWLIST: tuple[str, ...] = (
    "q_a_proj",
    "q_b_proj",
    "kv_proj",
    "o_b_proj",
    "compressor.wkv",
    "compressor.wgate",
    "indexer.q_b_proj",
    "indexer.wk",
)


def normalize_linear_allowlist(allowlist: Any) -> tuple[str, ...]:
    """Validate and freeze the allowlist of CSA2 child Linear projections."""
    if isinstance(allowlist, (str, bytes)) or not isinstance(allowlist, Sequence):
        raise TypeError("linear_allowlist must be a sequence of allowlisted CSA2 Linear paths")
    normalized = tuple(dict.fromkeys(allowlist))
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError("linear_allowlist entries must be non-empty strings")
    unsupported = sorted(set(normalized) - set(DEFAULT_LINEAR_ALLOWLIST))
    if unsupported:
        raise ValueError(
            f"Unsupported CSA2 Linear target(s) {unsupported}; "
            f"allowed targets: {DEFAULT_LINEAR_ALLOWLIST}"
        )
    return normalized


def active_low_precision_config(context: Any) -> LowPrecisionConfig | None:
    """Extract an enabled LowPrecisionConfig from the build context, if present."""
    policy = context.get("low_precision") if isinstance(context, Mapping) else None
    return policy if isinstance(policy, LowPrecisionConfig) and policy.enabled else None


def _validate_parallel_axes(context: Mapping[str, Any] | None, module_fqn: str) -> None:
    """Reject unsupported parallel axes on single-card low-precision CSA2."""
    if context is None:
        return
    prefix = f"{module_fqn or '<root>'}: LowPrecisionCompressedSparseAttention2Module"
    for axis in ("tp", "cp", "pp"):
        if context.get(axis) or context.get(f"{axis}_size", 1) > 1:
            raise NotImplementedError(f"{prefix} currently requires TP=CP=PP=1")


class LowPrecisionCompressedSparseAttention2Module(CompressedSparseAttention2Module):
    """Single-card low-precision CSA2 member selected by the Module Family allowlist."""

    def __init__(
        self,
        module: nn.Module | None = None,
        *,
        config: Any = None,
        layer_idx: int = 0,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        replace_indexer: bool = True,
        use_fused_ops: bool = False,
        linear_dtype_scheme: str | None = None,
        linear_allowlist: Sequence[str] = DEFAULT_LINEAR_ALLOWLIST,
    ) -> None:
        """Adopt CSA2 parameters, replace allowlisted Linears, and bind KV/Indexer policies."""
        _validate_parallel_axes(context, module_fqn)
        allowlist = normalize_linear_allowlist(linear_allowlist)
        if linear_dtype_scheme is not None and (
            not isinstance(linear_dtype_scheme, str) or not linear_dtype_scheme
        ):
            raise ValueError("linear_dtype_scheme must be None or a non-empty scheme name")
        super().__init__(
            module,
            config=config,
            layer_idx=layer_idx,
            replace_indexer=replace_indexer,
            use_fused_ops=use_fused_ops,
        )
        self.linear_dtype_scheme = linear_dtype_scheme
        self.linear_allowlist = allowlist
        self._kv_qat_strategy: MainKVCacheQATStrategy | None = getattr(module, "_kv_qat_strategy", None)
        self._swa_kv_qat_strategy: SWAKVQATStrategy | None = getattr(module, "_swa_kv_qat_strategy", None)

        direct_config = active_low_precision_config(context)
        if direct_config is not None:
            self._bind_from_config(direct_config, module_fqn=module_fqn, context=context)
        elif linear_dtype_scheme is not None:
            raise ValueError(
                "linear_dtype_scheme requires an enabled LowPrecisionConfig in context['low_precision']"
            )

    def _bind_from_config(
        self,
        direct_config: LowPrecisionConfig,
        *,
        module_fqn: str,
        context: Mapping[str, Any] | None,
    ) -> None:
        """Apply the configured Linear scheme and CSA2 KV/Indexer strategies."""
        if not direct_config.active_direct_formats() and self.linear_dtype_scheme is None:
            raise ValueError(
                "LowPrecisionCompressedSparseAttention2Module requires at least one CSA2 format "
                "(indexer_qk, main_kv, swa_kv) or linear_dtype_scheme"
            )
        if self.linear_dtype_scheme is not None:
            self._replace_allowlisted_linears(
                direct_config, module_fqn=module_fqn, context=context or {},
            )
        if direct_config.indexer_qk is not None and self.is_index_source:
            self.indexer.set_qk_policy(create_indexer_qk_policy(direct_config.indexer_qk))
        if direct_config.main_kv is not None:
            self.set_kv_qat_strategy(MainKVCacheQATStrategy(direct_config.main_kv))
        if direct_config.swa_kv is not None:
            self.set_swa_kv_qat_strategy(SWAKVQATStrategy(direct_config.swa_kv))

    def _replace_allowlisted_linears(
        self,
        direct_config: LowPrecisionConfig,
        *,
        module_fqn: str,
        context: Mapping[str, Any],
    ) -> None:
        """Replace allowlisted child nn.Linear projections via the Linear Module Family."""
        scheme = direct_config.resolve_dtype_scheme(self.linear_dtype_scheme)
        factory, extra_kwargs = resolve_module_family_target(
            Linear,
            {
                "weight_format": scheme.weight_format,
                "act_format": scheme.act_format,
                "is_fake_quantize": scheme.is_fake_quantize,
                "fused": False,
            },
        )
        present_modules = dict(self.named_modules())
        for rel_name in self.linear_allowlist:
            child = present_modules.get(rel_name)
            if type(child) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
                continue
            child_fqn = f"{module_fqn}.{rel_name}" if module_fqn else rel_name
            replaced = factory(module=child, module_fqn=child_fqn, context=context, **extra_kwargs)
            parent_path, _, leaf = rel_name.rpartition(".")
            owner = self.get_submodule(parent_path) if parent_path else self
            setattr(owner, leaf, replaced)

    def set_kv_qat_strategy(self, strategy: MainKVCacheQATStrategy) -> None:
        """Bind the Main KV QAT strategy applied once at the compressor output."""
        self._kv_qat_strategy = strategy

    def set_swa_kv_qat_strategy(self, strategy: SWAKVQATStrategy) -> None:
        """Bind the local SWA KV packer."""
        self._swa_kv_qat_strategy = strategy

    def _project_raw_kv(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        """Project, rotate, and optionally pack the local SWA K=V rows."""
        key_value = super()._project_raw_kv(hidden_states, cos, sin)
        return self._swa_kv_qat_strategy.pack(key_value) if self._swa_kv_qat_strategy is not None else key_value

    def _publish_compressed_kv(
        self, shared_state: SharedCompressedAttentionState, compressed: torch.Tensor,
    ) -> None:
        """Pack and publish the Full-layer compressed K=V bank once outside recomputation."""
        if is_recomputing():
            return
        if self._kv_qat_strategy is not None:
            compressed = self._kv_qat_strategy.pack(compressed)
        shared_state.publish_compressed_kv(self.layer_idx, compressed)
