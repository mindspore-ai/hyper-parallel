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
"""Engram parameter ownership, device lookup and local fusion."""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.module_binding import ParallelBinding, ParallelModuleSpec
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec

from .host_engram_table import HostEngramTable
from .ngram_hash_mapping import NgramHashMapping
from .parallel_engram import ParallelEngram, local_token_mask


def engram_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own replicated fusion weights and EP rows without splitting the fusion projection."""
    del module_fqn, member
    replicated = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    params = {name: dict(replicated) for name, _ in module.named_parameters()}
    if context.axis_sizes["ep"] > 1:
        params["embed.weight"]["ep"] = Shard(0)
    return ModuleShardingSpec(params=params, in_src={}, in_dst={}, is_boundary=False)


def validate_engram_layout(*, module: nn.Module, sharding: Any, context: Any) -> None:
    """Reject unsupported row partitions and fusion layouts before binding communication."""
    expected = engram_sharding(module=module, module_fqn="", member=None, context=context)
    if sharding.params != expected.params:
        raise ValueError("Engram requires replicated fusion weights and contiguous EP table rows")
    if module.padded_num_embeddings % context.axis_sizes["ep"]:
        raise ValueError("Engram physical table rows must be padded to an EP-size multiple")


class _LocalRowEmbedding(nn.Embedding):
    """Consume owner-local row indices while preserving ordinary child/FSDP hooks."""

    @classmethod

    def adopt(cls, source: nn.Embedding) -> "_LocalRowEmbedding":
        """Preserve the source Parameter identity without initializing or copying weights."""
        if source.padding_idx is not None or source.max_norm is not None or source.scale_grad_by_freq or source.sparse:
            raise ValueError("Engram requires an unpadded dense embedding without in-place row normalization")
        if isinstance(source, cls):
            return source
        result = cls(source.num_embeddings, source.embedding_dim, _weight=source.weight,
                     _freeze=not source.weight.requires_grad)
        result.weight = source.weight
        result.training = source.training
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
        """Look up already-routed local rows without a second global-vocabulary redistribution."""
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return F.embedding(input, weight)


class EngramModule(ParallelEngram, nn.Module):
    """Ordinary Engram parameter ownership and local fusion with separate parallel execution."""

    def __init__(
        self,
        *,
        module: nn.Module | None = None,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
        config: Any = None,
    ) -> None:
        """Reuse a source Engram module's parameters and hash mapping."""
        super().__init__()
        del module_fqn, context
        if (module is None) == (config is None):
            raise ValueError("provide exactly one of config or module")
        if config is not None:
            self._init_from_config(config)
            return
        required = ("hash_mapping", "embed", "wkv", "q_weight", "k_weight")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f"Engram source is missing required attributes: {missing}")
        self.hash_mapping = module.hash_mapping
        self.embed = _LocalRowEmbedding.adopt(module.embed) if isinstance(module.embed, nn.Embedding) else module.embed
        self.wkv = module.wkv
        self.q_weight = module.q_weight
        self.k_weight = module.k_weight
        self.layer_id = int(module.layer_id)
        self.hidden_size = int(module.hidden_size)
        self.hc_mult = int(module.hc_mult)
        self.eps = float(module.eps)
        self.clamp_value = float(module.clamp_value)
        self.logical_num_embeddings = int(module.logical_num_embeddings)
        self.padded_num_embeddings = int(module.padded_num_embeddings)
        self.train(module.training)

    def parallel_module_spec(self) -> ParallelModuleSpec:
        """Borrow existing job groups; Engram owns its sparse row transport and hash halo."""
        return ParallelModuleSpec(
            "engram.rows",
            "mesh",
            {"forward": ("hidden_states", "input_ids", "segment_starts", "token_mask")},
            validate=validate_engram_layout,
        )

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Record job-owned EP/CP/TP resources without constructing additional groups."""
        mesh = binding.mesh_binding.mesh_context
        sizes = {axis: int(getattr(mesh, f"{axis}_size", 1)) for axis in ("ep", "cp", "tp")}
        state = {}
        for axis in ("ep", "cp"):
            axis_mesh = binding.mesh_binding.axis_mesh(axis) if sizes[axis] > 1 else None
            state.update(
                {
                    f"{axis}_size": sizes[axis],
                    f"{axis}_rank": axis_mesh.get_local_rank() if axis_mesh is not None else 0,
                    f"{axis}_group": axis_mesh.get_group() if axis_mesh is not None else None,
                }
            )
        state.update(
            tp_size=sizes["tp"] if getattr(mesh, "sequence_parallel", False) else 1,
            tp_rank=int(getattr(mesh, "tp_rank", 0)),
        )
        self._parallel_state = state

    def _init_from_config(self, config: Any) -> None:
        """Validate hash assets and allocate local fusion parameters for direct construction."""
        for name in ("hidden_size", "hc_mult", "embedding_dim"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(config.layer_id, bool) or not isinstance(config.layer_id, int):
            raise ValueError("layer_id must be an integer")
        for name in ("eps", "clamp_value"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.layer_id, self.hidden_size, self.hc_mult = config.layer_id, config.hidden_size, config.hc_mult
        self.eps, self.clamp_value = config.eps, config.clamp_value
        self.hash_mapping = NgramHashMapping(config.assets, config.layer_id)
        self.logical_num_embeddings = self.hash_mapping.logical_num_embeddings
        rows = config.padded_num_embeddings
        if rows is None:
            rows = self.logical_num_embeddings
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < self.logical_num_embeddings:
            raise ValueError("padded_num_embeddings must cover all logical hash rows")
        self.padded_num_embeddings = rows
        columns = (self.hash_mapping.max_ngram_size - 1) * self.hash_mapping.num_heads
        self.embed = self._create_embedding(self.padded_num_embeddings, config.embedding_dim)
        self.wkv = nn.Linear(columns * config.embedding_dim, (config.hc_mult + 1) * config.hidden_size, bias=False)
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def _create_embedding(self, rows: int, width: int) -> nn.Embedding:
        """Allocate the device-table representation for direct construction."""
        return _LocalRowEmbedding(rows, width)

    def reset_parameters(self) -> None:
        """Initialize owned query/key scales after materialization."""
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    def make_transforms(self) -> list:
        """Keep source parameter names and layouts unchanged."""
        return []

    def checkpoint_logical_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the logical embedding shape for checkpoint loading and checking."""
        return {"embed.weight": (self.logical_num_embeddings, int(self.embed.weight.shape[1]))}

    def _fuse(self, hidden_states: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the exact V4.1 WKV projection and signed-root gate."""
        key_value = self.wkv(embeddings.flatten(start_dim=-2))
        key, value = key_value.split(
            [self.hc_mult * self.hidden_size, self.hidden_size], dim=-1
        )
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        hidden_fp32 = hidden_states.float()
        weight = self.q_weight.float() * self.k_weight.float()
        reciprocal_std = torch.rsqrt(hidden_fp32.square().mean(-1) + self.eps)
        reciprocal_std = reciprocal_std * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden_fp32 * weight * key).sum(-1)
        dot = dot * reciprocal_std * self.hidden_size**-0.5
        root = dot.abs().clamp_min(self.clamp_value).sqrt()
        signed_root = torch.where(dot < 0, -root, root)
        gate = torch.sigmoid(signed_root)
        return (
            hidden_fp32 + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        ).to(hidden_states.dtype)


    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run owned hash alignment, sparse table lookup and residual fusion.

        Args:
            hidden_states: Current decoder hidden rows.
            input_ids: Local token IDs for hash lookup.
            segment_starts: Optional packed sequence boundaries.
            token_mask: Optional mask selecting Engram updates.
        """
        state = getattr(self, "_parallel_state", {})
        if state.get("ep_size", 1) > 1 and not isinstance(self.embed, HostEngramTable):
            return self.parallel_forward(
                hidden_states, input_ids, segment_starts, token_mask,
                ep_group=state["ep_group"], ep_rank=state["ep_rank"], ep_size=state["ep_size"],
                **{key: value for key, value in state.items() if not key.startswith("ep_")},
            )
        sequence = {key: value for key, value in state.items() if not key.startswith("ep_")}
        hash_ids = self._aligned_hash_ids(hidden_states, input_ids, segment_starts, token_mask, **sequence)
        fused = self._fuse(hidden_states, self.embed(hash_ids))
        token_mask = local_token_mask(
            token_mask, hidden_states, tp_rank=state.get("tp_rank", 0), tp_size=state.get("tp_size", 1)
        )
        if token_mask is None:
            return fused
        return torch.where(token_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1), fused, hidden_states)
