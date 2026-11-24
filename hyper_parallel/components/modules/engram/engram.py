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

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.distributed.module_binding import ParallelBinding

from .engram_family import build_engram as Engram
from .ngram_hash_mapping import NgramHashMapping
from .parallel_engram import ParallelEngram

__all__ = ["Engram", "EngramModule", "EPEngramModule"]


def _positive_int(value: Any, name: str) -> int:
    """Return a strictly positive integer config value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer(value: Any, name: str) -> int:
    """Return an integer config value while rejecting Boolean aliases."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    """Return a finite positive floating-point config value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class EngramModule(ParallelEngram, nn.Module):
    """Ordinary Engram parameter ownership and local fusion with separate parallel execution."""

    _requires_parallel_binding = False

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
        self._parallel_coordinates: dict[str, Any] | None = None
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
        self.embed = module.embed
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

    def _init_from_config(self, config: Any) -> None:
        """Validate hash assets and allocate local fusion parameters for direct construction."""
        self.layer_id = _integer(config.layer_id, "layer_id")
        self.hidden_size = _positive_int(config.hidden_size, "hidden_size")
        self.hc_mult = _positive_int(config.hc_mult, "hc_mult")
        embedding_dim = _positive_int(config.embedding_dim, "embedding_dim")
        self.eps = _finite_positive(config.eps, "eps")
        self.clamp_value = _finite_positive(config.clamp_value, "clamp_value")
        self.hash_mapping = NgramHashMapping(config.assets, self.layer_id)
        self.logical_num_embeddings = self.hash_mapping.logical_num_embeddings
        rows = self.logical_num_embeddings if config.padded_num_embeddings is None else _positive_int(
            config.padded_num_embeddings, "padded_num_embeddings",
        )
        if rows < self.logical_num_embeddings:
            raise ValueError("padded_num_embeddings must cover all logical hash rows")
        self.padded_num_embeddings = rows
        columns = (self.hash_mapping.max_ngram_size - 1) * self.hash_mapping.num_heads
        self.embed = self._create_embedding(self.padded_num_embeddings, embedding_dim)
        self.wkv = nn.Linear(columns * embedding_dim, (self.hc_mult + 1) * self.hidden_size, bias=False)
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size))

    def _create_embedding(self, rows: int, width: int) -> nn.Embedding:
        """Allocate the device-table representation for direct construction."""
        return nn.Embedding(rows, width)

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

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Use the job's existing EP row owners and CP/TP sequence axes.

        Args:
            binding: Prebuilt topology managed by the common module binder.
        """
        mesh = binding.mesh_binding.mesh_context
        expert_mesh = getattr(mesh, "fsdp_moe_mesh", None)
        coordinates = {}
        for axis in ("ep", "cp", "tp"):
            region = expert_mesh if axis == "ep" and expert_mesh is not None else mesh.device_mesh
            names = getattr(region, "mesh_dim_names", None) or ()
            axis_mesh = binding.mesh_binding.axis_mesh(axis) if axis in names else None
            size = 1 if axis_mesh is None else axis_mesh.size()
            coordinates[f"{axis}_size"] = size
            coordinates[f"{axis}_rank"] = 0 if size == 1 else axis_mesh.get_local_rank()
            if axis != "tp":
                coordinates[f"{axis}_group"] = None if size == 1 else axis_mesh.get_group()
        ep_size = coordinates["ep_size"]
        if (self._requires_parallel_binding and ep_size == 1) or self.padded_num_embeddings % ep_size:
            raise ValueError("Engram requires a matching EP mesh that evenly divides the padded table")
        self._parallel_coordinates = coordinates

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run local Engram or use the bound job meshes.

        Args:
            hidden_states: Current decoder hidden rows.
            input_ids: Local token IDs for hash lookup.
            segment_starts: Optional packed sequence boundaries.
            token_mask: Optional mask selecting Engram updates.
        """
        if self._parallel_coordinates is not None:
            local_inputs = tuple(tensor.to_local() if isinstance(tensor, DTensor) else tensor
                                 for tensor in (hidden_states, input_ids, segment_starts, token_mask))
            # EP lookup owns global-to-local row offsets. A DTensor embedding
            # dispatch would try to apply the table sharding a second time.
            with SkipDTensorDispatch():
                fused = self.parallel_forward(*local_inputs, **self._parallel_coordinates)
            if isinstance(hidden_states, DTensor):
                return DTensor.from_local(fused, hidden_states.device_mesh, hidden_states.placements)
            return fused
        if self._requires_parallel_binding:
            raise RuntimeError("EP Engram must bind the job mesh before its first forward")
        hash_ids = self._aligned_hash_ids(hidden_states, input_ids, segment_starts, token_mask)
        fused = self._fuse(hidden_states, self.embed(hash_ids))
        if token_mask is None:
            return fused
        return torch.where(token_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1), fused, hidden_states)


class EPEngramModule(EngramModule):
    """Device Engram whose table is distributed over EP row owners."""

    _requires_parallel_binding = True
