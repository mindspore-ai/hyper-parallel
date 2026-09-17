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
"""Ulysses rows followed by state-CP columns, with full-CP convolution halos."""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    KimiDeltaAttentionLayerP2PCP,
    _num_value_heads,
    KimiDeltaAttentionP2PCP,
    KimiDeltaAttentionUlyssesCP,
    _slice_local_heads,
)

from .kimi_delta_attention_mesh import split_kda_mesh


def build_hybrid_mesh(device_mesh: DeviceMesh, degree: int) -> DeviceMesh:
    """Preserve chronological rank order in a state-by-Ulysses mesh.

    Args:
        device_mesh: Chronologically ordered one-dimensional CP mesh.
        degree: Number of consecutive ranks in a Ulysses row.
    """
    if (not isinstance(degree, int) or isinstance(degree, bool)) or degree < 2 or device_mesh.size() % degree:
        raise ValueError("ulysses_degree must be an integer >= 2 dividing the CP size.")
    return split_kda_mesh(device_mesh, degree, ("kda_state", "kda_ulysses"))


class KimiDeltaAttentionHybridCP(KimiDeltaAttentionUlyssesCP):
    """Exchange token/head shards before invoking P2P or fused AllGather state CP."""

    def __init__(self, device_mesh: DeviceMesh, *, ulysses_degree: int,
                 chunk_size: int = 64, lower_bound: float = -5.0, safe_gate: bool = True,
                 backend: str = "triton", **state_options: Any) -> None:
        """Create both submeshes once, outside forward and checkpoint replay."""
        protocol = state_options.get("boundary_protocol", "p2p")
        if protocol not in ("p2p", "allgather", "grouped_allgather_p2p"):
            raise ValueError("Hybrid requires a supported KDA state boundary protocol.")
        if backend.lower() != "triton" and protocol != "p2p":
            raise ValueError("KDA AllGather boundaries require backend='triton'.")
        if (not isinstance(ulysses_degree, int) or isinstance(ulysses_degree, bool)
                or ulysses_degree < 2 or device_mesh.size() % ulysses_degree):
            raise ValueError("ulysses_degree must be an integer >= 2 dividing the CP size.")
        width = state_options.get("group_size", 1)
        if (not isinstance(width, int) or isinstance(width, bool) or width < 1
                or (device_mesh.size() // ulysses_degree) % width):
            raise ValueError("group_size must be a positive integer dividing the state CP size.")
        if protocol != "grouped_allgather_p2p" and width != 1:
            raise ValueError("group_size is only valid for grouped_allgather_p2p.")
        hybrid = build_hybrid_mesh(device_mesh, ulysses_degree)
        super().__init__(hybrid["kda_ulysses"], chunk_size=chunk_size, lower_bound=lower_bound,
                         safe_gate=safe_gate, backend=backend)
        self.hybrid_mesh = hybrid
        self.state = KimiDeltaAttentionP2PCP(hybrid["kda_state"], chunk_size=chunk_size,
                                            lower_bound=lower_bound, safe_gate=safe_gate,
                                            backend=backend, **state_options)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
                dt_bias: torch.Tensor, scale: float | None = None) -> torch.Tensor:
        """Keep source-token order within each row and chronological rows in state CP.

        Args:
            query: Projected query tensor [B,T,H,K].
            key: Projected key tensor [B,T,H,K].
            value: Projected value tensor [B,T,H,V].
            gate: Per-token gate logits [B,T,H,K].
            beta: Per-token beta logits [B,T,H].
            a_log: Learned decay parameter with one entry per value head.
            dt_bias: Learned gate bias with H times K entries.
            scale: Attention scaling factor; None selects the local default.
        """
        self._validate_inputs(query, key, value, gate, beta, a_log, dt_bias)
        heads, key_dim = value.shape[2], query.shape[-1]
        inputs = tuple(self._seq_to_head(tensor) for tensor in (query, key, value, gate, beta))
        local_a = _slice_local_heads(a_log.reshape(heads), self.cp_rank, self.cp_size)
        local_bias = _slice_local_heads(dt_bias.reshape(heads, key_dim), self.cp_rank, self.cp_size)
        output = self.state(*inputs, a_log=local_a, dt_bias=local_bias, scale=scale)
        return self._head_to_seq(output)


class KimiDeltaAttentionLayerHybridCP(KimiDeltaAttentionLayerP2PCP):
    """Reuse full-layer projection/halo handling and replace only the projected core."""

    def __init__(self, module: nn.Module, device_mesh: DeviceMesh, *, ulysses_degree: int,
                 chunk_size: int = 64, backend: str = "triton", **state_options: Any) -> None:
        """Retain full-CP neighbors for ShortConv before any Ulysses exchange."""
        if not isinstance(ulysses_degree, int) or isinstance(ulysses_degree, bool) or ulysses_degree < 2:
            raise ValueError("Hybrid KDA requires integer ulysses_degree >= 2.")
        if module.num_heads % ulysses_degree or _num_value_heads(module) % ulysses_degree:
            raise ValueError("KDA query and value heads must both be divisible by ulysses_degree.")
        super().__init__(module, device_mesh, chunk_size=chunk_size, backend=backend)
        self.hybrid_core = KimiDeltaAttentionHybridCP(
            device_mesh, ulysses_degree=ulysses_degree, chunk_size=chunk_size,
            lower_bound=self.lower_bound, safe_gate=self.safe_gate, backend=backend, **state_options)

    def _run_local_kda(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       gate: torch.Tensor, beta: torch.Tensor, *, a_log: torch.Tensor,
                       dt_bias: torch.Tensor) -> torch.Tensor:
        return self.hybrid_core(query, key, value, gate, beta, a_log=a_log, dt_bias=dt_bias)


__all__ = ["KimiDeltaAttentionHybridCP", "KimiDeltaAttentionLayerHybridCP"]
