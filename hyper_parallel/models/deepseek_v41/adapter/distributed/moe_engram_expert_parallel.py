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
"""Dispatch DeepSeek-V4.1 MoE experts and Engram lookups over EP."""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.engram import local_token_mask
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import (
    DeepseekV41MegaMoeExperts, _swiglu_limit,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import deepseek_v41_swiglu


def _deepseek_v41_unclamped_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply native FP32 SwiGLU when a nonpositive limit disables clamping."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    return deepseek_v41_swiglu(gate, up, 0.0).to(gate_up.dtype)


def _deepseek_v41_grouped_expert_forward(
        experts: torch.nn.Module,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run FP32 clamped SwiGLU between two grouped GEMMs.

    The generic EP path has already sorted ``hidden_states`` into expert-major
    order. Routed and shared experts reuse the native FP32 clamp and SwiGLU
    implementation. Nonpositive limits disable clamping; the down projection
    still receives the original hidden-state dtype.
    """
    # The shared primitive imports optional torch_npu; keep CPU/GPU imports lazy.
    from hyper_parallel.components.functional import grouped_matmul  # pylint: disable=C0415

    group_list = torch.cumsum(
        tokens_per_expert.to(device=hidden_states.device, dtype=torch.int64),
        dim=0,
    )
    gate_up = grouped_matmul(  # pylint: disable=not-callable
        hidden_states,
        experts.gate_up_proj.transpose(1, 2),
        bias=None,
        group_list=group_list,
        group_type=0,
        group_list_type=0,
    )
    # Cast the packed projections once; the shared helper's casts are then no-ops.
    gate, up = gate_up.float().chunk(2, dim=-1)
    intermediate = deepseek_v41_swiglu(gate, up, experts.limit)
    return grouped_matmul(  # pylint: disable=not-callable
        intermediate.to(hidden_states.dtype),
        experts.down_proj.transpose(1, 2),
        bias=None,
        group_list=group_list,
        group_type=0,
        group_list_type=0,
    )


def _router(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the V4.1 learned router's selected experts and weights."""
    output = (
        module.gate(hidden_states)
        if image_mask is None
        else module.gate(hidden_states, image_mask=image_mask)
    )
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise TypeError("DeepSeek-V4.1 gate must return logits, weights, and indices")
    _, weights, indices = output
    return indices, weights


def _routed_and_shared_forward(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None,
        ep_group: Any,
) -> torch.Tensor:
    """Dispatch routed experts and add the V4.1 shared expert branch."""
    routed = ep_routed_forward(
        module,
        hidden_states,
        router_fn=lambda target_module, target_states: _router(target_module, target_states, image_mask),
        ep_group=ep_group,
    )
    return routed + module.shared_experts(hidden_states)


@local_compute
def deepseek_v41_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        use_grouped_gemm: bool = False,
) -> Callable:
    """Build the complete V4.1 routed-plus-shared expert forward."""
    del mesh, cp_mesh
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 EP")
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 EP forward requires an active ep_mesh")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 validation layers must use learned routing")
    apply_gate = (
        module.experts._apply_gate  # pylint: disable=protected-access
        if module.experts.limit > 0 else _deepseek_v41_unclamped_gate
    )
    if use_grouped_gemm:
        module.experts.forward_expert_major = MethodType(
            _deepseek_v41_grouped_expert_forward,
            module.experts,
        )
    ep_group = ep_mesh.get_group("ep")
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    tp_rank = 0 if tp_mesh is None else tp_mesh.get_local_rank()
    bind_local_expert_forward(
        module,
        ep_mesh["ep"].size(),
        use_grouped_gemm=use_grouped_gemm,
        apply_gate=None if use_grouped_gemm else apply_gate,
    )

    if "image_mask" not in inspect.signature(module.forward).parameters:
        def text_compute_fn(
                module: Any,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the text-only source contract without visual routing state.

            Args:
                module: Routed and shared expert module.
                hidden_states: Current decoder hidden rows.
                input_ids: Unused token IDs from the local compute contract.
            """
            del input_ids
            return _routed_and_shared_forward(module, hidden_states, None, ep_group)

        return text_compute_fn

    def multimodal_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the multimodal source contract with optional visual routing state.

        Args:
            module: Routed and shared expert module.
            hidden_states: TP-local decoder hidden rows.
            input_ids: Unused token IDs from the local compute contract.
            image_mask: CP-local mask identifying image tokens.
        """
        del input_ids
        local_image_mask = local_token_mask(
            image_mask, hidden_states, tp_rank=tp_rank, tp_size=tp_size,
        )
        return _routed_and_shared_forward(module, hidden_states, local_image_mask, ep_group)

    return multimodal_compute_fn


@local_compute
def deepseek_v41_megamoe_compute_fn(
        *, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Bind a declaratively replaced expert module to the full EP mesh.

    The source router, visual routing bias and shared experts retain their
    original semantics. Parameter sharding and FSDP remain framework-owned.
    """
    del mesh
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 MegaMoe")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 MegaMoe requires learned routing")
    del cp_mesh  # Pointwise experts consume the sequence rows already owned by CP.
    if not isinstance(module.experts, DeepseekV41MegaMoeExperts):
        raise TypeError("MegaMoe compute requires the DeepseekV41MegaMoeExperts module replacement")
    if _swiglu_limit(module.shared_experts.limit) != module.experts.swiglu_limit:
        raise ValueError("Routed and shared experts must have the same swiglu_limit")
    ep_group = None if ep_mesh is None else ep_mesh.get_group("ep")
    ep_size = 1 if ep_mesh is None else ep_mesh["ep"].size()
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    if tp_size > 1:
        if ep_size <= 1:
            raise ValueError("DSV4.1 MegaMoe with TP>1 requires EP>1")
        if not module.experts.sequence_parallel:
            raise ValueError("DSV4.1 MegaMoe with TP>1 requires sequence_parallel=true")
    tp_rank = 0 if tp_mesh is None else tp_mesh.get_local_rank()
    if dist.is_initialized() and dist.get_world_size(ep_group) != ep_size:
        raise ValueError("MegaMoe EP mesh and process group sizes must match")
    module.experts.configure(ep_group, ep_size, module.gate.top_k)

    def compute_fn(
            module: Any, hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None, image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run original routing and shared experts around the MegaMoe call."""
        del input_ids
        local_image_mask = local_token_mask(
            image_mask, hidden_states, tp_rank=tp_rank, tp_size=tp_size,
        )
        if local_image_mask is not None:
            # Slicing multiple batch rows leaves gaps, which native MegaGate rejects.
            local_image_mask = local_image_mask.contiguous()
        indices, weights = _router(module, hidden_states, local_image_mask)
        return module.experts(hidden_states, indices, weights) + module.shared_experts(hidden_states)

    if "image_mask" in inspect.signature(module.forward).parameters:
        return compute_fn

    def text_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match the text-only source forward without visual routing arguments.

        Args:
            module: Routed and shared expert module.
            hidden_states: Current decoder hidden rows.
            input_ids: Unused token IDs from the local compute contract.
        """
        return compute_fn(module, hidden_states, input_ids)

    return text_compute_fn


@local_compute
def deepseek_v41_engram_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
) -> Callable:
    """Build the sparse EP-row lookup with CP/TP sequence alignment."""
    del mesh
    require_attrs(
        module,
        "hash_mapping",
        "embed",
        "parallel_forward",
        owner="DeepSeek-V4.1 Engram EP",
    )
    ep_group = None if ep_mesh is None else ep_mesh.get_group("ep")
    ep_size = 1 if ep_mesh is None else ep_mesh["ep"].size()
    ep_rank = 0 if ep_mesh is None else ep_mesh.get_local_rank("ep")
    cp_group = None if cp_mesh is None else cp_mesh.get_group()
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    cp_rank = 0 if cp_mesh is None else cp_mesh.get_local_rank()
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    tp_rank = 0 if tp_mesh is None else tp_mesh.get_local_rank()

    def compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            segment_starts: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route hash-row requests and run the exact V4.1 fusion.

        Args:
            module: Active Engram module.
            hidden_states: TP-local decoder hidden rows.
            input_ids: CP-local token IDs.
            segment_starts: Optional packed sequence boundaries.
            token_mask: CP-local mask selecting Engram updates.
        """
        from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import (  # pylint: disable=C0415
            HostEngramTable,
        )
        if isinstance(module.embed, HostEngramTable):
            ids = module._aligned_hash_ids(  # pylint: disable=protected-access
                hidden_states, input_ids, segment_starts, token_mask,
                cp_group=cp_group, cp_rank=cp_rank, cp_size=cp_size,
                tp_rank=tp_rank, tp_size=tp_size,
            )
            fused = module._fuse(hidden_states, module.embed(ids))  # pylint: disable=protected-access
            if token_mask is None:
                return fused
            return torch.where(
                token_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1),
                fused, hidden_states,
            )
        if ep_size > 1:
            return module.parallel_forward(
                hidden_states, input_ids, segment_starts, token_mask,
                ep_group=ep_group, ep_rank=ep_rank, ep_size=ep_size,
                cp_group=cp_group, cp_rank=cp_rank, cp_size=cp_size,
                tp_rank=tp_rank, tp_size=tp_size,
            )
        hash_ids = module._aligned_hash_ids(  # pylint: disable=protected-access
            hidden_states, input_ids, segment_starts, token_mask,
            cp_group=cp_group, cp_rank=cp_rank, cp_size=cp_size,
            tp_rank=tp_rank, tp_size=tp_size,
        )
        fused = module._fuse(hidden_states, module.embed(hash_ids))  # pylint: disable=protected-access
        mask = local_token_mask(token_mask, hidden_states, tp_rank=tp_rank, tp_size=tp_size)
        return fused if mask is None else torch.where(
            mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1), fused, hidden_states,
        )

    return compute_fn


__all__ = ["deepseek_v41_engram_compute_fn", "deepseek_v41_ep_compute_fn", "deepseek_v41_megamoe_compute_fn"]
