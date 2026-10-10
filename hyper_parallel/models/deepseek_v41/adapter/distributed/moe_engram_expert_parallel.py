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

from functools import wraps
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper, local_compute


def _local_router_metadata(tensor: torch.Tensor | None, states: torch.Tensor, tp_mesh: Any) -> torch.Tensor | None:
    """Slice CP-local metadata when an EP router keeps the TP sequence shard."""
    if tensor is None or tensor.shape == states.shape[:-1]:
        return tensor
    local_length = states.shape[-2]
    if tp_mesh is None or tensor.shape[-1] != local_length * tp_mesh.size():
        raise ValueError("router metadata must match the local or TP-gathered token dimensions")
    return tensor.narrow(-1, tp_mesh.get_local_rank() * local_length, local_length)


@inner_wrapper
def deepseek_v41_router_aux_loss_wrapper(
        target_module: torch.nn.Module,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
) -> list[_ForwardRewriteRequest]:
    """Reduce router statistics over token partitions inside the MLP boundary.

    CP always partitions tokens. TP does so only with EP; the non-EP MLP
    gathers TP sequence shards. EP may span distinct DP samples, so it is
    never a statistics group.

    Args:
        target_module: V4.1 MLP containing the learned gate.
        mesh: Full mesh, including independent DP axes.
        tp_mesh: Optional tensor-parallel mesh.
        cp_mesh: Optional context-parallel mesh.
        ep_mesh: Optional expert-dispatch mesh.
    """
    groups = tuple(
        axis.get_group() for axis in (cp_mesh, tp_mesh if ep_mesh is not None else None)
        if axis is not None and axis.size() > 1
    )
    gate = target_module.gate
    original_forward = gate.forward
    dp_groups = tuple(
        mesh[name].get_group() for name in (mesh.mesh_dim_names if mesh is not None else ())
        if name in ("dp", "dp_replicate", "dp_shard") and mesh[name].size() > 1
    )

    @wraps(original_forward)
    def router_forward(
            hidden_states: torch.Tensor,
            image_mask: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
            sequence_partition_groups: tuple[Any, ...] = (),
            sequence_ids: torch.Tensor | None = None,
            num_sequences: int | None = None,
    ) -> Any:
        """Align sample/modality metadata with the router's token partitions.

        Args:
            hidden_states: Gate-local token activations.
            image_mask: Optional image-token mask.
            token_mask: Optional valid-token mask.
            sequence_partition_groups: Must be empty; owned by this wrapper.
            sequence_ids: Optional packed logical sample IDs.
            num_sequences: Global sample count within this microbatch.
        """
        if sequence_partition_groups:
            raise ValueError("sequence partition groups are owned by the parallel wrapper")
        local_tp_mesh = tp_mesh if ep_mesh is not None else None
        return original_forward(
            hidden_states,
            image_mask=_local_router_metadata(image_mask, hidden_states, local_tp_mesh),
            token_mask=_local_router_metadata(token_mask, hidden_states, local_tp_mesh),
            sequence_ids=_local_router_metadata(sequence_ids, hidden_states, local_tp_mesh),
            num_sequences=num_sequences,
            sequence_partition_groups=groups,
        )

    return [
        _ForwardRewriteRequest(target_module, target_module.forward),
        _ForwardRewriteRequest(gate, router_forward, companion_attrs={"expert_bias_update_groups": dp_groups + groups}),
    ]


def _router(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        router_token_mask: torch.Tensor | None = None,
        router_sequence_ids: torch.Tensor | None = None,
        router_num_sequences: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the V4.1 learned router's selected experts and weights."""
    output = module.gate(
        hidden_states, image_mask=image_mask, token_mask=router_token_mask,
        sequence_ids=router_sequence_ids, num_sequences=router_num_sequences,
    )
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise TypeError("DeepSeek-V4.1 gate must return logits, weights, and indices")
    _, weights, indices = output
    return indices, weights


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
    del mesh, tp_mesh, cp_mesh
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 EP")
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 EP forward requires an active ep_mesh")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 validation layers must use learned routing")
    if use_grouped_gemm:
        raise ValueError("DeepSeek-V4.1 clamp semantics currently require use_grouped_gemm=false")
    ep_group = ep_mesh.get_group("ep")
    bind_local_expert_forward(
        module,
        ep_mesh["ep"].size(),
        apply_gate=module.experts._apply_gate,  # pylint: disable=protected-access
    )

    def compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
            router_token_mask: torch.Tensor | None = None,
            router_sequence_ids: torch.Tensor | None = None,
            router_num_sequences: int | None = None,
    ) -> torch.Tensor:
        """Run text and image routing through the same dispatch/combine path.

        Args:
            module: MLP owning routed and shared experts.
            hidden_states: Token activations before expert dispatch.
            input_ids: Unused by learned routing.
            image_mask: Optional image-token mask.
            router_token_mask: Optional valid-token mask.
            router_sequence_ids: Optional packed logical sample IDs.
            router_num_sequences: Global sample count within this microbatch.
        """
        del input_ids
        routed = ep_routed_forward(
            module,
            hidden_states,
            router_fn=lambda target, states: _router(
                target, states, image_mask, router_token_mask, router_sequence_ids, router_num_sequences,
            ),
            ep_group=ep_group,
        )
        return routed + module.shared_experts(hidden_states)

    return compute_fn


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
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 Engram EP requires an active ep_mesh")
    ep_group = ep_mesh.get_group("ep")
    ep_size = ep_mesh["ep"].size()
    ep_rank = ep_mesh.get_local_rank("ep")
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
        """Route hash-row requests and run the exact V4.1 fusion."""
        return module.parallel_forward(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            cp_group=cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    return compute_fn


__all__ = ["deepseek_v41_engram_compute_fn", "deepseek_v41_ep_compute_fn", "deepseek_v41_router_aux_loss_wrapper"]
