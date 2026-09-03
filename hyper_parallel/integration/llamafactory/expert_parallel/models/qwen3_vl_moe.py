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

"""Qwen3-VL-MoE forward adapter for the core ExpertParallel call contract."""

from types import MethodType

# LlamaFactory is a PyTorch-only integration boundary.
# pylint: disable-next=forbidden-backend-import
import torch

# pylint: disable-next=forbidden-backend-import
from torch import nn

# pylint: disable-next=forbidden-backend-import
from torch.nn import functional
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
)

from hyper_parallel.auto_models.components.ops.npu_grouped_swiglu import (
    npu_grouped_swiglu,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.integration.llamafactory.expert_parallel.models.registry import (
    ExpertParallelModelPatch,
)


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local expert shard after core EP converts it to DTensor."""
    # Core ExpertParallel shards the expert dimension and stores it as DTensor.
    if isinstance(tensor, DTensor):
        # The local expert kernel must consume this rank's physical tensor shard.
        return tensor.to_local()
    # EP size 1 and CPU unit tests still use ordinary tensors.
    return tensor


def _run_local_experts_eager(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run rank-local Qwen3-VL experts when the NPU grouped kernel is unavailable."""
    # Read only the gate/up weights owned by the current EP rank.
    gate_up_proj = _to_local(experts.gate_up_proj)
    # Read only the down-projection weights owned by the current EP rank.
    down_proj = _to_local(experts.down_proj)
    # Each result entry corresponds to one non-empty local expert token segment.
    outputs = []
    # Dispatched tokens are contiguous and ordered by local expert.
    token_offset = 0
    # Iterate over local expert token counts in the same order as the weight shard.
    for expert_index, token_count in enumerate(tokens_per_expert.tolist()):
        # Empty experts have no GEMM work and contribute no output segment.
        if token_count == 0:
            continue
        # Slice the contiguous tokens routed to this local expert.
        expert_states = hidden_states[token_offset : token_offset + token_count]
        # Qwen3-VL fuses gate and up weights, so one linear result is split in half.
        gate, up = functional.linear(  # pylint: disable=not-callable
            expert_states, gate_up_proj[expert_index]
        ).chunk(2, dim=-1)
        # Reproduce the model's SwiGLU expert activation exactly.
        intermediate = experts.act_fn(gate) * up
        # Project the activated expert states back to the model hidden size.
        outputs.append(
            functional.linear(  # pylint: disable=not-callable
                intermediate, down_proj[expert_index]
            )
        )
        # Advance to the next expert-major token segment.
        token_offset += token_count

    # Keep shape and autograd connectivity when this rank receives no tokens.
    if not outputs:
        return hidden_states * 0.0
    # Restore the expert-major order expected by the core EP combine hook.
    return torch.cat(outputs, dim=0)


def _qwen3_vl_experts_ep_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    routed_probabilities: torch.Tensor | None = None,  # pylint: disable=unsupported-binary-operation
) -> torch.Tensor:
    """Compute expert-major local tokens supplied by the core EP hook."""
    # Core EP has already sharded both expert parameter tensors along dimension 0.
    gate_up_proj = _to_local(experts.gate_up_proj)
    down_proj = _to_local(experts.down_proj)
    # NPU training uses the grouped SwiGLU kernel for all local experts at once.
    if hidden_states.device.type == "npu":
        output = npu_grouped_swiglu(
            hidden_states,
            gate_up_proj,
            down_proj,
            tokens_per_expert,
        )
    else:
        # CPU tests and diagnostics use the numerically equivalent eager fallback.
        output = _run_local_experts_eager(experts, hidden_states, tokens_per_expert)

    # Router probabilities travel through the same EP permutation as their tokens.
    if routed_probabilities is not None:
        output = output * routed_probabilities.unsqueeze(-1).to(output.dtype)
    # The core post-hook reverse-dispatches this expert-major tensor to source ranks.
    return output


def _qwen3_vl_sparse_moe_ep_forward(
    moe_block: Qwen3VLMoeTextSparseMoeBlock,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Translate Qwen3-VL routing results to and from the core EP interface."""
    # Save the decoder-visible shape so the patched block preserves its API.
    input_shape = hidden_states.shape
    # The router and core EP contract both operate on a flat token dimension.
    flat_states = hidden_states.reshape(-1, input_shape[-1])
    # Reuse the loaded model's gate instead of duplicating router semantics.
    _, routing_weights, selected_experts = moe_block.gate(flat_states)
    # Read top-k from the actual routing result rather than hard-coding model config.
    experts_per_token = selected_experts.shape[-1]

    # Flatten token/top-k assignments into one route entry per expert invocation.
    flat_expert_indices = selected_experts.reshape(-1)
    # Group routes by global expert while preserving stable intra-expert order.
    route_order = torch.argsort(flat_expert_indices, stable=True)
    # Associate every flattened route with the source token that created it.
    source_token_indices = torch.arange(
        flat_states.shape[0], device=flat_states.device
    ).repeat_interleave(experts_per_token)
    # Apply the expert-major permutation to the route-to-token mapping.
    source_token_indices = source_token_indices.index_select(0, route_order)
    # Duplicate and reorder hidden states for their selected top-k experts.
    routed_states = flat_states.index_select(0, source_token_indices)
    # Apply the identical permutation to router probabilities.
    routed_probabilities = routing_weights.reshape(-1).index_select(0, route_order)
    # Count every global expert, including experts with zero routed tokens.
    tokens_per_expert = torch.bincount(
        flat_expert_indices,
        minlength=moe_block.experts.num_experts,
    )

    # Core EP hooks dispatch these tensors, run local experts, and reverse-dispatch.
    routed_output = moe_block.experts(
        routed_states,
        tokens_per_expert,
        routed_probabilities,
    )
    # Allocate the flat decoder output that will accumulate all top-k routes.
    output = torch.zeros_like(flat_states)
    # Sum each token's weighted expert outputs back into its source position.
    output.scatter_add_(
        0,
        source_token_indices.unsqueeze(-1).expand_as(routed_output),
        routed_output,
    )
    # Restore the original batch/sequence dimensions expected by the decoder layer.
    return output.reshape(input_shape)


def is_qwen3_vl_moe_model(model: nn.Module) -> bool:
    """Return whether the model tree contains a Qwen3-VL-MoE sparse block."""
    # Inspect the instantiated model tree; no model path or user-provided name is used.
    return any(
        isinstance(module, Qwen3VLMoeTextSparseMoeBlock)
        for module in model.modules()
    )


def _prepare_qwen3_vl_moe_expert_parallel(
    model: nn.Module, hp_args: object
) -> None:
    """Install the Qwen3-VL-MoE/core-EP adapter on every text MoE block."""
    # Count patched blocks so a stale supports/prepare mismatch cannot fail silently.
    patched_count = 0
    # Patch every decoder layer's sparse MoE block in the loaded model instance.
    for module in model.modules():
        # Leave visual modules and all unrelated text modules unchanged.
        if not isinstance(module, Qwen3VLMoeTextSparseMoeBlock):
            continue
        # Core EP uses an equal Shard(0), so expert count must divide by EP size.
        if module.experts.num_experts % hp_args.ep_size != 0:
            raise ValueError(
                f"Qwen3-VL-MoE num_experts ({module.experts.num_experts}) must be "
                f"divisible by ep_size ({hp_args.ep_size})."
            )
        # Reject repeated application before duplicate hooks or sharding can occur.
        if getattr(module, "_hyper_parallel_ep_forward_patched", False):
            raise RuntimeError(
                "Qwen3-VL-MoE forward has already been patched for expert parallelism."
            )

        # Bind only this model instance's expert contract; no global class is modified.
        module.experts.forward = MethodType(
            _qwen3_vl_experts_ep_forward, module.experts
        )
        # Bind the Qwen3-VL router-to-core-EP translation on this sparse block instance.
        module.forward = MethodType(_qwen3_vl_sparse_moe_ep_forward, module)
        # Mark the block after both method bindings succeed.
        module._hyper_parallel_ep_forward_patched = True  # pylint: disable=protected-access
        # Record this decoder layer as successfully adapted.
        patched_count += 1
    # A matched model must contain at least one block accepted by the prepare loop.
    if patched_count == 0:
        raise RuntimeError(
            "Qwen3-VL-MoE EP patch matched the model but found no text sparse MoE blocks."
        )


# Register model-structure detection and instance preparation as one immutable patch.
QWEN3_VL_MOE_EXPERT_PARALLEL_PATCH = ExpertParallelModelPatch(
    # The name is diagnostic only and never participates in model selection.
    name="qwen3_vl_moe",
    # Selection is based on the loaded module type discovered above.
    supports=is_qwen3_vl_moe_model,
    # Preparation installs the two instance-local forward adapters above.
    prepare=_prepare_qwen3_vl_moe_expert_parallel,
)
