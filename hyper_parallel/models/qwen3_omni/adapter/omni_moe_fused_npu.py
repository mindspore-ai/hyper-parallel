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
"""Route the omni thinker sparse-MoE through the NPU fused kernels.

The stock Transformers 4.57.1 ``Qwen3OmniMoeThinkerTextSparseMoeBlock`` loops
over hit experts with tiny per-expert GEMMs plus an ``index_add_`` scatter —
on 8x 910B2 this dominates the thinker step time. The fused NPU path
instead uses ``npu_moe_token_permute`` → grouped GEMM → ``npu_swiglu`` →
grouped GEMM → ``npu_moe_token_unpermute``.

This module routes the stock HF block through that path without changing the
parameter layout: HF keeps 128 separate ``Qwen3OmniMoeThinkerTextMLP``
experts (per-expert ``gate_proj``/``up_proj``/``down_proj`` nn.Linear), while
the NPU grouped-GEMM kernel wants packed ``(E, 2I, H)`` / ``(E, H, I)``
weights. The pack uses plain autograd ops (``torch.cat`` + ``torch.stack``)
in forward, so backward slices the packed wgrad back into the very same
Parameter grads automatically — FSDP/HSDP and the optimizer see unchanged
per-parameter state. Only ``npu_grouped_matmul`` lacks autograd in
torch_npu 2.9.0, so it gets a custom Function with dgrad (transposed-weight
grouped GEMM) and wgrad (``split_item=3, group_type=2``).

Router math, the ``(final_hidden_states, router_logits)`` return contract
and the no-aux-loss behaviour are preserved bit-for-bit from the stock
forward (H's trainer never sets ``output_router_logits``).

Gated by ``HYPER_OMNI_MOE_FUSED_NPU=1``; without it the stock forward is
untouched (baseline semantics unchanged). Idempotent.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_ENV_FLAG = "HYPER_OMNI_MOE_FUSED_NPU"


class GmmFunction(torch.autograd.Function):
    """``npu_grouped_matmul`` with explicit dgrad/wgrad autograd.

    Forward computes ``x @ weight`` per expert group; ``weight`` arrives
    already transposed to ``(E, K_in, K_out)``. ``group_list`` holds the
    per-expert token counts (``group_list_type=1``).
    """

    @staticmethod
    def forward(ctx, x, weight, group_list):
        """Run grouped matmul ``x @ weight`` per expert and save state."""
        import torch_npu  # pylint: disable=import-outside-toplevel

        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        return torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    def backward(ctx, grad_output):
        """Grouped dgrad (transposed weight) and wgrad (x.T @ grad_output)."""
        import torch_npu  # pylint: disable=import-outside-toplevel

        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        grad_output = grad_output.contiguous()

        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

        grad_weight = torch_npu.npu_grouped_matmul(
            [x.transpose(-1, -2)],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]

        return grad_input, grad_weight, None


def npu_group_gemm(x, weight, group_list):
    """Apply ``GmmFunction`` to ``(x, weight, group_list)``."""
    return GmmFunction.apply(x, weight, group_list)


def _npu_fused_moe_forward(
    hidden_states,
    routing_weights,
    selected_experts,
    gate_up_weight,
    down_weight,
    num_experts,
):
    """Fused single-device MoE forward (permute → GEMM → SwiGLU → GEMM → unpermute).

    ``gate_up_weight`` is ``(E, 2I, H)`` and ``down_weight`` is ``(E, H, I)``
    (both packed by the caller with autograd-tracked ops). Permute/SwiGLU/
    unpermute have native autograd in torch_npu 2.9.0; the routing weights
    are folded in by ``npu_moe_token_unpermute``'s ``probs`` argument, so
    gradients flow back to the router through them.
    """
    import torch_npu  # pylint: disable=import-outside-toplevel

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1]).contiguous()

    permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states, selected_experts.to(torch.int32)
    )
    tokens_per_expert = torch.histc(selected_experts, bins=num_experts, min=0, max=num_experts)

    intermediate_hidden_states = npu_group_gemm(
        permuted_hidden_states, gate_up_weight.transpose(1, 2), tokens_per_expert
    )
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    output = npu_group_gemm(
        intermediate_activations, down_weight.transpose(1, 2), tokens_per_expert
    )
    # probs-based unpermute applies the routing weights after fc2, matching
    # the stock block's ``expert_layer(x) * routing_weights`` order.
    return torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)


def _make_fused_block_forward():
    """Build a forward replacing the stock sparse-MoE eager expert loop."""

    def _fused_forward(self, hidden_states):
        """Fused-NPU sparse MoE forward with the stock router and contract."""
        import torch  # pylint: disable=import-outside-toplevel
        from torch.nn import functional as F  # pylint: disable=import-outside-toplevel

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Router math copied verbatim from the stock 4.57.1 forward.
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Pack the per-expert Linear weights for the grouped-GEMM kernel.
        # cat/stack are autograd ops: backward slices the packed wgrad back
        # into each Parameter's grad with no manual unpacking.
        gate_up_weight = torch.stack(
            [torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0) for expert in self.experts]
        )
        down_weight = torch.stack([expert.down_proj.weight for expert in self.experts])

        final_hidden_states = _npu_fused_moe_forward(
            hidden_states,
            routing_weights,
            selected_experts,
            gate_up_weight,
            down_weight,
            self.num_experts,
        )
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return _fused_forward


def _ensure_omni_moe_fused_npu() -> None:
    """Patch the omni thinker sparse-MoE to the NPU fused kernels.

    No-op unless ``HYPER_OMNI_MOE_FUSED_NPU=1`` and the NPU fused ops are
    importable. Idempotent; only the stock Transformers class is touched.
    """
    if os.environ.get(_ENV_FLAG, "0") != "1":
        return
    try:
        import torch_npu  # noqa: F401, pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        logger.warning("%s=1 but torch_npu is unavailable; keeping stock MoE forward.", _ENV_FLAG)
        return

    from transformers.models.qwen3_omni_moe import (  # pylint: disable=import-outside-toplevel
        modeling_qwen3_omni_moe as _M,
    )

    name = "Qwen3OmniMoeThinkerTextSparseMoeBlock"
    moe_cls = getattr(_M, name, None)
    if moe_cls is None:
        return
    if getattr(moe_cls, "_hyper_parallel_moe_fused_npu", False):
        return

    moe_cls.forward = _make_fused_block_forward()
    moe_cls._hyper_parallel_moe_fused_npu = True
    logger.info("Patched %s.forward -> NPU fused MoE kernels.", name)
