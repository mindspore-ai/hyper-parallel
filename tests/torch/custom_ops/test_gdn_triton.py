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
"""Correctness tests for the Triton-Ascend Gated DeltaNet backend."""

import pytest
import torch


def _require_npu():
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for the Triton GDN backend")


def _clone_leaves(tensors):
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


def _run_backend(inputs, dout, dht, backend):
    from hyper_parallel.models.modules.linear_attention import chunk_gated_delta_rule

    query, key, value, gate, beta, initial_state = inputs
    output, final_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        backend=backend,
    )
    objective = (output.float() * dout.float()).sum()
    objective = objective + (final_state.float() * dht.float()).sum()
    gradients = torch.autograd.grad(objective, inputs)
    return output, final_state, gradients


def _make_inputs():
    torch.manual_seed(42)
    device = torch.device("npu:0")
    shape = (1, 128, 4, 128)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    gate = torch.nn.functional.logsigmoid(
        torch.rand(shape[:3], device=device, dtype=torch.bfloat16)
    ).float()
    beta = torch.rand(
        shape[:3], device=device, dtype=torch.bfloat16
    ).sigmoid()
    initial_state = 0.1 * torch.randn(
        shape[0],
        shape[2],
        shape[3],
        shape[3],
        device=device,
        dtype=torch.float32,
    )
    return query, key, value, gate, beta, initial_state


def test_gdn_triton_matches_eager_forward_and_backward():
    """Compare fused output, final state, and all input gradients with eager."""
    _require_npu()
    inputs = _make_inputs()
    dout = torch.randn_like(inputs[2])
    dht = 0.1 * torch.randn_like(inputs[-1])
    eager = _run_backend(_clone_leaves(inputs), dout, dht, "eager")
    triton = _run_backend(_clone_leaves(inputs), dout, dht, "triton")
    torch.npu.synchronize()

    torch.testing.assert_close(triton[0].float(), eager[0].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(triton[1].float(), eager[1].float(), rtol=1e-2, atol=1e-2)
    for actual, expected in zip(triton[2], eager[2]):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=1e-2
        )


def test_gdn_state_summary_matches_recurrent_state_path():
    """Validate affine forward and backward summaries against fused state kernels."""
    _require_npu()
    from hyper_parallel.platform.torch.custom_ops.gdn import (
        apply_gdn_state_gradient_summary,
        apply_gdn_state_summary,
        chunk_gated_delta_rule_bwd_prepare_saved,
        chunk_gated_delta_rule_bwd_state_saved,
        chunk_gated_delta_rule_fwd_apply_state_saved,
        chunk_gated_delta_rule_fwd_prepare_saved,
        chunk_gated_delta_rule_state_gradient_summary_bwd,
        chunk_gated_delta_rule_state_summary_fwd,
    )

    query, key, value, gate, beta, initial_state = _make_inputs()
    (
        query_norm,
        key_norm,
        _,
        _,
        gate_cumsum,
        A,
        w,
        u,
        scale,
    ) = chunk_gated_delta_rule_fwd_prepare_saved(
        query,
        key,
        value,
        gate,
        beta,
        use_qk_l2norm_in_kernel=True,
    )
    state_ext, transition = chunk_gated_delta_rule_state_summary_fwd(
        key_norm,
        w,
        u,
        gate_cumsum,
    )
    _, _, final_state = chunk_gated_delta_rule_fwd_apply_state_saved(
        key_norm,
        gate_cumsum,
        w,
        u,
        initial_state=initial_state,
        output_final_state=True,
    )
    summary_state = apply_gdn_state_summary(state_ext, transition, initial_state)
    torch.testing.assert_close(
        summary_state.float(), final_state.float(), rtol=5e-3, atol=2e-3
    )

    grad_output = torch.randn_like(value)
    grad_final_state = 0.1 * torch.randn_like(initial_state)
    w, _, _, dv = chunk_gated_delta_rule_bwd_prepare_saved(
        query_norm,
        key_norm,
        value,
        gate_cumsum,
        beta,
        A,
        initial_state,
        grad_output,
        scale,
    )
    grad_state_ext = chunk_gated_delta_rule_state_gradient_summary_bwd(
        query_norm,
        key_norm,
        w,
        gate_cumsum,
        grad_output,
        dv,
        scale,
    )
    _, grad_initial_state, _ = chunk_gated_delta_rule_bwd_state_saved(
        query_norm,
        key_norm,
        gate_cumsum,
        w,
        initial_state,
        grad_final_state,
        grad_output,
        dv,
        scale,
    )
    summary_grad = apply_gdn_state_gradient_summary(
        grad_state_ext,
        transition,
        grad_final_state,
    )
    torch.testing.assert_close(
        summary_grad.float(), grad_initial_state.float(), rtol=5e-3, atol=2e-3
    )
