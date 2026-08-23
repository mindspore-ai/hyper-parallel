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
"""Tests for the pure PyTorch Kimi Delta Attention references."""
from unittest.mock import patch

import pytest
import torch

from hyper_parallel.core.context_parallel.kimi_delta_attention_context_parallel import (
    KimiDeltaAttentionContextParallel,
)
from hyper_parallel.models.modules.kimi_delta_attention import (
    KimiDeltaAttention,
    chunk_kda,
    torch_apply_kda_state_summary,
    torch_chunk_kda,
    torch_compose_kda_state_summaries,
    torch_kda_gate,
    torch_kda_state_summary,
    torch_recurrent_kda,
)


def test_kda_context_parallel_rejects_duplicate_apply():
    """Applying a second KDA CP wrapper fails before nesting executors."""
    module = torch.nn.Module()
    style = KimiDeltaAttentionContextParallel(mode="ulysses")
    patch_target = (
        "hyper_parallel.core.context_parallel.kimi_delta_attention_context_parallel."
        "KimiDeltaAttentionLayerUlyssesCP"
    )
    with patch(patch_target, return_value=torch.nn.Identity()):
        style.apply(module, object())
        with pytest.raises(RuntimeError, match="already been applied"):
            style.apply(module, object())


def test_kimi_delta_attention_layer_runs_eager_training_forward_backward():
    """The standalone training layer owns parameters and remains Triton-free."""
    torch.manual_seed(7)
    layer = KimiDeltaAttention(
        hidden_size=32,
        num_heads=2,
        num_v_heads=4,
        head_k_dim=4,
        head_v_dim=3,
        conv_kernel_size=3,
        chunk_size=4,
    )
    hidden_states = torch.randn(2, 9, 32, requires_grad=True)

    output = layer(hidden_states)
    output.float().square().mean().backward()

    assert output.shape == hidden_states.shape
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"


def _make_inputs(
    seed: int = 17,
    num_k_heads: int = 2,
    num_v_heads: int = 4,
    sequence_length: int = 17,
):
    """Build stable grouped-value KDA inputs with all gradients enabled."""
    torch.manual_seed(seed)
    shapes = {
        "query": (2, sequence_length, num_k_heads, 4),
        "key": (2, sequence_length, num_k_heads, 4),
        "value": (2, sequence_length, num_v_heads, 3),
        "gate": (2, sequence_length, num_v_heads, 4),
        "beta": (2, sequence_length, num_v_heads),
        "a_log": (num_v_heads,),
        "dt_bias": (num_v_heads, 4),
        "initial_state": (2, num_v_heads, 4, 3),
    }
    tensors = {
        name: (torch.randn(shape, dtype=torch.float32) * 0.2).requires_grad_(True)
        for name, shape in shapes.items()
    }
    tensors["dt_bias"] = (
        torch.randn(shapes["dt_bias"], dtype=torch.float32) * 0.2 - 4.0
    ).requires_grad_(True)
    return tensors


def _run_and_grad(function, tensors):
    """Run one KDA implementation and differentiate output plus final state."""
    output, final_state = function(
        tensors["query"],
        tensors["key"],
        tensors["value"],
        tensors["gate"],
        tensors["beta"],
        a_log=tensors["a_log"],
        dt_bias=tensors["dt_bias"],
        initial_state=tensors["initial_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        **({"chunk_size": 8} if function is torch_chunk_kda else {}),
    )
    torch.manual_seed(23)
    output_grad = torch.randn(
        output.shape,
        dtype=output.dtype,
        device=output.device,
    )
    final_state_grad = torch.randn(
        final_state.shape,
        dtype=final_state.dtype,
        device=final_state.device,
    )
    loss = (output.float() * output_grad).sum()
    loss = loss + (final_state.float() * final_state_grad).sum()
    names = tuple(tensors)
    grads = torch.autograd.grad(loss, tuple(tensors[name] for name in names))
    return output, final_state, dict(zip(names, grads))


def test_kda_eager_dispatch_matches_torch_reference():
    """The public eager backend preserves the differentiable Torch contract."""
    inputs = _make_inputs(sequence_length=9)
    kwargs = {
        "a_log": inputs["a_log"],
        "dt_bias": inputs["dt_bias"],
        "initial_state": inputs["initial_state"],
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": -5.0,
        "chunk_size": 8,
    }
    operands = tuple(
        inputs[name] for name in ("query", "key", "value", "gate", "beta")
    )

    expected_output, expected_state = torch_chunk_kda(*operands, **kwargs)
    actual_output, actual_state = chunk_kda(*operands, backend="eager", **kwargs)

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)


def test_kimi_k3_gate_is_lower_bounded_and_differentiable():
    """The K3 decay mapping stays in ``(lower_bound, 0)`` and has gradients."""
    gate_logits = torch.linspace(-20, 20, 48).reshape(1, 3, 4, 4).requires_grad_()
    a_log = torch.zeros(4, requires_grad=True)
    dt_bias = torch.zeros(4, 4, requires_grad=True)

    gate = torch_kda_gate(
        gate_logits,
        a_log,
        dt_bias,
        lower_bound=-5.0,
    )

    assert torch.all(gate < 0)
    assert torch.all(gate >= -5.0)
    gate.sum().backward()
    assert gate_logits.grad is not None
    assert a_log.grad is not None
    assert dt_bias.grad is not None


@pytest.mark.parametrize(
    "seed,num_k_heads,num_v_heads",
    ((0, 2, 2), (11, 2, 4), (29, 1, 4)),
)
def test_chunk_kda_matches_recurrent_output_state_and_all_gradients(
    seed,
    num_k_heads,
    num_v_heads,
):
    """Chunkwise KDA matches the token recurrence, including random ``dht``."""
    recurrent_inputs = _make_inputs(seed, num_k_heads, num_v_heads)
    chunk_inputs = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in recurrent_inputs.items()
    }

    expected_output, expected_state, expected_grads = _run_and_grad(
        torch_recurrent_kda,
        recurrent_inputs,
    )
    actual_output, actual_state, actual_grads = _run_and_grad(
        torch_chunk_kda,
        chunk_inputs,
    )

    torch.testing.assert_close(actual_output, expected_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)
    for name in expected_grads:
        torch.testing.assert_close(
            actual_grads[name],
            expected_grads[name],
            atol=2e-4,
            rtol=2e-4,
            msg=f"{name} gradient mismatch",
        )


def test_kda_rejects_incompatible_grouped_value_heads():
    """Grouped-value KDA requires value heads to be divisible by Q/K heads."""
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn_like(query)
    value = torch.randn(1, 4, 4, 8)
    gate = torch.randn(1, 4, 4, 8)
    beta = torch.randn(1, 4, 4)

    with pytest.raises(ValueError, match="num_v_heads must be divisible"):
        torch_recurrent_kda(query, key, value, gate, beta)


def _run_cp4_state_summary(summary_inputs, common_kwargs):
    """Run four local KDA shards and compose their affine summaries."""
    state = summary_inputs["initial_state"]
    output_shards = []
    composed_summary = None
    local_sequence_length = summary_inputs["query"].shape[1] // 4
    for cp_rank in range(4):
        token_slice = slice(
            cp_rank * local_sequence_length,
            (cp_rank + 1) * local_sequence_length,
        )
        local_tensors = {
            name: summary_inputs[name][:, token_slice]
            for name in ("query", "key", "value", "gate", "beta")
        }
        state_ext, transition = torch_kda_state_summary(
            local_tensors["query"],
            local_tensors["key"],
            local_tensors["value"],
            local_tensors["gate"],
            local_tensors["beta"],
            a_log=summary_inputs["a_log"],
            dt_bias=summary_inputs["dt_bias"],
            **common_kwargs,
        )
        local_output, _ = torch_chunk_kda(
            local_tensors["query"],
            local_tensors["key"],
            local_tensors["value"],
            local_tensors["gate"],
            local_tensors["beta"],
            a_log=summary_inputs["a_log"],
            dt_bias=summary_inputs["dt_bias"],
            initial_state=state,
            output_final_state=False,
            **common_kwargs,
        )
        output_shards.append(local_output)
        state = torch_apply_kda_state_summary(state_ext, transition, state)
        if composed_summary is None:
            composed_summary = (state_ext, transition)
        else:
            composed_summary = torch_compose_kda_state_summaries(
                *composed_summary,
                state_ext,
                transition,
            )

    composed_state = torch_apply_kda_state_summary(
        *composed_summary,
        summary_inputs["initial_state"],
    )
    return torch.cat(output_shards, dim=1), state, composed_state


def test_kda_state_summary_cp4_matches_output_state_and_gradients():
    """Four affine local summaries reproduce full-sequence KDA training."""
    reference_inputs = _make_inputs(sequence_length=64)
    summary_inputs = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in reference_inputs.items()
    }
    common_kwargs = {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": -5.0,
        "chunk_size": 8,
    }

    expected_output, expected_state = torch_chunk_kda(
        reference_inputs["query"],
        reference_inputs["key"],
        reference_inputs["value"],
        reference_inputs["gate"],
        reference_inputs["beta"],
        a_log=reference_inputs["a_log"],
        dt_bias=reference_inputs["dt_bias"],
        initial_state=reference_inputs["initial_state"],
        output_final_state=True,
        **common_kwargs,
    )

    actual_output, actual_state, composed_state = _run_cp4_state_summary(
        summary_inputs,
        common_kwargs,
    )
    torch.testing.assert_close(actual_output, expected_output, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(composed_state, expected_state, atol=3e-5, rtol=3e-5)

    torch.manual_seed(101)
    output_grad = torch.randn_like(expected_output)
    state_grad = torch.randn_like(expected_state)
    reference_loss = (expected_output * output_grad).sum()
    reference_loss = reference_loss + (expected_state * state_grad).sum()
    sequential_loss = (actual_output * output_grad).sum()
    sequential_loss = sequential_loss + (actual_state * state_grad).sum()
    composed_loss = (actual_output * output_grad).sum()
    composed_loss = composed_loss + (composed_state * state_grad).sum()
    names = tuple(reference_inputs)
    expected_grads = torch.autograd.grad(
        reference_loss,
        tuple(reference_inputs[name] for name in names),
    )
    sequential_grads = torch.autograd.grad(
        sequential_loss,
        tuple(summary_inputs[name] for name in names),
        retain_graph=True,
    )
    composed_grads = torch.autograd.grad(
        composed_loss,
        tuple(summary_inputs[name] for name in names),
    )
    for path, actual_grads in (
        ("sequential", sequential_grads),
        ("composed", composed_grads),
    ):
        for name, actual_grad, expected_grad in zip(
            names, actual_grads, expected_grads
        ):
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                atol=4e-4,
                rtol=4e-4,
                msg=f"{path} {name} gradient mismatch",
            )


def test_kda_state_summary_keeps_fp32_state_under_autocast():
    """Affine summary state must not inherit the BF16 compute dtype."""
    inputs = _make_inputs(sequence_length=16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        state_ext, transition = torch_kda_state_summary(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["gate"],
            inputs["beta"],
            a_log=inputs["a_log"],
            dt_bias=inputs["dt_bias"],
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=-5.0,
            chunk_size=8,
        )
        final_state = torch_apply_kda_state_summary(
            state_ext,
            transition,
            inputs["initial_state"].to(torch.bfloat16),
        )

    assert state_ext.dtype == torch.float32
    assert transition.dtype == torch.float32
    assert final_state.dtype == torch.float32
