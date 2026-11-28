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
import pytest
import torch

from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.components.functional.kimi_delta_attention import chunk_kda
from hyper_parallel.components.functional.kimi_delta_attention import (
    torch_apply_kda_state_summary,
    torch_chunk_kda,
    torch_compose_kda_state_summaries,
    torch_kda_gate,
    torch_kda_state_summary,
    torch_recurrent_kda,
)
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_kimi_delta_attention_layer_runs_eager_training_forward_backward():
    """Feature: Standalone eager KDA layer.

    Description: Run the parameter-owning layer through forward and backward.
    Expectation: The layer produces the expected shapes and trainable parameter gradients.
    """
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
    dt_bias_mean: float = -4.0,
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
        torch.randn(shapes["dt_bias"], dtype=torch.float32) * 0.2 + dt_bias_mean
    ).requires_grad_(True)
    return tensors


def _run_and_grad(function, tensors, chunk_size=8):
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
        **({"chunk_size": chunk_size} if function is torch_chunk_kda else {}),
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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_kda_eager_dispatch_matches_torch_reference():
    """Feature: KDA eager dispatch.

    Description: Compare the public eager backend with the Torch implementation.
    Expectation: The dispatched output and recurrent state match the reference.
    """
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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_kimi_gate_is_lower_bounded_and_differentiable():
    """Feature: Kimi gate transform.

    Description: Evaluate the gate and backpropagate through its parameters.
    Expectation: Gate values remain in the configured interval and gradients are available.
    """
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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize(
    "seed,num_k_heads,num_v_heads,sequence_length,chunk_size,dt_bias_mean",
    ((0, 2, 2, 17, 8, -4.0), (11, 2, 4, 17, 8, -4.0), (29, 1, 4, 17, 8, -4.0),
     (0, 2, 2, 128, 64, 0.0), (11, 2, 4, 129, 64, 0.0), (29, 1, 4, 65, 64, 2.0)),
)
def test_chunk_kda_matches_recurrent_output_state_and_all_gradients(
    seed,
    num_k_heads,
    num_v_heads,
    sequence_length,
    chunk_size,
    dt_bias_mean,
):
    """Feature: Chunked KDA recurrence.

    Description: Compare token recurrence with short chunks and strong decay in padded/unpadded chunk64.
    Expectation: Outputs, final state and all gradients remain finite and agree.
    """
    recurrent_inputs = _make_inputs(seed, num_k_heads, num_v_heads, sequence_length, dt_bias_mean)
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
        chunk_size=chunk_size,
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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_kda_rejects_incompatible_grouped_value_heads():
    """Feature: KDA grouped-value contract.

    Description: Pass a value-head count not divisible by the query-head count.
    Expectation: The eager implementation rejects the incompatible shape.
    """
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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("sequence_length,chunk_size,dt_bias_mean", ((64, 8, -4.0), (256, 64, 0.0)))
def test_kda_state_summary_cp4_matches_output_state_and_gradients(sequence_length, chunk_size, dt_bias_mean):
    """Feature: KDA affine state composition.

    Description: Compose four local summaries and compare with the full-sequence recurrence.
    Expectation: Output, final state and gradients match the full-sequence reference.
    """
    reference_inputs = _make_inputs(sequence_length=sequence_length, dt_bias_mean=dt_bias_mean)
    summary_inputs = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in reference_inputs.items()
    }
    common_kwargs = {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": -5.0,
        "chunk_size": chunk_size,
    }

    expected_output, expected_state = torch_recurrent_kda(
        reference_inputs["query"],
        reference_inputs["key"],
        reference_inputs["value"],
        reference_inputs["gate"],
        reference_inputs["beta"],
        a_log=reference_inputs["a_log"],
        dt_bias=reference_inputs["dt_bias"],
        initial_state=reference_inputs["initial_state"],
        output_final_state=True,
        **{name: value for name, value in common_kwargs.items() if name != "chunk_size"},
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


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_kda_state_summary_keeps_fp32_state_under_autocast():
    """Feature: KDA state precision under autocast.

    Description: Build and compose summaries inside a BF16 autocast region.
    Expectation: Summary and recurrent state tensors retain FP32 precision.
    """
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
