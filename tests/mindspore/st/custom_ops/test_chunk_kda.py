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
"""Single-card accuracy and capability tests for MindSpore Chunk KDA."""
import math

import numpy as np
import pytest
import torch

import mindspore as ms

from hyper_parallel.custom_ops.experimental import (
    npu_chunk_kda,
    npu_chunk_kda_bwd,
    npu_chunk_kda_fwd,
)
from tests.common.mark_utils import arg_mark


def _torch_reference(q, k, v, g, beta, scale, chunk_size):
    """Differentiable dense BSND CPU reference for preprocessed gates."""
    batch, tokens, heads, key_dim = q.shape
    value_heads = v.shape[2]
    value_dim = v.shape[3]
    ratio = value_heads // heads
    outputs = []
    final_states = []
    for batch_id in range(batch):
        state = torch.zeros(
            value_heads, key_dim, value_dim, dtype=q.dtype, device=q.device
        )
        chunks = []
        for begin in range(0, tokens, chunk_size):
            end = min(begin + chunk_size, tokens)
            q_block = q[batch_id, begin:end].permute(1, 0, 2).repeat_interleave(ratio, dim=0)
            k_block = k[batch_id, begin:end].permute(1, 0, 2).repeat_interleave(ratio, dim=0)
            v_block = v[batch_id, begin:end].permute(1, 0, 2)
            g_block = torch.cumsum(g[batch_id, begin:end].permute(1, 0, 2), dim=1) / math.log(2.0)
            beta_block = beta[batch_id, begin:end].transpose(0, 1)
            length = end - begin
            exp_g = torch.exp2(g_block)
            right = k_block * torch.exp2(-g_block)
            qk = torch.bmm(q_block * exp_g, right.transpose(1, 2)) * scale
            kk = torch.bmm(k_block * exp_g, right.transpose(1, 2))
            causal = torch.ones(length, length, dtype=torch.bool).tril()
            strict = torch.ones(length, length, dtype=torch.bool).tril(-1)
            qk = qk.masked_fill(~causal, 0.0)
            lhs = (kk * beta_block.unsqueeze(-1)).masked_fill(~strict, 0.0)
            identity = torch.eye(length, dtype=q.dtype).expand(value_heads, -1, -1)
            inverse = torch.linalg.solve_triangular(  # pylint: disable=not-callable
                lhs + identity, identity, upper=False
            )
            w_block = torch.bmm(inverse, k_block * beta_block.unsqueeze(-1) * exp_g)
            u_block = torch.bmm(inverse, v_block * beta_block.unsqueeze(-1))
            previous = state
            last_g = g_block[:, -1]
            kg_block = k_block * torch.exp2(last_g.unsqueeze(1) - g_block)
            v_new = u_block - torch.bmm(w_block, previous)
            state = torch.exp2(last_g).unsqueeze(-1) * previous + torch.bmm(
                kg_block.transpose(1, 2), v_new
            )
            out = torch.bmm(q_block * exp_g, previous) * scale + torch.bmm(qk, v_new)
            chunks.append(out.permute(1, 0, 2))
        outputs.append(torch.cat(chunks, dim=0))
        final_states.append(state)
    return torch.stack(outputs), torch.stack(final_states)


def _inputs(seed=11, tokens=17, heads=1, value_heads=2):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-0.04, 0.04, (1, tokens, heads, 128)).astype(np.float16)
    k = rng.uniform(-0.04, 0.04, (1, tokens, heads, 128)).astype(np.float16)
    v = rng.uniform(-0.04, 0.04, (1, tokens, value_heads, 128)).astype(np.float16)
    g = rng.uniform(-0.01, -0.001, (1, tokens, value_heads, 128)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, (1, tokens, value_heads)).astype(np.float32)
    return q, k, v, g, beta


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_chunk_kda_forward_and_grad_match_cpu_reference():
    """Compare a GQA tail case against a differentiable FP64 CPU reference."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    arrays = _inputs()
    scale = 1.0 / math.sqrt(128)
    ms_inputs = tuple(ms.Tensor(value) for value in arrays)
    output, final_state, intermediates = npu_chunk_kda_fwd(
        *ms_inputs, scale, 64, layout="BSND", output_final_state=True,
        return_intermediates=True,
    )

    torch_inputs = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in arrays
    )
    ref_output, ref_state = _torch_reference(*torch_inputs, scale, 64)
    np.testing.assert_allclose(output.asnumpy(), ref_output.detach().numpy(), rtol=0.15, atol=0.03)
    np.testing.assert_allclose(final_state.asnumpy(), ref_state.detach().numpy(), rtol=0.15, atol=0.03)

    sensitivity = np.random.default_rng(12).uniform(-0.04, 0.04, output.shape).astype(np.float32)
    ref_loss = (ref_output * torch.tensor(sensitivity, dtype=torch.float64)).sum()
    ref_grads = torch.autograd.grad(ref_loss, torch_inputs)

    def loss_fn(*inputs):
        actual, _ = npu_chunk_kda(*inputs, scale, 64, layout="BSND")
        return (actual.float() * ms.Tensor(sensitivity)).sum()

    grads = ms.grad(loss_fn, grad_position=(0, 1, 2, 3, 4))(*ms_inputs)
    explicit = npu_chunk_kda_bwd(
        ms_inputs[0], ms_inputs[1], ms_inputs[2], ms_inputs[4],
        *intermediates, ms.Tensor(sensitivity, dtype=output.dtype),
        scale, 64, layout="BSND",
    )
    explicit_grads = (explicit[0], explicit[1], explicit[2], explicit[4], explicit[3])
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(
            actual.asnumpy(), expected.detach().numpy(), rtol=0.30, atol=0.03
        )
    for actual, expected in zip(explicit_grads, grads):
        np.testing.assert_allclose(
            actual.asnumpy(), expected.asnumpy(), rtol=0.01, atol=0.01
        )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_chunk_kda_raw_gate_and_varlen_are_finite():
    """Cover raw-gate gradients and packed metadata in one-card execution."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    rng = np.random.default_rng(13)
    tokens, heads, dim = 64, 2, 128
    tensors = (
        ms.Tensor(rng.uniform(-0.03, 0.03, (tokens, heads, dim)).astype(np.float16)),
        ms.Tensor(rng.uniform(-0.03, 0.03, (tokens, heads, dim)).astype(np.float16)),
        ms.Tensor(rng.uniform(-0.03, 0.03, (tokens, heads, dim)).astype(np.float16)),
        ms.Tensor(rng.uniform(-0.1, 0.1, (tokens, heads, dim)).astype(np.float32)),
        ms.Tensor(rng.uniform(0.0, 1.0, (tokens, heads)).astype(np.float32)),
    )
    a_log = ms.Tensor(rng.uniform(-6.0, -2.0, (heads,)).astype(np.float32))
    dt_bias = ms.Tensor(rng.uniform(-0.2, 0.2, (heads * dim,)).astype(np.float32))
    scale = 1.0 / math.sqrt(dim)

    def loss_fn(q, k, v, g, beta, a, bias):
        output, _ = npu_chunk_kda(
            q, k, v, g, beta, scale, 64, layout="TND",
            safe_gate=True, use_gate_in_kernel=True, a_log=a, dt_bias=bias,
            cu_seqlens=[0, 31, 64],
        )
        return output.float().sum()

    grads = ms.grad(loss_fn, grad_position=(0, 1, 2, 3, 4, 5, 6))(
        *tensors, a_log, dt_bias
    )
    assert all(np.isfinite(grad.asnumpy()).all() for grad in grads)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_chunk_kda_forward_supports_extended_forward_features():
    """Cover forward-only capabilities that are not supported by backward."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    rng = np.random.default_rng(14)
    batch, tokens, heads, key_dim, value_dim = 1, 96, 1, 64, 256
    tensors = (
        ms.Tensor(rng.uniform(-0.03, 0.03, (batch, tokens, heads, key_dim)).astype(np.float16)),
        ms.Tensor(rng.uniform(-0.03, 0.03, (batch, tokens, heads, key_dim)).astype(np.float16)),
        ms.Tensor(rng.uniform(-0.03, 0.03, (batch, tokens, heads, value_dim)).astype(np.float16)),
        ms.Tensor(rng.uniform(-0.01, -0.001, (batch, tokens, heads, key_dim)).astype(np.float32)),
        ms.Tensor(rng.uniform(0.0, 1.0, (batch, tokens, heads)).astype(np.float32)),
    )
    initial_state = ms.Tensor(
        rng.uniform(-0.02, 0.02, (2, heads, value_dim, key_dim)).astype(np.float32)
    )

    output, final_state = npu_chunk_kda_fwd(
        *tensors, 1.0 / math.sqrt(key_dim), 128, layout="BSND",
        output_final_state=True, cu_seqlens=[0, 31, tokens],
        initial_state=initial_state, state_v_first=True,
    )

    assert output.shape == (batch, tokens, heads, value_dim)
    assert final_state.shape == (2, heads, value_dim, key_dim)
    assert np.isfinite(output.asnumpy()).all()
    assert np.isfinite(final_state.asnumpy()).all()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_chunk_kda_rejects_final_state_gradient():
    """Do not silently discard a loss contribution through final_state."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    tensors = tuple(ms.Tensor(value) for value in _inputs(tokens=64, value_heads=1))
    scale = 1.0 / math.sqrt(128)

    def loss_fn(q):
        _, state = npu_chunk_kda(
            q, *tensors[1:], scale, 64, layout="BSND", output_final_state=True
        )
        return state.sum()

    with pytest.raises(RuntimeError, match="final_state is not differentiable"):
        ms.grad(loss_fn)(tensors[0])
