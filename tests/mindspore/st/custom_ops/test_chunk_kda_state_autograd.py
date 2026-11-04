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
"""State-aware total interface against independent FP32 CPU autograd."""
from typing import Any

import numpy as np
import pytest
import mindspore as ms

from hyper_parallel.custom_ops.experimental import npu_chunk_kda
from hyper_parallel.platform.mindspore.custom_ops import chunk_kda_autograd as adapter
from tests.mindspore.st.custom_ops import test_chunk_kda_stages as stage_tests
from tests.mindspore.st.custom_ops.test_chunk_kda_stages import (
    _inputs, _reference, _check,
)

require_fla_stages = stage_tests.require_fla_stages


@pytest.mark.parametrize("recompute,v_first,cu", [
    (False, False, None), (True, False, None), (True, True, None),
    (False, True, [0, 1, 65]), (True, False, [0, 1, 65]),
])
def test_total_state_autograd_reference(recompute, v_first, cu):
    """Exercise output loss + final loss + dh0 with nonvanishing long memory."""
    inputs, d_o = _inputs(65)
    inputs = (*inputs[:3], ms.Tensor(np.full(inputs[3].shape, -6., np.float32)),
              inputs[4], ms.Tensor(np.zeros(1, np.float32)), inputs[6])
    rng = np.random.default_rng(912)
    shape = (2 if cu else 1, 1, 128, 128)
    h0 = ms.Tensor(rng.normal(0, .05, shape).astype(np.float32), ms.bfloat16)
    dht = ms.Tensor(rng.normal(0, .02, shape).astype(np.float32), ms.bfloat16)
    ref_out, ref_final, ref_grads = _reference(inputs, d_o, cu, h0=h0, dht=dht)
    original_h0 = h0.swapaxes(-1, -2).contiguous() if v_first else h0

    def loss(*values: ms.Tensor) -> ms.Tensor:
        """Differentiate both attention and terminal-state outputs."""
        q, k, v, g, beta, a_log, bias, state = values
        out, final = npu_chunk_kda(
            q, k, v, g, beta, 128**-.5, safe_gate=True, a_log=a_log, dt_bias=bias,
            initial_state=state, output_final_state=True, disable_recompute=not recompute,
            state_v_first=v_first, cu_seqlens=cu,
        )
        terminal = dht.swapaxes(-1, -2) if v_first else dht
        return (out * d_o.transpose(0, 2, 1, 3)).astype(ms.float32).sum() + (
            final.astype(ms.float32) * terminal.astype(ms.float32)).sum()

    grads = ms.grad(loss, grad_position=tuple(range(8)))(*inputs, original_h0)
    out, final = npu_chunk_kda(
        *inputs[:5], 128**-.5, a_log=inputs[5], dt_bias=inputs[6], safe_gate=True,
        initial_state=original_h0, output_final_state=True, disable_recompute=not recompute,
        state_v_first=v_first, cu_seqlens=cu,
    )
    _check(out.transpose(0, 2, 1, 3), ref_out, "total_out")
    _check(final.swapaxes(-1, -2) if v_first else final, ref_final, "total_final")
    for index, (actual, expected) in enumerate(zip(grads, ref_grads)):
        if index == 7 and v_first:
            actual = actual.swapaxes(-1, -2)
        _check(actual, expected, f"total_grad_{index}")


def test_total_final_state_only_loss():
    """A loss on final_state alone must still differentiate all token inputs."""
    inputs, d_o = _inputs(65)
    dht = ms.Tensor(np.full((1, 1, 128, 128), .01, np.float32), ms.bfloat16)
    _, _, reference = _reference(inputs, ms.mint.zeros_like(d_o), dht=dht)

    def loss(*values: ms.Tensor) -> ms.Tensor:
        """Differentiate only the terminal state."""
        _, final = npu_chunk_kda(*values[:5], 128**-.5, safe_gate=True,
                                  a_log=values[5], dt_bias=values[6], output_final_state=True)
        return (final.astype(ms.float32) * dht.astype(ms.float32)).sum()

    actual = ms.grad(loss, grad_position=tuple(range(7)))(*inputs)
    for index, (value, target) in enumerate(zip(actual, reference)):
        _check(value, target, f"state_only_grad_{index}")


@pytest.mark.parametrize("recompute", [False, True])
def test_total_recompute_rebuilds_relays(monkeypatch, recompute):
    """Require actual reconstruction, not a recompute flag with retained caches."""
    inputs, d_o = _inputs(65)
    modes, saved_names = [], []
    original_prepare = adapter._prepare  # pylint: disable=protected-access
    original_forward = adapter._ChunkKdaStageFunction.forward  # pylint: disable=protected-access

    def prepare(values: tuple, options: dict, mode: str) -> tuple:
        """Record which caches were requested."""
        modes.append(mode)
        return original_prepare(values, options, mode)

    def forward(ctx: Any, *args: Any) -> tuple:
        """Observe cache ownership without retaining tensor references."""
        result = original_forward(ctx, *args)
        saved_names.append(ctx.names)
        return result

    monkeypatch.setattr(adapter, "_prepare", prepare)
    monkeypatch.setattr(adapter._ChunkKdaStageFunction, "forward", staticmethod(forward))  # pylint: disable=protected-access

    def loss(*values: ms.Tensor) -> ms.Tensor:
        """Differentiate the public interface with a token-output loss."""
        output, _ = npu_chunk_kda(
            *values[:5], 128**-.5, safe_gate=True, a_log=values[5], dt_bias=values[6],
            disable_recompute=not recompute,
        )
        return (output.astype(ms.float32) * d_o.transpose(0, 2, 1, 3).astype(ms.float32)).sum()

    gradients = ms.grad(loss, grad_position=tuple(range(7)))(*inputs)
    _, _, expected = _reference(inputs, d_o)
    for index, (actual, target) in enumerate(zip(gradients, expected)):
        _check(actual, target, f"recompute_grad_{index}")
    assert modes == (["none", "save"] if recompute else ["save"])
    assert any(name.startswith("cache") for name in saved_names[0]) is not recompute
    assert ("h" in saved_names[0]) is not recompute


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "NTD", "TND"])
def test_total_grouped_heads_and_layout(layout):
    """Autograd must reverse layout conversion and grouped Q/K repetition."""
    inputs, d_o = _inputs(65, heads=2)
    q, k = (value[:, :1].contiguous() for value in inputs[:2])
    inputs = (q, k, *inputs[2:])
    missing_bias = layout == "TND"
    if missing_bias:
        inputs = (*inputs[:6], ms.mint.zeros_like(inputs[6]))
    reference_inputs = tuple(ms.mint.repeat_interleave(value, 2, dim=1) for value in (q, k)) + inputs[2:]
    _, _, expected = _reference(reference_inputs, d_o, norm=True, sigmoid=True)

    def convert(value: ms.Tensor, scalar: bool = False) -> ms.Tensor:
        """Apply the public layout to token inputs or their gradients."""
        if layout == "BNSD":
            return value
        if layout == "BSND":
            return value.transpose((0, 2, 1) if scalar else (0, 2, 1, 3)).contiguous()
        value = value.squeeze(0)
        if layout == "TND":
            value = value.transpose((1, 0) if scalar else (1, 0, 2))
        return value.contiguous()

    leaves = tuple(convert(value, index == 4) for index, value in enumerate(inputs[:5])) + inputs[5:]

    def loss(*values: ms.Tensor) -> ms.Tensor:
        """Exercise norm/beta fusion together with grouped heads and packing."""
        output, _ = npu_chunk_kda(
            *values[:5], 128**-.5, layout=layout, safe_gate=True, a_log=values[5],
            dt_bias=None if missing_bias else values[6], disable_recompute=False,
            use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            cu_seqlens=[0, 65] if layout in ("NTD", "TND") else None,
        )
        weights = d_o.transpose(0, 2, 1, 3)
        if layout in ("NTD", "TND"):
            weights = weights.squeeze(0)
        return (output.astype(ms.float32) * weights.astype(ms.float32)).sum()

    count = 6 if missing_bias else 7
    actual = ms.grad(loss, grad_position=tuple(range(count)))(*leaves)
    for index, (value, target) in enumerate(zip(actual, expected)):
        if index < 2:
            target = target.sum(dim=1, keepdim=True)
        if index < 5:
            target = ms.Tensor(target.detach().numpy())
            target = convert(target, index == 4)
        _check(value, target, f"grouped_{layout}_grad_{index}")
