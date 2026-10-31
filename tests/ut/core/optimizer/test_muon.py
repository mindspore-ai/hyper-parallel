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

"""Unit tests for the Muon optimizer."""

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.core.optimizer.muon import Muon, zeropower_via_newtonschulz5
from hyper_parallel.core.optimizer.muon_shard import build_pad_ns_inputs
from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer


def _plain_ns_group(**overrides):
    """Group dict covering every key consumed by the batched NS paths."""
    group = {
        "lr": 0.01,
        "matched_adamw_rms": 0.2,
        "ns_steps": 1,
        "ns_variant": "asym5",
        "ns_coefficients": None,
        "ns_epsilon": 1e-10,
        "zero_rms_scale_mode": "zero",
        "apply_lr_in_update": False,
    }
    group.update(overrides)
    return group


def test_compute_batched_ns_updates_accepts_plain_parameter_without_sharding():
    """The no-shard path should support ordinary torch parameters."""
    optimizer = Muon.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = None
    optimizer.zeropower_fn = None
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    ns_inputs = {parameter: torch.ones(2, 2)}
    group = _plain_ns_group()

    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], ns_inputs, group, no_shard=True)

    assert updates[parameter].shape == parameter.shape


def test_compute_batched_ns_updates_with_transform_accepts_plain_parameter():
    """The ns_transform_fn routing path should also support plain parameters."""
    optimizer = Muon.__new__(Muon)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = lambda param_fqn, ns_input: None
    optimizer.zeropower_fn = None
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    ns_inputs = {parameter: torch.ones(2, 2)}
    group = _plain_ns_group()

    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], ns_inputs, group, no_shard=True)

    assert updates[parameter].shape == parameter.shape


def test_muon_rejects_invalid_ns_variant():
    """ns_variant outside legacy/asym5/custom must raise."""
    with pytest.raises(ValueError, match="ns_variant"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_variant="unknown")


def test_muon_rejects_ns_coefficients_without_custom_variant():
    """ns_coefficients is only valid together with ns_variant='custom'."""
    with pytest.raises(ValueError, match="ns_coefficients"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="asym5",
            ns_coefficients=[(3.4445, -4.7750, 2.0315)],
        )


def test_muon_rejects_missing_or_short_ns_coefficients():
    """Custom NS requires at least ns_steps coefficient groups."""
    with pytest.raises(ValueError, match="ns_coefficients"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_variant="custom")
    with pytest.raises(ValueError, match="exceeds"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=2,
            ns_coefficients=[(3.4445, -4.7750, 2.0315)],
        )


def test_muon_rejects_malformed_ns_coefficients():
    """Each custom coefficient group must hold three finite values."""
    with pytest.raises(ValueError, match="exactly three"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=1,
            ns_coefficients=[(3.4445, -4.7750)],
        )
    with pytest.raises(ValueError, match="finite"):
        Muon(
            [torch.nn.Parameter(torch.ones(2, 2))],
            ns_variant="custom",
            ns_steps=1,
            ns_coefficients=[(3.4445, float("inf"), 2.0315)],
        )


def test_muon_rejects_invalid_ns_epsilon_and_zero_rms_scale_mode():
    """ns_epsilon must be finite non-negative; zero_rms_scale_mode must be known."""
    with pytest.raises(ValueError, match="ns_epsilon"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], ns_epsilon=-1e-10)
    with pytest.raises(ValueError, match="zero_rms_scale_mode"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], zero_rms_scale_mode="keep")


def test_async_hierarchical_broadcast_waits_between_mesh_dimensions():
    """A relay dimension must wait before forwarding the shared buffer."""
    first_work = MagicMock()
    final_work = MagicMock()
    process_groups = (MagicMock(), MagicMock())

    with patch(
            "hyper_parallel.core.optimizer.optimizer.dist.get_global_rank",
            side_effect=[0, 1],
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.dist.broadcast",
            side_effect=[first_work, final_work],
    ) as broadcast:
        handles = BaseDistributedOptimizer._hierarchical_broadcast_buffer_async(  # pylint: disable=protected-access
            torch.ones(4),
            src_coord=(0, 0),
            replicate_pgs=process_groups,
            local_coord=(0, 0),
        )

    assert broadcast.call_count == 2
    first_work.wait.assert_called_once_with()
    final_work.wait.assert_not_called()
    assert handles == [final_work]


def test_ns_normalizes_in_fp32_before_low_precision_compute():
    """Clamping applies to the FP32 norm, including zero and sub-epsilon inputs."""
    values = torch.tensor([[[3e-10], [4e-10]], [[0.0], [0.0]]])

    direction = zeropower_via_newtonschulz5(
        values, steps=0, epsilon=1e-8, epsilon_mode="clamp",
        compute_dtype=torch.bfloat16)

    # The first Frobenius norm is 5e-10, so the denominator is exactly epsilon.
    expected = torch.tensor([[[0.03], [0.04]], [[0.0], [0.0]]], dtype=torch.bfloat16)
    torch.testing.assert_close(direction, expected, rtol=0, atol=0)
    assert direction.dtype == torch.bfloat16
    assert torch.isfinite(direction).all()


@pytest.mark.parametrize("use_transform", [False, True])
def test_muon_precision_policy_preserves_normalization_and_scaling_boundaries(use_transform):
    """Both restoration paths scale in the iteration dtype before restoring buffers."""
    optimizer = Muon.__new__(Muon)
    optimizer.state = defaultdict(dict)
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = (lambda _name, _tensor: None) if use_transform else None
    optimizer.zeropower_fn = None
    optimizer.momentum_update_fn = None
    parameter = torch.nn.Parameter(torch.zeros(2, 1))
    parameter.model_name = "weight"
    parameter.grad = torch.tensor([[3e-9], [4e-9]])
    group = _plain_ns_group(
        momentum=0.95, nesterov=False, ns_variant="custom",
        ns_coefficients=((1.0, 0.0, 0.0),), ns_epsilon=1e-8,
        ns_input_dtype="float32", ns_compute_dtype="bfloat16", ns_epsilon_mode="clamp",
        lr=1e-5, apply_lr_in_update=True)

    inputs = optimizer._update_muon_momentum(group, [parameter])  # pylint: disable=protected-access
    assert inputs[parameter].dtype == torch.float32
    torch.testing.assert_close(inputs[parameter], parameter.grad, rtol=0, atol=0)
    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], inputs, group, no_shard=True)

    # One identity polynomial leaves the clamped direction [0.3, 0.4].
    # Quantize that direction, then scale once in BF16; restore into FP32 storage.
    direction = torch.tensor([[0.3], [0.4]], dtype=torch.bfloat16)
    expected = (direction * (-1e-5 * (2 ** 0.5) * 0.2)).float()
    assert updates[parameter].dtype == torch.float32
    torch.testing.assert_close(updates[parameter], expected, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], parameter.grad, rtol=0, atol=0)


def test_ns_padding_uses_explicit_dtype_on_empty_ranks():
    """A rank without gradients must use the same collective dtype as populated ranks."""
    parameter = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
    ns_input = torch.ones_like(parameter, dtype=torch.bfloat16)
    populated = build_pad_ns_inputs([parameter], {parameter: ns_input}, dtype=torch.bfloat16)
    empty = build_pad_ns_inputs([parameter], {}, dtype=torch.bfloat16)

    assert populated[0] is ns_input
    assert empty[0].dtype == populated[0].dtype
    torch.testing.assert_close(empty[0], torch.zeros_like(ns_input), rtol=0, atol=0)
    with pytest.raises(ValueError, match="same dtype"):
        build_pad_ns_inputs([parameter], {parameter: ns_input}, dtype=torch.float32)


@pytest.mark.parametrize("options", [
    {"ns_input_dtype": "float16"},
    {"ns_compute_dtype": "float16"},
    {"ns_epsilon_mode": "unknown"},
])
def test_muon_rejects_unsupported_precision_options(options):
    """Invalid precision policies fail before distributed setup."""
    with pytest.raises(ValueError, match="ns_"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], **options)


def test_muon_rejects_built_in_precision_policy_with_custom_ns():
    """Custom callbacks cannot silently ignore built-in normalization options."""
    with pytest.raises(ValueError, match="built-in"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))],
             zeropower_fn=lambda value, steps: value, ns_epsilon_mode="clamp")
