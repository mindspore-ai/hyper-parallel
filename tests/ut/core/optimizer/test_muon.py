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

from typing import Tuple, Union
from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.core.optimizer.muon import Muon
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


@pytest.mark.parametrize("mode", ["legacy", "vanilla"])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("momentum", [0.0, 0.95, (0.7, 0.9)])
def test_muon_momentum_matches_multistep_scalar_reference(
        mode: str, nesterov: bool, momentum: Union[float, Tuple[float, float]],
) -> None:
    """Feature: Selectable Muon momentum formulas.

    Description: Compare both modes and Nesterov settings with multistep scalar recurrences.
    Expectation: BF16 updates and persistent momentum buffers match the reference.
    """
    parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
    optimizer = Muon([parameter], momentum=momentum, momentum_mode=mode, nesterov=nesterov)
    update_beta, buffer_beta = momentum if isinstance(momentum, tuple) else (momentum, momentum)
    reference_buffer = [0.0] * 4
    gradients = [[1.0, -2.0, 0.5, 3.0], [-0.5, 1.0, -4.0, 0.25], [2.0, 0.0, 1.0, -1.0]]

    for gradient in gradients:
        parameter.grad = torch.tensor(gradient, dtype=torch.float64).view(2, 2)
        expected_update = []
        next_buffer = []
        for grad, old_buffer in zip(gradient, reference_buffer):
            if mode == "legacy":
                candidate = grad + update_beta * old_buffer
                update = grad + update_beta * candidate if nesterov else candidate
                buffer = grad + buffer_beta * old_buffer
            else:
                candidate = (1 - update_beta) * grad + update_beta * old_buffer
                update = (1 - update_beta) * grad + update_beta * candidate if nesterov else candidate
                buffer = (1 - buffer_beta) * grad + buffer_beta * old_buffer
            expected_update.append(update)
            next_buffer.append(buffer)

        actual = optimizer._update_muon_momentum(  # pylint: disable=protected-access
            optimizer.param_groups[0], [parameter])
        expected = torch.tensor(expected_update, dtype=torch.float64).view(2, 2).to(torch.bfloat16)
        torch.testing.assert_close(actual[parameter], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"],
            torch.tensor(next_buffer, dtype=torch.float64).view(2, 2), rtol=1e-14, atol=1e-14,
        )
        reference_buffer = next_buffer


def test_muon_default_step_matches_explicit_legacy() -> None:
    """Feature: Backward-compatible Muon momentum defaults.

    Description: Run the default and explicit legacy optimizers with identical gradients.
    Expectation: Parameter trajectories remain identical across multiple steps.
    """
    first = torch.nn.Parameter(torch.eye(2))
    second = torch.nn.Parameter(torch.eye(2))
    default = Muon([first])
    legacy = Muon([second], momentum_mode="legacy")

    for step in range(3):
        gradient = torch.tensor([[1.0 + step, -0.5], [0.25, 2.0 - step]])
        first.grad = gradient.clone()
        second.grad = gradient.clone()
        default.step()
        legacy.step()
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["unknown", None, 1])
def test_muon_rejects_invalid_momentum_mode(mode: object) -> None:
    """Feature: Muon momentum configuration validation.

    Description: Construct Muon with unsupported momentum mode values.
    Expectation: Invalid modes raise a descriptive ValueError.
    """
    with pytest.raises(ValueError, match="momentum_mode"):
        Muon([torch.nn.Parameter(torch.ones(2, 2))], momentum_mode=mode)


def test_muon_rejects_invalid_stored_momentum_mode() -> None:
    """Feature: Muon momentum runtime validation.

    Description: Attempt a step after the stored momentum mode has become invalid.
    Expectation: The optimizer rejects the invalid mode instead of updating the parameter.
    """
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = Muon([parameter])
    optimizer._update_muon_momentum_mode = "unknown"  # pylint: disable=protected-access
    parameter.grad = torch.ones_like(parameter)

    with pytest.raises(ValueError, match="momentum_mode"):
        optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
