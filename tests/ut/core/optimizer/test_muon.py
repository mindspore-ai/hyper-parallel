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

from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.components.functional import make_gram_newton_schulz
from hyper_parallel.core.optimizer.muon import (
    Muon,
    NSInputTransform,
    _NS_ASYM5_COEFFS,
    _NS_LEGACY_COEFFS,
    zeropower_via_newtonschulz5,
)
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


@pytest.mark.parametrize("input_mode", ["plain", "reshape", "transform"])
def test_muon_gram_callback_restores_updates(input_mode: str) -> None:
    """Feature: Gram NS callback integration.

    Description: Run plain, reshape and reversible transform preparation routes.
    Expectation: The two-argument callback restores updates to the original shape.
    """
    calls = []

    def backend(inputs: torch.Tensor) -> torch.Tensor:
        """Record the prepared shape and return deterministic updates."""
        calls.append(tuple(inputs.shape))
        return inputs.clone().mul_(2)

    implementation = MagicMock(return_value=backend)
    with patch(
        "hyper_parallel.components.functional.gram_newton_schulz._load_gram_newton_schulz",
        return_value=implementation,
    ):
        callback = make_gram_newton_schulz([(1, 0, 0)] * 3, use_kernels=False)

    optimizer = Muon.__new__(Muon)
    optimizer.zeropower_fn = callback
    optimizer.reshape_fn = None
    optimizer.ns_transform_fn = None
    parameter = torch.nn.Parameter(torch.ones(8, 8))
    parameter.model_name = "weight"
    original = torch.arange(64, dtype=torch.float32).view(8, 8)

    if input_mode == "reshape":
        optimizer.reshape_fn = lambda name, tensor: [tensor.view(4, 2, 8)]
    elif input_mode == "transform":
        def transform(name: str, tensor: torch.Tensor) -> NSInputTransform:
            """Split a matrix into slices with a matching restoration callback."""
            assert name == parameter.model_name, f"expected={parameter.model_name}, got={name}"
            def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
                """Join transformed slice updates in their original order."""
                output.copy_(torch.cat(updates, dim=0))

            return NSInputTransform(tensors=[tensor[:4], tensor[4:]], restore=restore)

        optimizer.ns_transform_fn = transform

    updates = optimizer._compute_batched_ns_updates(  # pylint: disable=protected-access
        [parameter], {parameter: original.clone()},
        _plain_ns_group(ns_steps=3, matched_adamw_rms=0, zero_rms_scale_mode="use_lr"), no_shard=True,
    )
    torch.testing.assert_close(updates[parameter], original * 2)
    expected_shapes = {"plain": (8, 8), "reshape": (4, 2, 8), "transform": (2, 4, 8)}
    assert calls == [expected_shapes[input_mode]], f"expected={expected_shapes[input_mode]}, got={calls}"
    assert len(implementation.call_args.kwargs["ns_coefficients"]) == 3, (
        f"expected=3 coefficient groups, got={implementation.call_args.kwargs['ns_coefficients']}"
    )


@pytest.mark.parametrize("variant,steps", [("legacy", 1), ("legacy", 6), ("asym5", 1), ("asym5", 5), ("asym5", 6)])
def test_native_ns_coefficient_resolution_preserves_updates(variant: str, steps: int) -> None:
    """Feature: Native NS coefficient resolution.

    Description: Compare legacy/asym5 schedules against explicit custom schedules.
    Expectation: Updates remain bitwise equal, including asym5 step truncation.
    """
    coefficients = (_NS_LEGACY_COEFFS,) * steps if variant == "legacy" else _NS_ASYM5_COEFFS[:steps]
    inputs = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(17))
    expected = zeropower_via_newtonschulz5(
        inputs, steps=len(coefficients), ns_variant="custom", ns_coefficients=coefficients,
    )
    actual = zeropower_via_newtonschulz5(inputs, steps=steps, ns_variant=variant)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
