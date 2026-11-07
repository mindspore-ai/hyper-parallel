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

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.core.optimizer.muon import Muon, NSInputTransform
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


def test_flat_experts_preserve_muon_callbacks_and_resume():
    """Feature: Logical matrices. Description: Resume with callbacks. Expectation: Equal weights and momentum."""
    def split(_name: str, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Return independent gate/up views."""
        return list(tensor.chunk(2, dim=-1))

    def transform(name: str, tensor: torch.Tensor) -> NSInputTransform:
        """Exercise the reversible callback on logical matrices."""
        def restore(updates: list[torch.Tensor], output: torch.Tensor) -> None:
            """Restore both projection blocks."""
            output.copy_(torch.cat(updates, dim=-1))
        return NSInputTransform(split(name, tensor), restore)

    def parameter(value: torch.Tensor, flat: bool) -> torch.nn.Parameter:
        """Keep metadata attached when rebuilding an optimizer for resume."""
        result = torch.nn.Parameter(value.flatten(0, 1).clone() if flat else value.clone())
        result.model_name = "experts"
        if flat:
            result.logical_shape = tuple(value.shape)
        return result

    for options in ({}, {"reshape_fn": split}, {"ns_transform_fn": transform}):
        source = torch.randn(3, 4, 12, dtype=torch.bfloat16)
        params = [parameter(source, flat) for flat in (False, True)]
        optimizers = [Muon([param], ns_steps=5, **options) for param in params]
        for step in range(2):
            gradient = torch.randn_like(source)
            for param, optimizer in zip(params, optimizers):
                param.grad = gradient.view_as(param)
                optimizer.step()
            torch.testing.assert_close(params[1].view_as(params[0]), params[0], rtol=0, atol=0)
            left, right = [optimizer.state[param] for optimizer, param in zip(optimizers, params)]
            for key, value in left.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(right[key].view_as(value), value, rtol=0, atol=0)
            if step == 0:
                state = copy.deepcopy(optimizers[1].state_dict())
                params[1] = parameter(params[1].detach().view_as(source), True)
                optimizers[1] = Muon([params[1]], ns_steps=5, **options)
                optimizers[1].load_state_dict(state)
    for shape in ((3, 4, 5), (3, 0, 4), (True, 12, 4)):
        params[1].logical_shape = shape
        with pytest.raises(ValueError, match="logical_shape"):
            Muon._logical_ns_input(params[1], torch.ones(12, 12))  # pylint: disable=protected-access
