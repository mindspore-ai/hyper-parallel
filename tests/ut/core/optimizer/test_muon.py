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
from unittest.mock import MagicMock, call, patch

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
from hyper_parallel.core.optimizer.muon_shard import (
    build_pad_ns_inputs,
    build_param_shard_metadata_for_group,
    fused_allgather_dtensor_params,
)
from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer
from hyper_parallel.core.optimizer.sharding_category import CommDomainKey, HSDPCommGroup, ParamLayoutSpec, ParamRecord


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


def test_build_shard_metadata_updates_existing_record():
    """Refresh shard metadata without invalidating a batch's parameter record."""
    parameter = torch.nn.Parameter(torch.ones(3, 4))
    record = ParamRecord(0, parameter)
    domain = CommDomainKey((1,), (0,), (), (0,))
    layout = ParamLayoutSpec(2, ((0, 0),), ())
    records = [record]
    group = HSDPCommGroup(domain, layout, shard_pgs=(None,), records=records)

    first_meta = build_param_shard_metadata_for_group(group)[parameter]
    latest_meta = build_param_shard_metadata_for_group(group)[parameter]

    assert group.records is records
    assert group.records[0] is record
    assert record.shard_meta is latest_meta
    assert first_meta is not latest_meta
    assert latest_meta.stage_metas[0].split_sizes == (3,)


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


def test_muon_overlap_spreads_local_batches_by_default() -> None:
    """Split five local batches across one shard batch's gather and broadcast windows."""
    local_params = [torch.nn.Parameter(torch.ones(2, cols)) for cols in range(2, 7)]
    for param in local_params:
        param.grad = torch.ones_like(param)
    shard_param = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = Muon([shard_param, *local_params], overlap_shard_comm=True)
    shard_group = MagicMock()
    assignment = MagicMock(owned_params=[shard_param], shard_pgs=(shard_group,), is_shard=True,
                           is_replicated=False)
    optimizer._hsdp_assignment_batches = {  # pylint: disable=protected-access
        0: {"batch_groups": [{"sub_batches": [assignment]}], "no_comm": local_params},
    }
    calls = MagicMock()

    def run_shard_collectives(*_args, overlap_work, **_kwargs):
        """Run the callback once during gather and once during broadcast."""
        calls.gather()
        overlap_work()
        calls.broadcast()
        overlap_work()
        calls.apply()

    with patch("hyper_parallel.core.optimizer.muon.dist.get_world_size", return_value=2), \
            patch.object(optimizer, "_shard_pipeline_key", return_value=(torch.device("cpu"), (shard_group,), ())), \
            patch.object(optimizer, "_process_shard_params", side_effect=run_shard_collectives), \
            patch.object(optimizer, "_process_unshard_params", side_effect=lambda *_: calls.local()):
        optimizer.step()

    expected = [call.gather(), call.local(), call.local(), call.broadcast(),
                call.local(), call.local(), call.local(), call.apply()]
    assert calls.mock_calls == expected, f"expected {expected}, got {calls.mock_calls}"


@pytest.mark.parametrize("local_error", [False, True])
def test_muon_async_gather_waits_after_local_work(local_error: bool) -> None:
    """Feature: Run independent Muon work during an asynchronous gather.

    Description: Mock a gather whose output only becomes readable when its work is waited on.
    Expectation: Local work runs before the wait, including when local work raises.
    """
    calls = MagicMock()
    if local_error:
        calls.local_work.side_effect = RuntimeError("local work failed")
    group = MagicMock()
    layout = ParamLayoutSpec(tensor_ndim=2, shard_axes=((0, 0),), replicate_mesh_dims=())
    local = torch.arange(1, 3, dtype=torch.bfloat16).view(1, 2)

    def gather(output, packed, *, group, async_op):
        """Expose gathered data only when the caller waits for the handle."""
        calls.launch(group, async_op)
        calls.wait.side_effect = lambda: output.copy_(packed.repeat(2))
        return calls

    with patch("hyper_parallel.core.optimizer.muon_shard.dist") as dist:
        dist.get_world_size.return_value = 2
        dist.all_gather_into_tensor.side_effect = gather
        if local_error:
            with pytest.raises(RuntimeError, match="local work failed"):
                fused_allgather_dtensor_params([local], [group], layout, while_gathering=calls.local_work)
        else:
            result, = fused_allgather_dtensor_params([local], [group], layout, while_gathering=calls.local_work)
            torch.testing.assert_close(result, local.repeat(2, 1), rtol=0, atol=0)

    expected = [call.launch(group, True), call.local_work(), call.wait()]
    assert calls.mock_calls == expected, f"expected {expected}, got {calls.mock_calls}"


def test_muon_pad_ns_inputs_uses_fallback_only_without_real_inputs() -> None:
    """Feature: Match the communication dtype when a rank has no valid gradient.

    Description: Construct padding for float32 parameters with and without a real NS input.
    Expectation: An explicit fallback controls empty input batches; real inputs take precedence.
    """
    first = torch.nn.Parameter(torch.ones(2, 2))
    second = torch.nn.Parameter(torch.ones(2, 2))
    inputs = build_pad_ns_inputs([first], {}, fallback_dtype=torch.bfloat16)
    torch.testing.assert_close(inputs[0], torch.zeros((2, 2), dtype=torch.bfloat16))

    reference = torch.ones((2, 2), dtype=torch.float64)
    inputs = build_pad_ns_inputs([first, second], {first: reference}, fallback_dtype=torch.bfloat16)
    torch.testing.assert_close(inputs[1], torch.zeros_like(reference))


def test_muon_gather_reuses_owner_slot_with_column_shards() -> None:
    """Feature: Reusable pipeline storage for owner NS inputs.

    Description: Gather column shards with padding and unpack only the owner's matrix into reusable storage.
    Expectation: Reused storage contains the next gather and never aliases the raw gather buffer.
    """
    layout = ParamLayoutSpec(tensor_ndim=2, shard_axes=((0, 1),), replicate_mesh_dims=())
    group = MagicMock()
    local = torch.arange(6, dtype=torch.bfloat16).view(3, 2)
    gather_buffers, owner_buffers = {}, {}

    def gather(output, packed, *, group):  # pylint: disable=unused-argument
        """Return a copy of the local shard for each rank."""
        output.copy_(packed.repeat(2))

    with patch("hyper_parallel.core.optimizer.muon_shard.dist") as dist:
        dist.get_world_size.return_value = 2
        dist.all_gather_into_tensor.side_effect = gather
        first, skipped = fused_allgather_dtensor_params(
            [local, local], [group], layout,
            buffer_cache=gather_buffers, keep_indices={0}, owner_buffer_cache=owner_buffers,
        )
        torch.testing.assert_close(first, local.repeat(1, 2), rtol=0, atol=0)
        assert skipped is None, f"Unowned input should be empty, got {skipped}"
        second, = fused_allgather_dtensor_params(
            [local + 1], [group], layout,
            buffer_cache=gather_buffers, keep_indices={0}, owner_buffer_cache=owner_buffers,
        )

    torch.testing.assert_close(second, (local + 1).repeat(1, 2), rtol=0, atol=0)
    assert first.data_ptr() == second.data_ptr(), f"Owner buffer was replaced: {first.data_ptr()}, {second.data_ptr()}"
    assert all(tmp.data_ptr() != second.data_ptr() for tmp in gather_buffers.values()), (
        f"Owner buffer {second.data_ptr()} aliases a raw gather buffer"
    )


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


@pytest.mark.parametrize("dtypes", [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
])
@pytest.mark.parametrize("mode", ["legacy", "vanilla"])
def test_muon_mixed_dtype_momentum_returns_per_tensor_bfloat16_inputs(
        dtypes: Tuple[torch.dtype, torch.dtype], mode: str,
) -> None:
    """Feature: Mixed-dtype Muon momentum preparation.

    Description: Compute one momentum update for BF16 and FP32 parameters in both orders.
    Expectation: Every Newton-Schulz input is BF16 regardless of parameter order.
    """
    parameters = [
        torch.nn.Parameter(torch.zeros(2, 2, dtype=dtype))
        for dtype in dtypes
    ]
    optimizer = Muon(parameters, momentum=0.95, nesterov=True, momentum_mode=mode)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    with patch.object(torch, "_foreach_add", wraps=torch._foreach_add) as foreach_add, \
            patch.object(torch, "_foreach_lerp", wraps=torch._foreach_lerp) as foreach_lerp:
        updates = optimizer._update_muon_momentum(  # pylint: disable=protected-access
            optimizer.param_groups[0], parameters,
        )

    actual_dtypes = tuple(updates[parameter].dtype for parameter in parameters)
    expected_dtypes = (torch.bfloat16, torch.bfloat16)
    assert actual_dtypes == expected_dtypes, (
        f"Muon NS input dtype mismatch: expected={expected_dtypes}, actual={actual_dtypes}, "
        f"parameter_dtypes={dtypes}"
    )
    foreach_calls = foreach_add.call_args_list + foreach_lerp.call_args_list
    actual_foreach_dtypes = [
        tuple(tensor.dtype for tensor in call.args[0])
        for call in foreach_calls
    ]
    expected_foreach_dtypes = [(dtypes[0],), (dtypes[1],)]
    assert actual_foreach_dtypes == expected_foreach_dtypes, (
        f"Muon foreach dtype regions mismatch: expected={expected_foreach_dtypes}, "
        f"actual={actual_foreach_dtypes}"
    )


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
