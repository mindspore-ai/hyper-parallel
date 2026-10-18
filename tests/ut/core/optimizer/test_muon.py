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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Shard, StridedShard
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.muon_shard import build_param_shard_metadata_for_group, chunk_update_by_layout
from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer
from hyper_parallel.core.optimizer.sharding_category import HSDPCommGroup, ParamLayoutSpec, ParamRecord


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


@pytest.mark.parametrize("fsdp_rank", [0, 20])
@pytest.mark.parametrize("fake", [False, True])
def test_muon_metadata_handles_empty_fsdp_shards_before_tp(fsdp_rank, fake):
    """FSDP gathers only TP-local rows, including ranks with empty shards."""
    layout = Layout((64, 2), ("fsdp", "tp"), init_backend=False)
    layout.set_placements((StridedShard(0, split_factor=2, uneven_shard=True), Shard(0)))
    layout.placement_to_tensor_map(2)
    mesh = MagicMock()
    mesh.get_coordinate.return_value = (fsdp_rank, 0)
    mesh.size.side_effect = lambda dim: (64, 2)[dim]
    spec = ParamLayoutSpec(tensor_ndim=2, shard_axes=((0, 0), (1, 0)), replicate_mesh_dims=())
    groups = (MagicMock(), MagicMock())
    with FakeTensorMode() if fake else nullcontext():
        local = torch.empty((1 if fsdp_rank < 16 else 0, 4))
        parameter = MagicMock(data=local, shape=(32, 4))
        parameter.layout = layout
        parameter.device_mesh = mesh
        parameter._global_shape = (32, 4)
        group = HSDPCommGroup(comm_key=None, layout_spec=spec, shard_pgs=groups,
                              records=[ParamRecord(index=0, param=parameter)])
        with patch("hyper_parallel.core.optimizer.muon_shard.dist.get_world_size", side_effect=[64, 2]), \
                patch("hyper_parallel.core.optimizer.muon_shard.dist.get_rank", side_effect=[fsdp_rank, 0]):
            meta = build_param_shard_metadata_for_group(group)[parameter]
        assert meta.stage_metas[0].split_sizes == (1,) * 16 + (0,) * 48
        assert meta.stage_metas[1].split_sizes == (16, 16)
        local_update = chunk_update_by_layout(torch.empty(32, 4), parameter, spec, meta)
        assert local_update.shape == local.shape


def test_fake_muon_reshape_rejects_independent_storage():
    """FakeTensor data pointers are zero even when the storages do not alias."""
    optimizer = Muon.__new__(Muon)
    optimizer.reshape_fn = lambda _, value: [value.clone()]
    with FakeTensorMode():
        parameter = torch.nn.Parameter(torch.empty(4, 4))
        parameter.model_name = "weight"
        with pytest.raises(ValueError, match="share storage"):
            optimizer._reshape_ns_input(parameter, torch.empty(4, 4))


def test_fake_subgroup_creation_uses_local_rank_membership():
    """Fake process groups create only the subgroup owned by this virtual rank."""
    optimizer = object.__new__(BaseDistributedOptimizer)
    optimizer._split_sub_pg_cache = {}
    parent_group = MagicMock()
    subgroup = MagicMock()
    with patch(
            "hyper_parallel.core.optimizer.optimizer.dist.get_process_group_ranks",
            return_value=[0, 1, 2, 3],
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.dist.get_backend",
            return_value="fake",
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.dist.get_rank",
            return_value=1,
    ), patch(
            "hyper_parallel.core.optimizer.optimizer.dist.new_group",
            return_value=subgroup,
    ) as new_group, patch(
            "hyper_parallel.core.optimizer.optimizer.dist.all_gather_into_tensor",
    ) as all_gather:
        result = optimizer._get_or_create_sub_pg(parent_group, 2)

    assert result is subgroup
    new_group.assert_called_once_with(ranks=[0, 1], use_local_synchronization=True)
    all_gather.assert_not_called()
