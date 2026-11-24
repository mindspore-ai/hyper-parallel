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
"""Unit tests for the family-owned Engram EP execution path."""
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

from examples.torch.module_family.config import EngramConfig
from hyper_parallel.components.modules import Engram as BuildEngram
from hyper_parallel.components.modules.engram import EPEngramModule, EngramModule
from hyper_parallel.components.modules.engram.host_engram import HostEngramModule
from hyper_parallel.components.modules.module_family import FamilyShardingContext, get_family_build_info
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.module_binding import MeshBinding, ParallelBinding
from tests.common.mark_utils import arg_mark


def _assets() -> dict[str, Any]:
    """Return small, disjoint n-gram hash tables."""
    return {
        "layer_ids": [0], "max_ngram_size": 3, "num_heads": 2,
        "token_map": list(range(16)), "pad_token_id": 0,
        "primes": [[[5, 7], [11, 13]]], "multipliers": [[1, 3, 5]],
    }


def _config() -> EngramConfig:
    """Return a padded device-table configuration."""
    return EngramConfig(_assets(), 0, 8, 2, 4, padded_num_embeddings=40)


class _AxisMesh:
    """Expose the mesh methods consumed by the Engram resource binder."""

    def __init__(self, size: int, rank: int, group: object | None) -> None:
        self._size = size
        self._rank = rank
        self._group = group

    def size(self) -> int:
        """Return the logical degree of this test axis."""
        return self._size

    def get_local_rank(self) -> int:
        """Return the local coordinate for this test axis."""
        return self._rank

    def get_group(self) -> object | None:
        """Return the prebuilt process-group placeholder."""
        return self._group


class _MeshRegion:
    """Provide indexed axis lookup with the shape of a DeviceMesh view."""

    def __init__(self, axes: tuple[tuple[str, _AxisMesh], ...]) -> None:
        self._axes = dict(axes)
        self.mesh_dim_names = tuple(self._axes)

    def __getitem__(self, axis: str) -> _AxisMesh:
        """Return the requested test axis mesh."""
        return self._axes[axis]


def _parallel_binding() -> tuple[ParallelBinding, object, object]:
    """Build an EP2 and CP2 binding without a distributed process group."""
    ep_group = object()
    cp_group = object()
    mesh = SimpleNamespace(
        device_mesh=_MeshRegion((("tp", _AxisMesh(1, 0, None)), ("cp", _AxisMesh(2, 1, cp_group)))),
        fsdp_moe_mesh=_MeshRegion((("ep", _AxisMesh(2, 1, ep_group)),)),
    )
    return ParallelBinding(MeshBinding(mesh)), ep_group, cp_group


class TestEngramFamilyParallel(unittest.TestCase):
    """Validate device and Host Engram selection and EP binding."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_public_family_selects_device_and_host_variants(self) -> None:
        """Feature: Engram family selection.

        Description: Build the public module factory for device and Host storage.
        Expectation: It selects the expected family member and its planned layouts.
        """
        layout_context = FamilyShardingContext(
            mesh_dim_names=("ep",),
            axis_sizes={"tp": 1, "cp": 1, "ep": 2, "pp": 1},
            parameter_shapes={},
        )
        device = BuildEngram(config=_config(), context={"ep_size": 2})  # pylint: disable=not-callable
        host = BuildEngram(  # pylint: disable=not-callable
            config=_config(), context={"ep_size": 2}, engram_storage_backend="host",
        )
        self.assertIsInstance(device, EPEngramModule)
        self.assertIsInstance(host, HostEngramModule)

        for member in (device, host):
            with self.subTest(member=type(member).__name__):
                info = get_family_build_info(member)
                self.assertIsNotNone(info)
                sharding = info.resolved.sharding(
                    module=member, module_fqn="model.layers.0.engram",
                    member=info.resolved, context=layout_context,
                )
                self.assertEqual(sharding.params["embed.weight"], {"ep": Shard(0)})
                self.assertEqual(sharding.params["wkv.weight"], {"tp": Replicate()})
                parallel = info.resolved.family.parallel_spec(module=member, member=info.resolved)
                self.assertEqual(parallel.protocol, "engram.table.mesh")
                self.assertEqual(parallel.binding_kind, "mesh")
                parallel.validate(module=member, sharding=sharding, context=layout_context)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_ep_member_requires_binding_and_uses_mesh_coordinates(self) -> None:
        """Feature: Engram EP binding.

        Description: Invoke an EP member before and after binding a mock mesh.
        Expectation: It rejects the unbound call and forwards bound coordinates exactly once.
        """
        member = BuildEngram(config=_config(), context={"ep_size": 2})  # pylint: disable=not-callable
        hidden = torch.randn(1, 3, 2, 8)
        input_ids = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(RuntimeError, "bind the job mesh"):
            member(hidden, input_ids)

        binding, ep_group, cp_group = _parallel_binding()
        member.bind_parallel(binding)
        expected_coordinates = {
            "ep_size": 2, "ep_rank": 1, "ep_group": ep_group,
            "cp_size": 2, "cp_rank": 1, "cp_group": cp_group,
            "tp_size": 1, "tp_rank": 0,
        }
        self.assertEqual(
            member._parallel_coordinates, expected_coordinates,  # pylint: disable=protected-access
        )
        with patch.object(member, "parallel_forward", return_value=hidden) as parallel_forward:
            actual = member(hidden, input_ids)
        self.assertIs(actual, hidden)
        parallel_forward.assert_called_once_with(
            hidden, input_ids, None, None,
            **expected_coordinates,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_single_ep_lookup_keeps_the_embedding_path(self) -> None:
        """Feature: Engram EP1 lookup.

        Description: Look up hashed table rows with an EP1 mesh coordinate.
        Expectation: It returns direct embedding output without a collective.
        """
        member = EngramModule(config=_config())
        input_ids = torch.tensor([[[1, 2], [3, 4]]])
        actual = member._sparse_lookup(  # pylint: disable=protected-access
            input_ids, member.embed, ep_group=None, ep_rank=0, ep_size=1,
            padded_num_embeddings=member.padded_num_embeddings,
        )
        torch.testing.assert_close(actual, member.embed(input_ids))
