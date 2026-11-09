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
"""Borrowed mesh inputs for construction and concrete family selection."""

import unittest
from unittest.mock import patch

from torch import nn

from hyper_parallel.components.modules import family as registry
from hyper_parallel.components.modules.module_family import FamilyRequest, build_family, get_family_build_info
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache, init_device_mesh
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.models.module_build import build_module_context, construct_family_model
from tests.common.mark_utils import arg_mark
from tests.common.module_family_fixtures import LinearConfig, LinearFamily, ParallelLinear, register_linear_family


class _BorrowedMesh(MeshContext):
    """Reject attempts to copy borrowed runtime owners into saved requests."""

    def __deepcopy__(self, memo):
        raise AssertionError("mesh resources must be borrowed")


class TestMeshFamilyConstruction(unittest.TestCase):
    """Validate constructor topology without process-group initialization."""

    def setUp(self) -> None:
        """Keep test registrations local to this case."""
        handle = patch.dict(registry._FAMILIES, clear=True)
        handle.start()
        self.addCleanup(handle.stop)
        register_linear_family()
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)

    @staticmethod
    def logical_mesh() -> MeshContext:
        """Use an offline dense topology and its expert-parallel degree."""
        return _BorrowedMesh(device_mesh=LogicalDeviceMesh(("dp", "cp", "tp"), (2, 2, 1)),
                             fsdp_moe_mesh=LogicalDeviceMesh(("edp", "ep"), (2, 2)),
                             dp_size=2, cp_size=2, ep_size=2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mesh_selects_a_concrete_member_without_copying_resources(self) -> None:
        """Feature: Actual job axes select the family member.
        Description: Pass a non-copyable logical mesh to shared family construction.
        Expectation: EP selects ParallelLinear and mesh stays outside static arguments.
        """
        mesh = self.logical_mesh()
        result = build_family(LinearFamily, config=LinearConfig(), mesh=mesh)
        self.assertIs(type(result), ParallelLinear)
        info = get_family_build_info(result)
        self.assertEqual(info.resolved.axis_sizes["ep"], 2)
        self.assertNotIn("mesh", info.request.kwargs)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_managed_factory_can_accept_mesh_without_internal_context(self) -> None:
        """Feature: Ordinary model constructors can receive only config and mesh.
        Description: Inject an existing job mesh through common model construction.
        Expectation: The nested family selects EP without requiring a context argument.
        """
        def factory(config: LinearConfig, *, mesh: MeshContext) -> nn.Module:
            """Compose a model using only configuration and the borrowed job mesh."""
            return nn.Sequential(build_family(LinearFamily, config=config, mesh=mesh))

        mesh = self.logical_mesh()
        context = build_module_context(DistributedSetup(mesh_context=mesh))
        result = construct_family_model(factory, LinearConfig(), context=context)
        self.assertIs(type(result[0]), ParallelLinear)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mesh_conflicts_and_static_mesh_arguments_are_rejected(self) -> None:
        """Feature: Runtime owners cannot enter serialized selectors.
        Description: Mix incompatible logical axes or save mesh in a FamilyRequest.
        Expectation: Both invalid inputs fail before a member is allocated.
        """
        mesh = self.logical_mesh()
        with self.assertRaisesRegex(TypeError, "unknown"):
            FamilyRequest(LinearFamily, {"mesh": mesh})
        with self.assertRaisesRegex(ValueError, "disagree"):
            build_family(LinearFamily, config=LinearConfig(), mesh=mesh, context={"ep_size": 1})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_active_parallel_axes_require_their_standard_names(self) -> None:
        """Feature: Families require named CP/EP domains before allocating members.
        Description: Supply missing or renamed axes in logical and backend-free runtime meshes.
        Expectation: Construction reports the required axis instead of selecting or binding a member.
        """
        for runtime in (False, True):
            for axis in ("cp", "ep"):
                for actual_name in ("dp", "sequence" if axis == "cp" else "experts"):
                    with self.subTest(runtime=runtime, axis=axis, actual_name=actual_name):
                        if runtime:
                            device_mesh = init_device_mesh(
                                "cpu", (2,), mesh_dim_names=(actual_name,), rank_list=(0, 1), init_backend=False,
                            )
                        else:
                            device_mesh = LogicalDeviceMesh((actual_name,), (2,))
                        mesh = MeshContext(device_mesh=device_mesh, **{f"{axis}_size": 2})
                        with patch("hyper_parallel.components.modules.module_family._build_family_member",
                                   side_effect=AssertionError("member allocated before mesh validation")):
                            with self.assertRaisesRegex(ValueError, f"requires an axis named '{axis}'"):
                                build_family(LinearFamily, config=LinearConfig(), mesh=mesh)
