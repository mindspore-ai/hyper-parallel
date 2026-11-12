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
"""Direct family construction from borrowed runtime and logical meshes."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from examples.torch.module_family import CausalAttention, DecoderConfig, FamilyDecoder, GroupedExperts
from examples.torch.module_family.attention.cp_attention import CPCausalAttention
from examples.torch.module_family.grouped_experts.ep_grouped_experts import EPGroupedExperts
from examples.torch.module_family.model import MODEL_SPEC
from hyper_parallel.components.modules import family as registry
from hyper_parallel.components.modules.module_family import FamilyRequest, get_family_build_info
from hyper_parallel.core.dtensor.device_mesh import clear_device_mesh_cache, init_device_mesh
from hyper_parallel.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.distributed._builder.parameter_sharding import _shard_module_params
from hyper_parallel.distributed.mesh import DistributedSetup, LogicalDeviceMesh, MeshContext
from hyper_parallel.models.module_build import build_module_context, construct_family_model, prepare_model_components
from tests.common.mark_utils import arg_mark


class _BorrowedMesh(MeshContext):
    """Fail if a component tries to copy process-group-owning job state."""

    def __deepcopy__(self, memo):
        raise AssertionError("mesh resources must be borrowed")


class TestMeshFamilyConstruction(unittest.TestCase):
    """Exercise public family entries without creating real distributed groups."""

    def setUp(self) -> None:
        """Keep registries, mesh caches and random state isolated."""
        handle = patch.dict(registry._FAMILIES)
        handle.start()
        self.addCleanup(handle.stop)
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)

    @staticmethod
    def logical_mesh() -> MeshContext:
        """Represent the same four ranks as dense DP2/CP2 and expert EP2."""
        return _BorrowedMesh(
            device_mesh=LogicalDeviceMesh(("dp", "cp", "tp"), (2, 2, 1)),
            fsdp_moe_mesh=LogicalDeviceMesh(("edp", "ep"), (2, 2)),
            dp_size=2, cp_size=2, ep_size=2,
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_logical_mesh_selects_members_without_runtime_resources(self) -> None:
        """Feature: Constructor mesh selects real classes before runtime.
        Description: Build a complete meta model with a non-copyable logical job mesh.
        Expectation: CP/EP classes and global shapes are retained without groups or bindings.
        """
        mesh = self.logical_mesh()
        with patch("torch.distributed.new_group", side_effect=AssertionError("new group")), torch.device("meta"):
            model = FamilyDecoder(DecoderConfig(), mesh=mesh)
        self.assertIs(type(model.attention), CPCausalAttention)
        self.assertIs(type(model.experts), EPGroupedExperts)
        self.assertTrue(all(parameter.is_meta for parameter in model.parameters()))
        self.assertEqual(tuple(model.experts.gate_up.shape), (4, 8, 32))
        for component in (model.attention, model.experts):
            layout = get_family_build_info(component).layout
            self.assertIs(layout.mesh, mesh)
            self.assertFalse(layout.applied)
            self.assertIsNone(getattr(component, "_hp_parallel_binding", None))
            self.assertNotIn("mesh", get_family_build_info(component).request.kwargs)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_preparation_reuses_constructor_selection_and_layout(self) -> None:
        """Feature: A framework continues the constructor lifecycle.
        Description: Prepare and globally plan the meta model after its family providers ran.
        Expectation: No selector or family provider runs again and aliases retain their identity.
        """
        declaration = registry.get_module_family(GroupedExperts)
        provider = Mock(wraps=declaration.sharding)
        registry._FAMILIES[id(GroupedExperts)] = replace(declaration, sharding=provider)
        mesh = self.logical_mesh()
        with torch.device("meta"):
            model = FamilyDecoder(DecoderConfig(), mesh=mesh)
        self.assertEqual(provider.call_count, 1)
        model.alias = model.experts
        original = model.experts
        context = build_module_context(DistributedSetup(mesh_context=mesh), model_spec=MODEL_SPEC)
        with patch("hyper_parallel.models.module_build.resolve_family", side_effect=AssertionError("resolved twice")):
            result, _ = prepare_model_components(model, context)
        ShardingPlanner().plan(result, mesh.device_mesh, cp_size=2, ep_size=2)
        self.assertEqual(provider.call_count, 1)
        self.assertIs(result.experts, original)
        self.assertIs(result.alias, original)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_size_one_runtime_mesh_builds_a_usable_local_module(self) -> None:
        """Feature: Direct mesh construction is usable without a model facade.
        Description: Supply an uninitialized size-one mesh, where no collective is required.
        Expectation: The module computes finite gradients without constructing a process group.
        """
        device_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",), rank_list=(0,), init_backend=False)
        mesh = MeshContext(device_mesh=device_mesh)
        with patch("torch.distributed.new_group", side_effect=AssertionError("new group")):
            experts = GroupedExperts(config=DecoderConfig(), mesh=mesh)
        self.assertTrue(get_family_build_info(experts).layout.applied)
        tokens = torch.randn(3, 8, requires_grad=True)
        ids = torch.tensor([[0, 3], [1, 2], [0, 2]])
        probabilities = torch.full((3, 2), 0.5, requires_grad=True)
        output = experts(tokens, ids, probabilities)
        self.assertEqual(tuple(output.shape), (3, 8))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(probabilities.grad).all())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_managed_constructor_accepts_a_mesh_only_model_signature(self) -> None:
        """Feature: An ordinary model constructor need not accept HP's internal context.
        Description: Let common construction pass its existing mesh to a mesh-only factory.
        Expectation: CP/EP members select correctly and keep the original borrowed mesh.
        """
        def factory(config: DecoderConfig, *, mesh: MeshContext) -> FamilyDecoder:
            """Compose a model using only configuration and the borrowed job mesh."""
            return FamilyDecoder(config, mesh=mesh)

        mesh = self.logical_mesh()
        context = build_module_context(DistributedSetup(mesh_context=mesh), model_spec=MODEL_SPEC)
        with torch.device("meta"):
            model = construct_family_model(factory, DecoderConfig(), context=context)
        self.assertIs(type(model.attention), CPCausalAttention)
        self.assertIs(type(model.experts), EPGroupedExperts)
        self.assertIs(get_family_build_info(model.experts).layout.mesh, mesh)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_runtime_inputs_are_not_request_selectors(self) -> None:
        """Feature: Process-group owners remain outside static configuration.
        Description: Attempt to store mesh in a saved family request and mix conflicting axes.
        Expectation: Both errors are rejected before any runtime allocation.
        """
        mesh = self.logical_mesh()
        with self.assertRaisesRegex(TypeError, "unknown"):
            FamilyRequest(GroupedExperts, {"mesh": mesh})
        with self.assertRaisesRegex(ValueError, "disagree"):
            GroupedExperts(config=DecoderConfig(), mesh=mesh, context={"ep_size": 1})
        with self.assertRaisesRegex(ValueError, "different jobs"):
            GroupedExperts(config=DecoderConfig(), mesh=mesh, context={"mesh": self.logical_mesh()})
        invalid = SimpleNamespace(device_mesh=mesh.device_mesh, cp_size=True, ep_size=2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            CausalAttention(config=DecoderConfig(), mesh=invalid)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_active_parallel_axes_require_their_standard_names(self) -> None:
        """Feature: Families require named CP/EP domains before allocating members.
        Description: Supply missing or renamed axes in logical and backend-free runtime meshes.
        Expectation: Construction reports the required axis instead of selecting or binding a member.
        """
        for runtime in (False, True):
            for axis, factory in (("cp", CausalAttention), ("ep", GroupedExperts)):
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
                                factory(config=DecoderConfig(), mesh=mesh)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_expert_mesh_names_and_sizes_are_validated(self) -> None:
        """Feature: The expert view is the authoritative EP domain.
        Description: Supply a renamed expert axis or a degree that differs from its named view.
        Expectation: Invalid expert views fail before member construction.
        """
        renamed = self.logical_mesh()
        renamed.fsdp_moe_mesh = LogicalDeviceMesh(("edp", "experts"), (2, 2))
        mismatched = self.logical_mesh()
        mismatched.ep_size = 4
        for mesh, error in ((renamed, "requires an axis named 'ep'"), (mismatched, "disagrees with the mesh size")):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    GroupedExperts(config=DecoderConfig(), mesh=mesh)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_named_ep_axis_can_be_provided_by_the_main_mesh(self) -> None:
        """Feature: EP may use the main mesh when no expert view is supplied.
        Description: Construct a meta component from a logical mesh with an explicit ep axis.
        Expectation: The family selects EP while retaining the borrowed mesh.
        """
        mesh = MeshContext(device_mesh=LogicalDeviceMesh(("ep",), (2,)), ep_size=2)
        with torch.device("meta"):
            experts = GroupedExperts(config=DecoderConfig(), mesh=mesh)
        self.assertIs(type(experts), EPGroupedExperts)
        self.assertIs(get_family_build_info(experts).layout.mesh, mesh)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_context_mesh_is_validated_for_construction_and_preparation(self) -> None:
        """Feature: Managed entry points obey the same mesh-name contract.
        Description: Pass a renamed expert view through job context to a new or cached family.
        Expectation: Both entry points reject it before constructing another member.
        """
        invalid = self.logical_mesh()
        invalid.fsdp_moe_mesh = LogicalDeviceMesh(("edp", "experts"), (2, 2))
        context = build_module_context(DistributedSetup(mesh_context=invalid), model_spec=MODEL_SPEC)
        with torch.device("meta"):
            model = FamilyDecoder(DecoderConfig(), mesh=self.logical_mesh())
            deferred = FamilyDecoder(DecoderConfig())
        with patch("hyper_parallel.components.modules.module_family._build_family_member",
                   side_effect=AssertionError("member allocated before mesh validation")):
            with self.assertRaisesRegex(ValueError, "requires an axis named 'ep'"):
                GroupedExperts(config=DecoderConfig(), context=context)
            with self.assertRaisesRegex(ValueError, "requires an axis named 'ep'"):
                prepare_model_components(model, context)
            with self.assertRaisesRegex(ValueError, "requires an axis named 'ep'"):
                prepare_model_components(deferred, context)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_offline_declarations_continue_on_a_runtime_mesh(self) -> None:
        """Feature: Offline declarations can continue in an actual compatible job.
        Description: Prepare a meta model using a runtime mesh after offline family construction.
        Expectation: Concrete members remain unchanged and layout application runs once.
        """
        logical = MeshContext(device_mesh=LogicalDeviceMesh(("dp",), (1,)))
        with torch.device("meta"):
            model = FamilyDecoder(DecoderConfig(), mesh=logical)
        original = model.experts
        device_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",), rank_list=(0,), init_backend=False)
        runtime = MeshContext(device_mesh=device_mesh)
        context = build_module_context(DistributedSetup(mesh_context=runtime), model_spec=MODEL_SPEC)
        model, _ = prepare_model_components(model, context)
        plan = ShardingPlanner().plan(model, runtime.device_mesh)
        model, _ = apply_sharding_plan(model, plan, runtime)
        self.assertIs(model.experts, original)
        self.assertTrue(get_family_build_info(original).layout.applied)
        with patch("hyper_parallel.distributed._builder.parameter_sharding._shard_module_params",
                   wraps=_shard_module_params) as shard:
            apply_sharding_plan(model, plan, runtime.device_mesh)
        self.assertTrue(all(call.args[0] is not original and call.args[0] is not model.attention
                            for call in shard.call_args_list))
        self.assertNotIn("forward", vars(original))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_constructed_layout_rejects_a_conflicting_framework_override(self) -> None:
        """Feature: Already selected computation ownership cannot be silently overwritten.
        Description: Override an EP component with replicated expert weights after construction.
        Expectation: Global planning rejects the incompatible layout instead of reshaping it twice.
        """
        mesh = self.logical_mesh()
        with torch.device("meta"):
            experts = GroupedExperts(config=DecoderConfig(), mesh=mesh)
        layout = get_family_build_info(experts).layout.sharding
        changed = replace(layout, params={"gate_up": {}, "down": {}})
        with self.assertRaisesRegex(ValueError, "constructed family layout"):
            ShardingPlanner({"": changed}).plan(experts, mesh.device_mesh, cp_size=2, ep_size=2)
