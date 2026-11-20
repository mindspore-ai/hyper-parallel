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
"""Native method results must execute without runtime method installation."""

from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn

from hyper_parallel.codegen.emit.methods import (
    materialize_method_hosts, materialize_observed_methods, verify_observed_methods,
)
from hyper_parallel.codegen.emit.validation import verify_generated_source_structure
from hyper_parallel.codegen.plan.methods import observe_local_methods
from hyper_parallel.distributed.mesh import LogicalDeviceMesh, MeshContext
from hyper_parallel.distributed.recipe_spec import local_compute


class MethodExperts(nn.Module):
    """A child whose native factory installs a different local forward."""

    def __init__(self):
        super().__init__()
        self.scale = 1

    def forward(self, value):
        """Return the source computation."""
        return value


class MethodBlock(nn.Module):
    """A native region boundary."""

    def __init__(self):
        super().__init__()
        self.experts = MethodExperts()

    def forward(self, hidden_states):
        """Match the declared region signature."""
        return self.experts(hidden_states)


def _expert_forward(experts, value):
    return value * experts.scale


def _with_group(value, group):
    if group is None:
        raise ValueError("missing runtime group")
    return value


def bound_forward(self, hidden_states):
    """Represent a model-owned method bound during ordinary model construction."""
    return self.experts(hidden_states)


@local_compute
def method_factory(*, module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Use the same stateful factory contract as native EP recipes."""
    del mesh, tp_mesh, cp_mesh
    group = ep_mesh.get_group("ep")
    module.experts.scale = 2
    module.experts.forward = MethodType(_expert_forward, module.experts)

    def combine(target, value):
        result = target.experts(value)
        return _with_group(result, group)

    def compute(module, hidden_states):
        return combine(module, hidden_states)

    return compute


@local_compute
def ranked_factory(*, module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """Capture rank-dependent resources without freezing the export process's rank."""
    del module, mesh, cp_mesh
    rank = tp_mesh.get_local_rank()
    groups = (ep_mesh.get_group("ep"), tp_mesh.get_group())

    def combine(target, value):
        return _with_group(value, groups) + rank + tp_mesh.get_local_rank()

    def compute(module, hidden_states):
        return combine(module, hidden_states)

    return compute


class TestMethodMaterialization(unittest.TestCase):
    """Validate the emitted parent method and the factory's child-side patch."""

    def test_imported_local_region_has_a_source_forward_before_lowering(self):
        records = {"block": {"type": f"{__name__}:MethodBlock", "local_compute": True}}
        source = materialize_method_hosts("", records)
        namespace = {}
        exec(source, namespace)
        value = torch.ones(2)
        generated = namespace["MethodBlock"]()
        self.assertIs(generated.forward(value), value)
        self.assertIsNot(type(generated), MethodBlock)
        self.assertEqual(source, materialize_method_hosts(source, records))

    def test_native_factory_result_is_a_class_method(self):
        model = nn.Module()
        model.block = MethodBlock()
        plan = SimpleNamespace(
            modules={"block": SimpleNamespace(local_compute_fn=method_factory, region_dispatch=False)},
            mesh_dim_names=("tp", "cp"),
        )
        topology = MeshContext(
            device_mesh=LogicalDeviceMesh(("tp", "cp"), (1, 1)),
            fsdp_moe_mesh=LogicalDeviceMesh(("edp", "ep"), (1, 2)), ep_size=2,
        )
        records = observe_local_methods(plan, model, topology)
        self.assertEqual(model.block.experts.scale, 1)
        self.assertNotIn("forward", model.block.experts.__dict__)
        source = (
            f"from {__name__} import MethodExperts\n"
            "class MethodBlock:\n"
            "    ep_enable = False\n"
            "    def __init__(self):\n"
            "        self.experts = MethodExperts()\n"
            "    def forward(self, value):\n"
            "        if self.ep_enable:\n"
            "            return self.method_factory(value)\n"
            "        return self.experts(value)\n"
        )
        rendered = materialize_observed_methods(source, records)
        verify_observed_methods(rendered, records)
        with self.assertRaisesRegex(ValueError, "was not materialized"):
            verify_observed_methods(source, records)
        namespace = {}
        exec(rendered, namespace)
        generated = namespace["MethodBlock"]()
        generated._hp_ep_group = object()
        value = torch.tensor(3.0, requires_grad=True)
        output = generated.forward(value)
        self.assertEqual(output.item(), 6)
        output.backward()
        self.assertEqual(value.grad.item(), 2)
        self.assertNotIn("forward", generated.experts.__dict__)
        self.assertNotIn("method_factory", generated.__dict__)
        self.assertTrue(generated.ep_enable)
        self.assertEqual(generated.experts.scale, 2)

    def test_no_injection_does_not_require_any_mesh(self):
        self.assertEqual(observe_local_methods(SimpleNamespace(modules={}), nn.Module(), None), {})

    def test_selected_inner_wrapper_is_active_after_direct_construction(self):
        model = nn.Module()
        model.block = MethodBlock()
        plan = SimpleNamespace(modules={"block": SimpleNamespace(
            local_compute_fn=None, inner_wrapper=object(), inner_target="self",
        )}, mesh_dim_names=("tp", "cp"))
        topology = MeshContext(device_mesh=LogicalDeviceMesh(("tp", "cp"), (1, 2)), cp_size=2)
        records = observe_local_methods(plan, model, topology)
        source = ("class MethodBlock:\n    cp_enable = False\n"
                  "    def forward(self, value):\n        return value * (2 if self.cp_enable else 1)\n")
        namespace = {}
        exec(materialize_observed_methods(source, records), namespace)
        generated = namespace["MethodBlock"]()
        self.assertTrue(generated.cp_enable)
        self.assertEqual(generated.forward(3), 6)
        self.assertFalse(hasattr(model.block, "cp_enable"))
        model.other = MethodBlock()
        with self.assertRaisesRegex(ValueError, "partially selected class"):
            observe_local_methods(plan, model, topology)

    def test_native_instance_dispatch_gets_the_materialized_class_flag(self):
        model = nn.Module()
        model.block = MethodBlock()
        model.block.forward = MethodType(bound_forward, model.block)
        plan = SimpleNamespace(
            modules={"block": SimpleNamespace(local_compute_fn=method_factory, region_dispatch=False)},
            mesh_dim_names=("tp", "cp"),
        )
        topology = MeshContext(device_mesh=LogicalDeviceMesh(("tp", "cp"), (1, 1)),
                               fsdp_moe_mesh=LogicalDeviceMesh(("edp", "ep"), (1, 2)), ep_size=2)
        records = observe_local_methods(plan, model, topology)
        source = (f"from {__name__} import MethodExperts\n"
                  "class MethodBlock:\n    def __init__(self):\n        self.experts = MethodExperts()\n"
                  "def bound_forward(self, hidden_states):\n"
                  "    if self.ep_enable:\n        return self.method_factory(hidden_states)\n"
                  "    return self.experts(hidden_states)\n")
        namespace = {}
        rendered = materialize_observed_methods(source, records)
        verify_generated_source_structure({"modeling_future.py": rendered})
        exec(rendered, namespace)
        generated = namespace["MethodBlock"]()
        generated._hp_ep_group = object()
        self.assertTrue(generated.ep_enable)
        self.assertEqual(generated.forward(3), 6)

    def test_mesh_rank_and_group_tuple_remain_runtime_resources(self):
        model = nn.Module()
        model.block = MethodBlock()
        plan = SimpleNamespace(
            modules={"block": SimpleNamespace(local_compute_fn=ranked_factory, region_dispatch=False)},
            mesh_dim_names=("tp", "cp"),
        )
        topology = MeshContext(
            device_mesh=LogicalDeviceMesh(("tp", "cp"), (2, 1)),
            fsdp_moe_mesh=LogicalDeviceMesh(("edp", "ep"), (1, 2)), ep_size=2,
        )
        records = observe_local_methods(plan, model, topology)
        source = ("class MethodBlock:\n    ep_enable = False\n"
                  "    def forward(self, value):\n        return self.ranked_factory(value)\n")
        rendered = materialize_observed_methods(source, records)
        namespace = {}
        exec(rendered, namespace)
        generated = namespace["MethodBlock"]()
        generated._hp_ep_group = object()
        generated._hp_tp_group = object()
        for rank in (0, 1):
            generated._hp_tp_mesh = SimpleNamespace(get_local_rank=lambda: rank)
            self.assertEqual(generated.forward(3), 3 + 2 * rank)
