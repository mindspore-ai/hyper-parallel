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
"""Imported primitives must contain their source-visible parallel computation."""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hyper_parallel.codegen.emit.context import EmissionContext
from hyper_parallel.codegen.emit.parallel import lower_forward_boundaries
from hyper_parallel.codegen.emit.primitives import materialize_primitive_boundaries, observe_primitive_boundaries
from hyper_parallel.codegen.plan.observation import observe_emission_plan
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.distributed.mesh import LogicalDeviceMesh, MeshContext
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


class PrimitiveModel(nn.Module):
    """An ordinary source embedding construction."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 2)


class TestPrimitiveSource(unittest.TestCase):
    """Validate native masking without invoking distributed setup."""

    def test_vocab_mask_is_in_the_class_without_forward_installation(self):
        model = PrimitiveModel()
        spec = ModuleShardingSpec(
            params={'weight': {'tp': Shard(0)}},
            in_src={'input': {'tp': Replicate()}}, in_dst={'input': {'tp': Replicate()}},
            out_src={'output': {'tp': Partial()}}, out_dst={'output': {'tp': Shard(1)}},
        )
        plan = SimpleNamespace(modules={'embed': spec}, mesh_dim_names=('tp',))
        capture = SimpleNamespace(plan=plan, model=model, manifest=SimpleNamespace(replacements=()),
                                  mesh=MeshContext(device_mesh=LogicalDeviceMesh(('tp',), (2,)), tp_size=2))
        records = observe_primitive_boundaries(capture)
        observed = observe_emission_plan(plan, model)
        meta = EmissionContext(source={}, parallel_dims={'tp_size': 2}, status='derived',
                               param_plan=observed.param_plan, boundary_classes=observed.boundary_classes,
                               mesh_dim_names=['tp'], primitive_boundaries=records)
        source = (
            'from torch import nn\n'
            'class PrimitiveModel(nn.Module):\n'
            '    def __init__(self):\n'
            '        super().__init__()\n'
            '        self.embed = nn.Embedding(8, 2)\n'
        )
        rendered = materialize_primitive_boundaries(source, meta)
        rendered = lower_forward_boundaries(rendered, meta, boundary_classes=meta.boundary_classes)
        namespace = {}
        exec(rendered, namespace)
        actual = namespace['PrimitiveModel']().embed
        actual.weight = nn.Parameter(torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]))
        actual._hp_tp_mesh = SimpleNamespace(get_local_rank=lambda: 1)
        output = actual._forward_impl(torch.tensor([0, 4, 7]))
        torch.testing.assert_close(output, torch.tensor([[0., 0.], [1., 2.], [7., 8.]]))
        self.assertNotIn('forward', actual.__dict__)
        self.assertIn('reduce_scatter', rendered)

    def test_no_plan_does_not_create_primitive_shells(self):
        self.assertEqual(observe_primitive_boundaries(SimpleNamespace(plan=None)), [])
