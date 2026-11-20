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
"""Parameter freezing must survive direct construction of exported source."""

import unittest

import torch
from torch import nn

from hyper_parallel.codegen.emit.frozen import materialize_frozen_modules, observe_frozen_modules


class FrozenModel(nn.Module):
    """Two children of the same class, with independently configured gradients."""

    def __init__(self):
        super().__init__()
        self.media = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


class FrozenRouter(nn.Module):
    """A later native conversion introduces a non-trainable router parameter."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2), requires_grad=False)


class TestFrozenSource(unittest.TestCase):
    """Check both subtree freezing and a single frozen parameter."""

    def test_imported_owner_freezes_after_its_actual_construction(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in model.layers:
            layer.gate = FrozenRouter()
        source = (f"from {__name__} import FrozenRouter\nfrom torch import nn\n"
                  "class Layer(nn.Module):\n    def __init__(self):\n"
                  "        super().__init__()\n        self.gate = nn.Identity()\n"
                  "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
                  "        self.layers = nn.ModuleList([Layer(), Layer()])\n"
                  "        for layer in self.layers:\n            layer.gate = FrozenRouter()\n")
        rendered = materialize_frozen_modules(source, observe_frozen_modules(model))
        namespace = {}
        exec(rendered, namespace)
        actual = namespace["Model"]()
        self.assertEqual([parameter.requires_grad for parameter in actual.parameters()],
                         [parameter.requires_grad for parameter in model.parameters()])
        self.assertIn("layer.gate.bias.requires_grad_(False)", rendered)
        self.assertNotIn("self.gate.bias.requires_grad_", rendered)

    def test_frozen_subtree_does_not_freeze_another_instance(self):
        model = FrozenModel()
        model.media.requires_grad_(False)
        model.head.weight.requires_grad_(False)
        source = (
            "from torch import nn\n"
            "class FrozenModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.media = nn.Linear(2, 2)\n"
            "        self.head = nn.Linear(2, 2)\n"
        )
        records = observe_frozen_modules(model)
        rendered = materialize_frozen_modules(source, records)
        namespace = {}
        exec(rendered, namespace)
        actual = namespace["FrozenModel"]()
        self.assertEqual({key: value.requires_grad for key, value in model.named_parameters()},
                         {key: value.requires_grad for key, value in actual.named_parameters()})
        self.assertIn("self.media.requires_grad_(False)", rendered)
        self.assertIn("self.head.weight.requires_grad_(False)", rendered)

    def test_trainable_model_requires_no_source_edits(self):
        self.assertEqual(observe_frozen_modules(FrozenModel()), [])
