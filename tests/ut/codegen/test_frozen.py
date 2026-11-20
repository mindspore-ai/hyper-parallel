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

from torch import nn

from hyper_parallel.codegen.emit.frozen import materialize_frozen_modules, observe_frozen_modules


class FrozenModel(nn.Module):
    """Two children of the same class, with independently configured gradients."""

    def __init__(self):
        super().__init__()
        self.media = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


class TestFrozenSource(unittest.TestCase):
    """Check both subtree freezing and a single frozen parameter."""

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
