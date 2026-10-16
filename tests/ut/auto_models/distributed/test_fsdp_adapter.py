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
"""CPU tests for FSDP child-unit prefetch ordering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager, _WrapModuleInfo
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.deepseek_v41.adapter.policies.sharding import (
    get_fsdp_execution_order,
)


class _PrefetchUnit(nn.Module):
    """Record the actual target modules assigned by the builder."""

    def __init__(self) -> None:
        """Initialize empty target lists."""
        super().__init__()
        self.forward_targets = []
        self.backward_targets = []

    def set_modules_to_forward_prefetch(self, targets: list[nn.Module]) -> None:
        """Store forward prefetch targets."""
        self.forward_targets = targets

    def set_modules_to_backward_prefetch(self, targets: list[nn.Module]) -> None:
        """Store backward prefetch targets."""
        self.backward_targets = targets


class _TinyV41(nn.Module):
    """Give the ordering providers real module names without loading weights."""

    def __init__(self, layer_count: int = 3) -> None:
        """Construct selected child units in HF module registration order."""
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for _ in range(layer_count):
            layer = _PrefetchUnit()
            layer.engram = nn.Module()
            layer.engram.embed = _PrefetchUnit()
            layer.engram.wkv = _PrefetchUnit()
            layer.mlp = nn.Module()
            layer.mlp.experts = _PrefetchUnit()
            self.model.layers.append(layer)
        self.model.vision = nn.Module()
        self.model.vision.blocks = nn.ModuleList([_PrefetchUnit(), _PrefetchUnit()])
        self.model.aligner = _PrefetchUnit()


def _selected_units(model):
    """Choose the wrap units in module-registration order."""
    return [
        _WrapModuleInfo(name, module)
        for name, module in model.named_modules()
        if isinstance(module, _PrefetchUnit)
    ]


class TestFSDPPrefetchOrder(unittest.TestCase):
    """Check real V4.1 hook order and the generic reverse-order fallback."""

    def test_v41_backward_prefetch_targets_follow_nested_hooks(self):
        """Layer roots prefetch their children before prior layer roots."""
        model = _TinyV41(layer_count=40)
        model.config = SimpleNamespace(model_type="deepseek_v41")
        manager = FSDP2Manager(FSDP2Config(), mesh_context=None)
        forward = manager._order_wrap_modules(model, _selected_units(model))
        backward = manager._order_wrap_modules(model, forward, backward=True)
        manager._configure_prefetch(
            [unit.module for unit in forward], [unit.module for unit in backward]
        )
        backward_names = [unit.fqn for unit in backward]
        self.assertEqual(backward_names[-3:], ["model.aligner", "model.vision.blocks.1", "model.vision.blocks.0"])
        for layer_index in reversed(range(40)):
            root = f"model.layers.{layer_index}"
            root_position = backward_names.index(root)
            self.assertEqual(backward_names[root_position:root_position + 4], [
                root, f"{root}.mlp.experts", f"{root}.engram.wkv", f"{root}.engram.embed",
            ])
        units = dict(model.named_modules())
        self.assertEqual(units["model.layers.39"].backward_targets, [units["model.layers.39.mlp.experts"]])
        self.assertEqual(units["model.layers.39.engram.embed"].backward_targets, [units["model.layers.38"]])
        self.assertEqual(units["model.layers.1"].forward_targets, [units["model.layers.1.engram.embed"]])

    def test_default_backward_order_preserves_generic_behavior(self):
        """Models without a backward provider continue to reverse forward order."""
        model = _TinyV41(layer_count=1)
        manager = FSDP2Manager(FSDP2Config(forward_prefetch_depth=2, backward_prefetch_depth=2), None)
        provider = SimpleNamespace(fsdp_execution_order=get_fsdp_execution_order)
        with patch.object(FSDP2Manager, "_get_model_adapter_spec", return_value=provider):
            forward = manager._order_wrap_modules(model, _selected_units(model))
            backward = manager._order_wrap_modules(model, forward, backward=True)
        self.assertEqual([unit.fqn for unit in backward], [unit.fqn for unit in reversed(forward)])
        manager._configure_prefetch(
            [unit.module for unit in forward], [unit.module for unit in backward]
        )
        self.assertEqual(backward[0].module.backward_targets, [backward[1].module, backward[2].module])
        self.assertEqual(backward[-1].module.backward_targets, [])

    def test_backward_provider_must_name_every_selected_unit_once(self):
        """An incomplete hook order fails before assigning prefetch targets."""
        model = _TinyV41(layer_count=1)
        manager = FSDP2Manager(FSDP2Config(), None)
        provider = SimpleNamespace(fsdp_backward_execution_order=lambda _model, _fqns: ("model.layers.0",))
        with patch.object(FSDP2Manager, "_get_model_adapter_spec", return_value=provider):
            forward = manager._order_wrap_modules(model, _selected_units(model))
            with self.assertRaisesRegex(ValueError, "fsdp_backward_execution_order"):
                manager._order_wrap_modules(model, forward, backward=True)
