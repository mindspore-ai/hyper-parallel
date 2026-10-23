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
"""Unit tests for FSDP child-unit prefetch configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager, _WrapModuleInfo
from hyper_parallel.models import get_model_adapter
from hyper_parallel.models.qwen3_vl.adapter.fsdp import (
    get_fsdp_execution_order,
    get_fsdp_prefetch_groups,
    get_fsdp_wrap_modules,
)


class _PrefetchModule(nn.Module):
    """Minimal FSDP-like module that records configured prefetch targets."""

    def __init__(self, requires_grad: bool) -> None:
        """Create one parameter with the requested trainability."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=requires_grad)
        self.forward_prefetch: list[nn.Module] = []
        self.backward_prefetch: list[nn.Module] = []

    def set_modules_to_forward_prefetch(self, modules: list[nn.Module]) -> None:
        """Record forward-prefetch targets."""
        self.forward_prefetch = modules

    def set_modules_to_backward_prefetch(self, modules: list[nn.Module]) -> None:
        """Record backward-prefetch targets."""
        self.backward_prefetch = modules


class TestFSDP2Prefetch(unittest.TestCase):
    """Verify prefetch selection without distributed initialization."""

    def test_configure_prefetch_keeps_frozen_units_forward_only(self) -> None:
        """Frozen units stay in the forward chain but not the backward chain."""
        manager = object.__new__(FSDP2Manager)
        manager.config = SimpleNamespace(
            forward_prefetch_depth=1,
            backward_prefetch_depth=1,
        )
        frozen = _PrefetchModule(requires_grad=False)
        first = _PrefetchModule(requires_grad=True)
        second = _PrefetchModule(requires_grad=True)

        manager._configure_prefetch([first, frozen, second])  # pylint: disable=protected-access

        self.assertEqual(first.forward_prefetch, [frozen])
        self.assertEqual(frozen.forward_prefetch, [second])
        self.assertEqual(second.forward_prefetch, [])
        self.assertEqual(first.backward_prefetch, [])
        self.assertEqual(frozen.backward_prefetch, [])
        self.assertEqual(second.backward_prefetch, [first])

    def test_independent_prefetch_groups(self) -> None:
        """No forward or backward prefetch crosses an optional branch boundary."""
        manager = object.__new__(FSDP2Manager)
        manager.config = SimpleNamespace(forward_prefetch_depth=1, backward_prefetch_depth=1)
        units = [_PrefetchModule(True) for _ in range(4)]
        entries = [_WrapModuleInfo(str(index), unit) for index, unit in enumerate(units)]
        spec = SimpleNamespace(fsdp_prefetch_groups=lambda model, fqns: (fqns[:2], fqns[2:]))
        with patch.object(manager, "_get_model_adapter_spec", return_value=spec):
            manager._configure_model_prefetch(nn.Module(), entries)
        self.assertEqual(units[1].forward_prefetch, [])
        self.assertEqual(units[2].backward_prefetch, [])
        self.assertEqual(units[0].forward_prefetch, [units[1]])
        self.assertEqual(units[3].backward_prefetch, [units[2]])

    def test_prefetch_groups_validate_partition(self) -> None:
        """Reject duplicate, missing, unknown and malformed unit declarations."""
        manager = object.__new__(FSDP2Manager)
        entries = [_WrapModuleInfo("unit", _PrefetchModule(True))]
        for groups in (((),), (("unit", "unit"),), (), (("unknown",),), "unit", ((1,),)):
            with self.subTest(groups=groups):
                spec = SimpleNamespace(fsdp_prefetch_groups=lambda model, fqns: groups)
                with patch.object(manager, "_get_model_adapter_spec", return_value=spec):
                    with self.assertRaises(ValueError):
                        manager._configure_model_prefetch(nn.Module(), entries)


class TestQwenVisualPrefetch(unittest.TestCase):
    """Keep Qwen visual FSDP boundaries aligned with native execution."""

    @staticmethod
    def _model(indices: tuple[int, ...] = (0, 2)) -> nn.Module:
        """Construct a small visual hierarchy with interleaved DeepStack units."""
        root = nn.Module()
        root.model = nn.Module()
        root.model.visual = nn.Module()
        visual = root.model.visual
        visual.patch_embed = nn.Linear(1, 1)
        visual.pos_embed = nn.Embedding(1, 1)
        visual.blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        visual.merger = nn.Linear(1, 1)
        visual.deepstack_visual_indexes = list(indices)
        visual.deepstack_merger_list = nn.ModuleList([nn.Linear(1, 1) for _ in indices])
        return root

    def test_visual_order_and_branch_partition(self) -> None:
        """Main merger follows all interleaved DeepStack calls, never precedes them."""
        model = self._model()
        prefix = "model.visual."
        expected = tuple(prefix + name for name in (
            "patch_embed", "blocks.0", "deepstack_merger_list.0",
            "blocks.1", "blocks.2", "deepstack_merger_list.1", "merger",
        ))
        self.assertEqual(get_fsdp_wrap_modules(model), expected)
        self.assertNotIn(prefix + "pos_embed", expected)
        selected = ("model.language_model.layers.0", *reversed(expected))
        ordered = get_fsdp_execution_order(model, selected)
        self.assertEqual(ordered, expected + selected[:1])
        groups = get_fsdp_prefetch_groups(model, ordered)
        self.assertEqual(groups[0], tuple(prefix + f"blocks.{index}" for index in range(3)))
        self.assertEqual(groups[-1], selected[:1])
        self.assertIn((prefix + "deepstack_merger_list.0",), groups)
        self.assertIn((prefix + "merger",), groups)
        self.assertCountEqual([fqn for group in groups for fqn in group], ordered)

    def test_without_deepstack_and_invalid_indices(self) -> None:
        """Support the Qwen3.5 tower and reject unexecuted DeepStack declarations."""
        model = self._model(())
        self.assertEqual(len(get_fsdp_wrap_modules(model)), 5)
        for indices in ((0, 0), (3,), (-1,)):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    get_fsdp_wrap_modules(self._model(indices))

    def test_model_families_register_visual_policies(self) -> None:
        """Dense, MoE and Qwen3.5 VLMs opt in; text-only Qwen3.5 does not."""
        for family in ("qwen3_vl", "qwen3_vl_moe", "qwen3_5"):
            with self.subTest(family=family):
                spec = get_model_adapter(family)
                self.assertIs(spec.fsdp_wrap_modules, get_fsdp_wrap_modules)
                self.assertIs(spec.fsdp_prefetch_groups, get_fsdp_prefetch_groups)
        self.assertIsNone(get_model_adapter("qwen3_5_text").fsdp_prefetch_groups)


if __name__ == "__main__":
    unittest.main()
