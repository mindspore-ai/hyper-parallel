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
"""Unit tests for fully_shard list support on MindSpore platform (no NPU required)."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

# Skip entire module if mindspore is not installed (avoids import failure)
pytest.importorskip("mindspore")

# Force mindspore platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

# These test-only helpers intentionally accept the production hsdp_init signature.
# pylint: disable=unused-argument

from mindspore import nn as ms_nn

from hyper_parallel.core.fully_shard.api import (
    HSDPModule,
    _UnshardHandle,
    _check_hsdp_input_valid,
    _check_strict_keys,
    _get_device_from_mesh,
    _get_root_modules,
    _resolve_comm_fusion_zero_copy_default,
    _validate_module_for_fully_shard,
    fully_shard,
)
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform.platform import PlatformType


def _default_mp_policy():
    """Create default MixedPrecisionPolicy for tests."""
    return MixedPrecisionPolicy(
        param_dtype=None,
        reduce_dtype=None,
        output_dtype=None,
    )


def _make_parent_cell():
    """Create a parent Cell with child."""

    class ParentCell(ms_nn.Cell):
        """Parent cell containing child cell."""

        def __init__(self):
            super().__init__()
            self.child = ms_nn.Dense(4, 4)

        def construct(self, x):
            return self.child(x)

    return ParentCell


class TestGetRootModulesMindSpore(unittest.TestCase):
    """Unit tests for _get_root_modules with MindSpore Cells."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
        self.platform_type_patcher = patch(
            "hyper_parallel.core.fully_shard.api.platform.platform_type",
            PlatformType.MINDSPORE,
        )
        self.platform_type_patcher.start()
        self.addCleanup(self.platform_type_patcher.stop)

    def test_sibling_cells_both_roots(self):
        """Two sibling cells (neither contains the other) are both roots."""
        dense1 = ms_nn.Dense(4, 4)
        dense2 = ms_nn.Dense(4, 4)
        roots = _get_root_modules([dense1, dense2])
        self.assertEqual(len(roots), 2)
        self.assertIn(dense1, roots)
        self.assertIn(dense2, roots)

    def test_parent_child_only_parent_root(self):
        """When list contains parent and child, only parent is root."""
        ParentCell = _make_parent_cell()
        parent = ParentCell()
        child = parent.child
        roots = _get_root_modules([parent, child])
        self.assertEqual(len(roots), 1)
        self.assertIn(parent, roots)
        self.assertNotIn(child, roots)

    def test_single_cell_is_root(self):
        """Single cell in list is root."""
        cell = ms_nn.Dense(4, 4)
        roots = _get_root_modules([cell])
        self.assertEqual(len(roots), 1)
        self.assertIn(cell, roots)

    def test_three_siblings_all_roots(self):
        """Three sibling cells are all roots."""
        c1, c2, c3 = ms_nn.Dense(4, 4), ms_nn.Dense(4, 4), ms_nn.Dense(4, 4)
        roots = _get_root_modules([c1, c2, c3])
        self.assertEqual(len(roots), 3)
        self.assertIn(c1, roots)
        self.assertIn(c2, roots)
        self.assertIn(c3, roots)


class TestValidateModuleForFullyShardMindSpore(unittest.TestCase):
    """Unit tests for _validate_module_for_fully_shard with MindSpore."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

    def test_single_cell_valid(self):
        """Single Cell passes validation."""
        cell = ms_nn.Dense(4, 4)
        _validate_module_for_fully_shard(cell, PlatformType.MINDSPORE)

    def test_list_of_cells_valid(self):
        """List of Cells passes validation (MindSpore supports list)."""
        cells = [ms_nn.Dense(4, 4), ms_nn.Dense(4, 4)]
        _validate_module_for_fully_shard(cells, PlatformType.MINDSPORE)

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard([], PlatformType.MINDSPORE)
        self.assertIn("empty list", str(ctx.exception))

    def test_list_with_non_cell_raises(self):
        """List containing non-Cell raises ValueError."""
        cell = ms_nn.Dense(4, 4)
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard([cell, 1], PlatformType.MINDSPORE)
        self.assertIn("index 1", str(ctx.exception))

    def test_non_cell_raises(self):
        """Non-Cell input raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _validate_module_for_fully_shard("not a cell", PlatformType.MINDSPORE)
        self.assertIn("nn.cell", str(ctx.exception))


class TestCoreApiHelpersMindSpore(unittest.TestCase):
    """Unit tests for pure fully_shard API helpers on MindSpore platform."""

    def test_resolve_comm_fusion_zero_copy_default(self):
        """Backend defaults should only enable zero-copy on PyTorch comm-fusion."""
        self.assertTrue(
            _resolve_comm_fusion_zero_copy_default(PlatformType.MINDSPORE, True, True)
        )
        self.assertFalse(
            _resolve_comm_fusion_zero_copy_default(PlatformType.MINDSPORE, True, None)
        )
        self.assertFalse(
            _resolve_comm_fusion_zero_copy_default(PlatformType.PYTORCH, False, None)
        )
        self.assertTrue(
            _resolve_comm_fusion_zero_copy_default(PlatformType.PYTORCH, True, None)
        )

    def test_check_strict_keys_accepts_exact_match(self):
        """Strict state-dict checks pass when all keys match."""
        module = MagicMock()
        module.state_dict.return_value = {"weight": object(), "bias": object()}

        _check_strict_keys(module, {"weight": object(), "bias": object()})

    def test_check_strict_keys_reports_missing_and_unexpected(self):
        """Strict state-dict checks should report both missing and unexpected keys."""
        module = MagicMock()
        module.__class__.__name__ = "TinyCell"
        module.state_dict.return_value = {"weight": object(), "bias": object()}

        with self.assertRaisesRegex(RuntimeError, r"(?s)Missing key.*bias.*Unexpected key.*extra"):
            _check_strict_keys(module, {"weight": object(), "extra": object()})

    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_get_device_from_mesh_accepts_mindspore_npu(self, mock_platform):
        """MindSpore mesh validation returns the mesh device type directly."""
        mock_platform.platform_type = PlatformType.MINDSPORE

        self.assertEqual(_get_device_from_mesh(SimpleNamespace(device_type="npu")), "npu")

    def test_get_device_from_mesh_rejects_unsupported_device(self):
        """Only npu/cuda/cpu mesh device types are accepted by fully_shard."""
        with self.assertRaisesRegex(AssertionError, "support device"):
            _get_device_from_mesh(SimpleNamespace(device_type="xpu"))

    def test_check_hsdp_input_valid_rejects_invalid_options(self):
        """Input validation should reject invalid scalar options before setup."""
        cell = ms_nn.Dense(4, 4)
        valid_args = {
            "platform_type": PlatformType.MINDSPORE,
            "module": cell,
            "shard_size": 1,
            "threshold": 0,
            "optimizer_level": "level1",
            "enable_grad_accumulation": False,
            "grad_scale": 1.0,
            "reduce_dtype": None,
            "comm_async": False,
            "comm_fusion": False,
            "bucket_size": 0,
        }

        _check_hsdp_input_valid(**valid_args)
        for key, value in [
            ("shard_size", 0),
            ("threshold", -1),
            ("optimizer_level", "bad"),
            ("enable_grad_accumulation", 1),
            ("grad_scale", 1),
            ("reduce_dtype", "float32"),
            ("comm_async", 0),
            ("comm_fusion", 0),
            ("bucket_size", -2),
        ]:
            bad_args = dict(valid_args)
            bad_args[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _check_hsdp_input_valid(**bad_args)


class TestHSDPModuleInterfaceMindSpore(unittest.TestCase):
    """Unit tests for HSDPModule interface methods that delegate to scheduler/state."""

    class FakeHSDPModule(HSDPModule):
        """Small module stub with PyTorch-like traversal for shared interface tests."""

        def __init__(self, children=()):
            super().__init__()
            self.hsdp_scheduler = MagicMock()
            self._children = list(children)

        def modules(self):
            return [self, *self._children]

    def test_unshard_async_handle_waits_once(self):
        """Async unshard should return an idempotent wait handle."""
        state = MagicMock()
        module = self.FakeHSDPModule()
        module.hsdp_scheduler.hsdp_state = state

        handle = module.unshard(async_op=True)
        handle.wait()
        handle.wait()

        state.unshard.assert_called_once_with(True)
        state.wait_for_unshard.assert_called_once_with()

    def test_unshard_handle_without_state_is_noop(self):
        """A no-op unshard handle should tolerate repeated waits."""
        handle = _UnshardHandle()

        handle.wait()
        handle.wait()

    @patch("hyper_parallel.core.fully_shard.api.platform.get_cells_and_names")
    def test_reshard_and_reduce_op_delegate_to_state(self, mock_cells):
        """State operations should be routed through the current scheduler."""
        state = MagicMock()
        module = self.FakeHSDPModule()
        module.hsdp_scheduler.hsdp_state = state
        mock_cells.return_value = [("", module)]

        module.reshard()
        module.set_reduce_op_type("sum")

        state.shard.assert_called_once_with()
        state.set_reduce_op_type.assert_called_once_with("sum")

    @patch("hyper_parallel.core.fully_shard.api.platform.platform_type", PlatformType.MINDSPORE)
    @patch("hyper_parallel.core.fully_shard.api.platform.get_cells_and_names")
    def test_set_requires_gradient_sync_and_zero_grad_walk_cells(self, mock_cells):
        """MindSpore traversal should update every nested HSDP module."""
        child = self.FakeHSDPModule()
        module = self.FakeHSDPModule(children=[child])
        mock_cells.return_value = [("", module), ("child", child)]

        module.set_requires_gradient_sync(False)
        module.zero_grad()

        module.hsdp_scheduler.set_requires_grad_sync.assert_called_once_with(False)
        child.hsdp_scheduler.set_requires_grad_sync.assert_called_once_with(False)
        module.hsdp_scheduler.zero_grad.assert_called_once_with()
        child.hsdp_scheduler.zero_grad.assert_called_once_with()

    def test_prefetch_setters_validate_module_list(self):
        """Prefetch setters should accept only HSDPModule sequences."""
        first = self.FakeHSDPModule()
        second = self.FakeHSDPModule()

        first.set_modules_to_forward_prefetch([second])
        first.set_modules_to_backward_prefetch((second,))

        first.hsdp_scheduler.set_forward_prefetch_cells.assert_called_once_with([second])
        first.hsdp_scheduler.set_backward_prefetch_cells.assert_called_once_with((second,))
        with self.assertRaisesRegex(ValueError, "HSDPModule list"):
            first.set_modules_to_forward_prefetch([object()])

    def test_recursive_scheduler_flags_use_module_traversal(self):
        """Recursive flags should be forwarded to every nested HSDP scheduler."""
        child = self.FakeHSDPModule()
        module = self.FakeHSDPModule(children=[child])

        with patch(
            "hyper_parallel.core.fully_shard.api.platform.get_cells_and_names",
            return_value=[("", module), ("child", child)],
        ):
            module.set_requires_all_reduce(False)
            module.set_reshard_after_forward(False)
            module.set_reshard_after_backward(True)

        for scheduler in (module.hsdp_scheduler, child.hsdp_scheduler):
            scheduler.set_requires_all_reduce.assert_called_once_with(False)
            scheduler.set_reshard_after_forward.assert_called_once_with(False)
            scheduler.set_reshard_after_backward.assert_called_once_with(True)

    def test_recursive_scheduler_flags_validate_inputs(self):
        """Recursive setters should reject non-bool values and recurse=False."""
        module = self.FakeHSDPModule()

        with self.assertRaises(ValueError):
            module.set_requires_all_reduce(1)
        with self.assertRaises(NotImplementedError):
            module.set_requires_all_reduce(True, recurse=False)
        with self.assertRaises(ValueError):
            module.set_reshard_after_forward(1)
        with self.assertRaises(ValueError):
            module.set_reshard_after_backward(1)


class TestFullyShardListAPIMindSpore(unittest.TestCase):
    """Unit tests for fully_shard list support on MindSpore (mocked to avoid NPU/dist)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
        self.params_patcher = patch(
            "hyper_parallel.core.fully_shard.api._get_modules_parameters",
            return_value=[],
        )
        self.params_patcher.start()
        self.addCleanup(self.params_patcher.stop)

    def _create_mock_mesh(self):
        """Create mock DeviceMesh to avoid distributed init."""
        mesh = MagicMock()
        mesh.ndim = 1
        return mesh

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_single_cell_returns_cell(self, mock_platform, mock_get_device):
        """fully_shard with single cell returns the same cell (in-place)."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        cell = ms_nn.Dense(4, 4)
        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            lambda self, *a, **k: None,
        ):
            result = fully_shard(
                cell,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        self.assertIs(result, cell)
        self.assertIsInstance(result, ms_nn.Cell)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    @patch("hyper_parallel.platform.mindspore.autograd_compat.enable_mindspore_backward_compat")
    def test_fully_shard_list_returns_list_and_enables_backward_compat(
        self, mock_enable_backward_compat, mock_platform, mock_get_device
    ):
        """fully_shard with list returns the same list and enables backward compat."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        dense1 = ms_nn.Dense(4, 4)
        dense2 = ms_nn.Dense(4, 4)
        cells_list = [dense1, dense2]

        def _fake_hsdp_init(self, *args, **kwargs):
            self.hsdp_scheduler = object()

        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            _fake_hsdp_init,
        ):
            result = fully_shard(
                cells_list,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )

        self.assertIs(result, cells_list)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIs(dense2.hsdp_scheduler, dense1.hsdp_scheduler)
        mock_enable_backward_compat.assert_called_once_with()

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_root_filtering(self, mock_platform, mock_get_device):
        """fully_shard with parent+child list filters to root only."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        ParentCell = _make_parent_cell()
        parent = ParentCell()
        child = parent.child
        cells_list = [parent, child]
        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            lambda self, *a, **k: None,
        ):
            result = fully_shard(
                cells_list,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        self.assertIs(result, cells_list)
        self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_passes_root_modules_to_hsdp_init(
        self, mock_platform, mock_get_device
    ):
        """fully_shard([cell1, cell2]) initializes the scheduler with the root modules tuple."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        cell1 = ms_nn.Dense(4, 4)
        cell2 = ms_nn.Dense(4, 4)
        cells_list = [cell1, cell2]
        captured = {}

        def _fake_hsdp_init(self, platform_type, module, *args, **kwargs):
            captured["platform_type"] = platform_type
            captured["module"] = module
            self.hsdp_scheduler = object()

        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            _fake_hsdp_init,
        ):
            result = fully_shard(
                cells_list,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )

        self.assertIs(result, cells_list)
        self.assertEqual(captured["platform_type"], PlatformType.MINDSPORE)
        self.assertEqual(captured["module"], (cell1, cell2))

    @patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh")
    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_list_second_root_shares_scheduler_handle(
        self, mock_platform, mock_get_device
    ):
        """fully_shard([cell1, cell2]) backfills the same scheduler handle to the second root."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mock_get_device.return_value = "npu"
        mesh = self._create_mock_mesh()
        cell1 = ms_nn.Dense(4, 4)
        cell2 = ms_nn.Dense(4, 4)
        cells_list = [cell1, cell2]

        scheduler_sentinel = object()

        def _fake_hsdp_init(self, *args, **kwargs):
            self.hsdp_scheduler = scheduler_sentinel

        with patch(
            "hyper_parallel.core.fully_shard.api.HSDPModule.hsdp_init",
            _fake_hsdp_init,
        ):
            result = fully_shard(
                cells_list,
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )

        self.assertIs(result, cells_list)
        self.assertIs(cell1.hsdp_scheduler, scheduler_sentinel)
        self.assertIs(cell2.hsdp_scheduler, scheduler_sentinel)

    @patch("hyper_parallel.core.fully_shard.api.platform")
    def test_fully_shard_empty_list_raises(self, mock_platform):
        """fully_shard with empty list raises ValueError."""
        mock_platform.platform_type = PlatformType.MINDSPORE
        mesh = self._create_mock_mesh()
        with self.assertRaises(ValueError) as ctx:
            fully_shard(
                [],
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=_default_mp_policy(),
            )
        self.assertIn("empty list", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
