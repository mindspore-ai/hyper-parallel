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
"""Unit tests for ``hyper_parallel.core.tensor_parallel.api`` (``test_api.py`` ↔ ``api.py``).

Primary coverage: ``parallelize_module`` and related helpers.

Cases align with PyTorch ``test/distributed/tensor/parallel/test_parallelize_api.py``:
path matching (fnmatch), single-style root apply, ``src_data_rank``, empty plan,
2-D mesh rejection, and implicit mesh via internal context (PyTorch ``with mesh``).
"""
import os
import unittest
import warnings
from unittest.mock import patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import (
    _DEVICE_MESH_MAP,
    _mesh_resources,
    init_device_mesh,
)
from hyper_parallel.core.tensor_parallel.api import (
    _tensor_parallel_mesh_context,
    parallelize_module,
)
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class RecordingParallelStyle(ParallelStyle):
    """Records ``apply`` calls for assertions (no real sharding)."""

    def __init__(self, name: str = "recording"):
        super().__init__()
        self.name = name
        self.apply_log: list[tuple[int, int]] = []  # (id(module), id(mesh))

    def apply(self, module, device_mesh):
        self.apply_log.append((id(module), id(device_mesh)))
        return module


class TestParallelizeModule(unittest.TestCase):
    """Tests for ``parallelize_module`` dispatch and validation."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        # avoid leaking mesh stack if a test failed inside a context manager
        _mesh_resources.mesh_stack.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 4):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size: int = 4):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    def _make_2d_mesh(self, mock_platform):
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_raises_without_device_mesh(self, mock_platform):
        """
        Feature: parallelize_module validation with missing device_mesh
        Description: call parallelize_module with device_mesh=None and no active mesh context
        Expectation: RuntimeError from _mesh_resources.get_current_mesh (no active mesh)
        """
        self._setup_mock_platform(mock_platform)
        m = nn.Linear(3, 3)
        with self.assertRaises(RuntimeError) as ctx:
            parallelize_module(m, None, RecordingParallelStyle())
        self.assertIn("device mesh", str(ctx.exception).lower())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_rejects_nd_mesh(self, mock_platform):
        """
        Feature: parallelize_module validation for mesh dimension
        Description: call parallelize_module with a 2-D DeviceMesh and a ParallelStyle
        Expectation: throw ValueError whose message contains 1D
        """
        mesh = self._make_2d_mesh(mock_platform)
        m = nn.Linear(3, 3)
        with self.assertRaises(ValueError) as ctx:
            parallelize_module(m, mesh, RecordingParallelStyle())
        self.assertIn("1D", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_none_plan_warns_and_noop(self, mock_platform):
        """
        Feature: parallelize_module with None parallelize_plan
        Description: call parallelize_module with parallelize_plan=None and valid 1-D mesh
        Expectation: return same module; UserWarning mentions parallelize_plan; params unchanged
        """
        mesh = self._make_1d_mesh(mock_platform)
        m = nn.Linear(3, 3)
        before = list(m.parameters())[0].data.clone()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = parallelize_module(m, mesh, None)
        self.assertIs(out, m)
        self.assertTrue(any("parallelize_plan" in str(x.message) for x in w))
        self.assertTrue(torch.nn.functional.mse_loss(before, list(m.parameters())[0].data).item() == 0.0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_style_applies_to_root(self, mock_platform):
        """
        Feature: parallelize_module with single ParallelStyle
        Description: pass one RecordingParallelStyle (not dict) for root Linear module
        Expectation: apply_log has one entry with id(module) and id(mesh) matching root and mesh
        """
        mesh = self._make_1d_mesh(mock_platform)
        m = nn.Linear(4, 4)
        style = RecordingParallelStyle()
        parallelize_module(m, mesh, style)
        self.assertEqual(len(style.apply_log), 1)
        self.assertEqual(style.apply_log[0][0], id(m))
        self.assertEqual(style.apply_log[0][1], id(mesh))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_src_data_rank_set_on_style(self, mock_platform):
        """
        Feature: parallelize_module propagates src_data_rank to style
        Description: two calls with src_data_rank=None then src_data_rank=2 on Identity module
        Expectation: first style src_data_rank is None; second style src_data_rank equals 2
        """
        mesh = self._make_1d_mesh(mock_platform)
        m = nn.Identity()
        style = RecordingParallelStyle()
        parallelize_module(m, mesh, style, src_data_rank=None)
        self.assertIsNone(style.src_data_rank)
        style2 = RecordingParallelStyle()
        parallelize_module(m, mesh, style2, src_data_rank=2)
        self.assertEqual(style2.src_data_rank, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dict_nested_path(self, mock_platform):
        """
        Feature: parallelize_module with dict plan and nested path
        Description: Root with encoder Sequential; plan key encoder.1 maps to one style
        Expectation: apply runs once on encoder submodule index 1 only
        """

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Identity(), nn.Linear(2, 2))

        mesh = self._make_1d_mesh(mock_platform)
        root = Root()
        target = root.encoder[1]
        style = RecordingParallelStyle()
        parallelize_module(root, mesh, {"encoder.1": style})
        self.assertEqual(len(style.apply_log), 1)
        self.assertEqual(style.apply_log[0][0], id(target))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_fnmatch_star_like_pytorch(self, mock_platform):
        """
        Feature: parallelize_module with fnmatch wildcard star
        Description: MLP with net1, net2, other; dict plan net* with one RecordingParallelStyle
        Expectation: apply_log targets net1 and net2 only; two apply calls total
        """

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = nn.Identity()
                self.net2 = nn.Identity()
                self.other = nn.Identity()

        mesh = self._make_1d_mesh(mock_platform)
        model = MLP()
        style = RecordingParallelStyle()
        parallelize_module(model, mesh, {"net*": style})
        applied_ids = {mid for mid, _ in style.apply_log}
        self.assertEqual(applied_ids, {id(model.net1), id(model.net2)})
        self.assertEqual(len(style.apply_log), 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_fnmatch_question_like_pytorch(self, mock_platform):
        """
        Feature: parallelize_module with fnmatch wildcard question mark
        Description: module with net1 and net2; dict plan net? with one style
        Expectation: apply_log contains net1 and net2 submodule ids only
        """

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = nn.Identity()
                self.net2 = nn.Identity()

        mesh = self._make_1d_mesh(mock_platform)
        model = M()
        style = RecordingParallelStyle()
        parallelize_module(model, mesh, {"net?": style})
        applied_ids = {mid for mid, _ in style.apply_log}
        self.assertEqual(applied_ids, {id(model.net1), id(model.net2)})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_fnmatch_bracket_like_pytorch(self, mock_platform):
        """
        Feature: parallelize_module with fnmatch bracket range
        Description: module with net1, net2, net3; dict plan net[1-2] with one style
        Expectation: apply_log contains net1 and net2 only; net3 not matched
        """

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = nn.Identity()
                self.net2 = nn.Identity()
                self.net3 = nn.Identity()

        mesh = self._make_1d_mesh(mock_platform)
        model = M()
        style = RecordingParallelStyle()
        parallelize_module(model, mesh, {"net[1-2]": style})
        applied_ids = {mid for mid, _ in style.apply_log}
        self.assertEqual(applied_ids, {id(model.net1), id(model.net2)})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_multi_wildcard_path_like_pytorch(self, mock_platform):
        """
        Feature: parallelize_module with multi-level wildcard path
        Description: Stack with ModuleList of blocks; plans layers.*.net1 and layers.*.net2
        Expectation: col style hits all net1; row style hits all net2 across layers
        """

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = nn.Identity()
                self.net2 = nn.Identity()

        class Stack(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        mesh = self._make_1d_mesh(mock_platform)
        model = Stack()
        s1 = RecordingParallelStyle("col")
        s2 = RecordingParallelStyle("row")
        parallelize_module(
            model,
            mesh,
            {
                "layers.*.net1": s1,
                "layers.*.net2": s2,
            },
        )
        self.assertEqual({mid for mid, _ in s1.apply_log}, {id(b.net1) for b in model.layers})
        self.assertEqual({mid for mid, _ in s2.apply_log}, {id(b.net2) for b in model.layers})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_submesh_slice_accepted(self, mock_platform):
        """
        Feature: parallelize_module accepts 1-D submesh sliced from N-D mesh
        Description: build 2-D mesh then use mesh["tp"] 1-D slice with Linear and style
        Expectation: parallelize_module succeeds; apply_log length is 1
        """
        mesh_2d = self._make_2d_mesh(mock_platform)
        mesh_1d = mesh_2d["tp"]
        self.assertEqual(mesh_1d.ndim, 1)
        m = nn.Linear(2, 2)
        style = RecordingParallelStyle()
        parallelize_module(m, mesh_1d, style)
        self.assertEqual(len(style.apply_log), 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_bad_parallelize_plan_type(self, mock_platform):
        """
        Feature: parallelize_module validation for parallelize_plan type
        Description: pass string literal as parallelize_plan with valid mesh and module
        Expectation: throw TypeError
        """
        mesh = self._make_1d_mesh(mock_platform)
        mod = nn.Linear(1, 1)
        with self.assertRaises(TypeError):
            parallelize_module(mod, mesh, "not-a-style")  # type: ignore[arg-type]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_implicit_mesh_internal_context(self, mock_platform):
        """
        Feature: parallelize_module with implicit mesh via _tensor_parallel_mesh_context
        Description: enter _tensor_parallel_mesh_context(mesh) then parallelize_module(..., None, style)
        Expectation: one apply; apply_log mesh id matches context mesh
        """
        mesh = self._make_1d_mesh(mock_platform)
        m = nn.Linear(2, 2)
        style = RecordingParallelStyle()
        with _tensor_parallel_mesh_context(mesh):
            parallelize_module(m, None, style)
        self.assertEqual(len(style.apply_log), 1)
        self.assertEqual(style.apply_log[0][1], id(mesh))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_implicit_mesh_with_statement_like_pytorch(self, mock_platform):
        """
        Feature: parallelize_module with implicit mesh via ``with mesh:``
        Description: use DeviceMesh as context manager (PyTorch parity)
        Expectation: parallelize_module(..., None, style) resolves mesh from stack
        """
        mesh = self._make_1d_mesh(mock_platform)
        m = nn.Linear(2, 2)
        style = RecordingParallelStyle()
        with mesh:
            parallelize_module(m, None, style)
        self.assertEqual(len(style.apply_log), 1)
        self.assertEqual(style.apply_log[0][1], id(mesh))


if __name__ == "__main__":
    unittest.main()
