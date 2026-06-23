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
"""Unit tests for ``hyper_parallel.core.tensor_parallel.style.NoParallel``.

Tests cover:
- Default and custom constructor parameters
- Parameter replication (no sharding)
- Input preparation (local tensor -> DTensor, redistribute)
- Output preparation (redistribute, to_local)
- Integration with ``distribute_module`` via ``apply``
- Integration with ``parallelize_module`` dict-based plan
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.tensor_parallel.api import parallelize_module
from hyper_parallel.core.tensor_parallel.style import NoParallel, ParallelStyle
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestNoParallelInit(unittest.TestCase):
    """Tests for NoParallel constructor defaults and overrides."""

    def test_default_parameters(self):
        """
        Feature: NoParallel default constructor values
        Description: instantiate NoParallel with no arguments
        Expectation: input_layout is Replicate(), output_layout is Replicate(),
            desired_input_layout is Replicate(), use_local_output is True
        """
        style = NoParallel()
        self.assertEqual(style.input_layout, Replicate())
        self.assertEqual(style.output_layout, Replicate())
        self.assertEqual(style.desired_input_layout, Replicate())
        self.assertTrue(style.use_local_output)

    def test_custom_input_layout(self):
        """
        Feature: NoParallel custom input_layout
        Description: pass Shard(0) as input_layout
        Expectation: input_layout is Shard(0)
        """
        style = NoParallel(input_layout=Shard(0))
        self.assertEqual(style.input_layout, Shard(0))

    def test_custom_output_layout(self):
        """
        Feature: NoParallel custom output_layout
        Description: pass Shard(1) as output_layout
        Expectation: output_layout is Shard(1)
        """
        style = NoParallel(output_layout=Shard(1))
        self.assertEqual(style.output_layout, Shard(1))

    def test_custom_desired_input_layout(self):
        """
        Feature: NoParallel custom desired_input_layout
        Description: pass Shard(0) as desired_input_layout
        Expectation: desired_input_layout is Shard(0)
        """
        style = NoParallel(desired_input_layout=Shard(0))
        self.assertEqual(style.desired_input_layout, Shard(0))

    def test_custom_use_local_output(self):
        """
        Feature: NoParallel custom use_local_output
        Description: pass True as use_local_output
        Expectation: use_local_output is True
        """
        style = NoParallel(use_local_output=True)
        self.assertTrue(style.use_local_output)

    def test_is_subclass_of_parallel_style(self):
        """
        Feature: NoParallel inheritance
        Description: check NoParallel is a ParallelStyle
        Expectation: isinstance check passes
        """
        style = NoParallel()
        self.assertIsInstance(style, ParallelStyle)

    def test_src_data_rank_inherited(self):
        """
        Feature: NoParallel inherits src_data_rank default
        Description: fresh NoParallel instance
        Expectation: src_data_rank equals 0
        """
        style = NoParallel()
        self.assertEqual(style.src_data_rank, 0)

    def test_repr_contains_key_fields(self):
        """
        Feature: NoParallel.__repr__
        Description: instantiate with custom layouts
        Expectation: repr includes class name and layout fields
        """
        style = NoParallel(
            input_layout=Shard(0),
            output_layout=Shard(1),
            desired_input_layout=Replicate(),
            use_local_output=True,
        )
        r = repr(style)
        self.assertIn("NoParallel", r)
        self.assertIn("input_layout", r)
        self.assertIn("output_layout", r)


class TestNoParallelApply(unittest.TestCase):
    """Tests for NoParallel.apply with mocked distribute_module."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, platform_type=None, world_size=4):
        if platform_type is not None:
            mock_platform.platform_type = platform_type
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size=4):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_linear_calls_distribute_module(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: NoParallel.apply on Linear module
        Description: mock distribute_module
        Expectation: distribute_module called with partition_fn=None; returns module from mock
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.Module = nn.Module

        style = NoParallel()
        module = nn.Linear(8, 16)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.return_value = module
            result = style.apply(module, mesh)
            mock_dist.assert_called_once()
            call_args = mock_dist.call_args
            self.assertIs(call_args[0][0], module)
            self.assertIs(call_args[0][1], mesh)
            self.assertIs(result, module)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_invokes_distribute_module_callbacks(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: NoParallel.apply registers input/output callbacks
        Description: invoke callbacks captured from mocked distribute_module
        Expectation: partition_fn is None; input_fn/output_fn delegate to static helpers
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.Module = nn.Module

        style = NoParallel()
        module = nn.Linear(8, 16)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist, \
             patch.object(NoParallel, "_prepare_input_fn", return_value="inp") as mock_in, \
             patch.object(NoParallel, "_prepare_output_fn", return_value="out") as mock_out:
            mock_dist.return_value = module
            style.apply(module, mesh)
            partition_fn = mock_dist.call_args.kwargs["partition_fn"]
            input_fn = mock_dist.call_args.kwargs["input_fn"]
            output_fn = mock_dist.call_args.kwargs["output_fn"]
            self.assertIsNone(partition_fn)
            inp = input_fn(module, (torch.randn(2, 4),), mesh)
            out = output_fn(module, MagicMock(spec=DTensor), mesh)
            mock_in.assert_called_once()
            mock_out.assert_called_once()
            self.assertEqual(inp, "inp")
            self.assertEqual(out, "out")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_layernorm_calls_distribute_module(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: NoParallel.apply on LayerNorm module
        Description: mock distribute_module
        Expectation: distribute_module called once; returns module from mock
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.Module = nn.Module

        style = NoParallel()
        module = nn.LayerNorm(8)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.return_value = module
            result = style.apply(module, mesh)
            mock_dist.assert_called_once()
            self.assertIs(result, module)


class TestNoParallelIO(unittest.TestCase):
    """Tests for NoParallel input/output preparation functions."""

    def test_prepare_input_fn_wraps_plain_tensor(self):
        """
        Feature: _prepare_input_fn wraps local tensor with from_local
        Description: first input is torch.Tensor
        Expectation: returns tuple with DTensor.from_local result as first element
        """
        input_layout = Replicate()
        desired_input_layout = Replicate()
        mesh = MagicMock()
        local_tensor = torch.randn(2, 4)
        inputs = (local_tensor,)

        with patch.object(DTensor, "from_local") as mock_from_local:
            mock_dtensor = MagicMock(spec=DTensor)
            mock_dtensor.placements = (input_layout,)
            mock_from_local.return_value = mock_dtensor

            out = NoParallel._prepare_input_fn(
                input_layout, desired_input_layout, inputs, mesh
            )
            mock_from_local.assert_called_once_with(
                local_tensor, mesh, (input_layout,)
            )
            self.assertEqual(out, (mock_dtensor,))

    def test_prepare_input_fn_dtensor_no_redistribute_when_matching(self):
        """
        Feature: _prepare_input_fn leaves DTensor unchanged when layouts match
        """
        input_layout = Replicate()
        desired_input_layout = Replicate()
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Replicate(),)
        inputs = (mock_dt,)

        out = NoParallel._prepare_input_fn(
            input_layout, desired_input_layout, inputs, mesh
        )
        self.assertEqual(out, (mock_dt,))
        mock_dt.redistribute.assert_not_called()

    def test_prepare_input_fn_dtensor_redistributes_when_mismatch(self):
        """
        Feature: _prepare_input_fn redistributes DTensor when layouts differ
        """
        input_layout = Shard(0)
        desired_input_layout = Replicate()
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Shard(0),)
        redistributed = MagicMock(spec=DTensor)
        redistributed.placements = (Replicate(),)
        mock_dt.redistribute.return_value = redistributed
        inputs = (mock_dt,)

        out = NoParallel._prepare_input_fn(
            input_layout, desired_input_layout, inputs, mesh
        )
        mock_dt.redistribute.assert_called_once_with(mesh, (desired_input_layout,))
        self.assertEqual(out, (redistributed,))

    def test_prepare_input_fn_preserves_extra_positional_args(self):
        """
        Feature: _prepare_input_fn preserves remaining positional inputs
        Description: inputs has two elements; only first is a tensor
        Expectation: return tuple has (dtensor, second_input)
        """
        input_layout = Replicate()
        desired_input_layout = Replicate()
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Replicate(),)
        extra_arg = torch.randn(3, 5)
        inputs = (mock_dt, extra_arg)

        out = NoParallel._prepare_input_fn(
            input_layout, desired_input_layout, inputs, mesh
        )
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], mock_dt)
        self.assertIs(out[1], extra_arg)

    def test_prepare_output_fn_redistributes_when_mismatch(self):
        """
        Feature: _prepare_output_fn redistributes DTensor when placements differ
        """
        output_layout = Replicate()
        use_local_output = False
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Shard(0),)
        redistributed = MagicMock(spec=DTensor)
        redistributed.placements = (Replicate(),)
        mock_dt.redistribute.return_value = redistributed

        out = NoParallel._prepare_output_fn(
            output_layout, use_local_output, mock_dt, mesh
        )
        mock_dt.redistribute.assert_called_once_with(mesh, (output_layout,))
        self.assertIs(out, redistributed)

    def test_prepare_output_fn_keeps_dtensor_when_use_local_output_false(self):
        """
        Feature: _prepare_output_fn with use_local_output=False
        Description: output stays as DTensor
        """
        output_layout = Replicate()
        use_local_output = False
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Replicate(),)

        out = NoParallel._prepare_output_fn(
            output_layout, use_local_output, mock_dt, mesh
        )
        mock_dt.to_local.assert_not_called()
        self.assertIs(out, mock_dt)

    def test_prepare_output_fn_to_local_when_use_local_output_true(self):
        """
        Feature: _prepare_output_fn with use_local_output=True
        Description: output converted to local tensor via to_local
        """
        output_layout = Replicate()
        use_local_output = True
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Replicate(),)
        mock_dt.to_local.return_value = torch.randn(2, 4)

        out = NoParallel._prepare_output_fn(
            output_layout, use_local_output, mock_dt, mesh
        )
        mock_dt.to_local.assert_called_once()
        self.assertIsInstance(out, torch.Tensor)


class TestNoParallelIntegration(unittest.TestCase):
    """Integration tests for NoParallel with parallelize_module."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size=4):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size=4):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_parallelize_module_dict_plan_with_no_parallel(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: parallelize_module with dict plan containing NoParallel
        Description: apply NoParallel via parallelize_module dict
        Expectation: distribute_module called on target module
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.Module = nn.Module

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 16)
                self.norm = nn.LayerNorm(16)

        model = SimpleModel()

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.return_value = model.norm
            parallelize_module(
                model,
                mesh,
                {"norm": NoParallel()},
            )
            mock_dist.assert_called_once()


if __name__ == "__main__":
    unittest.main()
