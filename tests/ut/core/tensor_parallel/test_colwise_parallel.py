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
"""Unit tests for ``hyper_parallel.core.tensor_parallel.style.ColwiseParallel``.

Tests cover:
- Default and custom constructor parameters
- Linear / Embedding partition logic (weight placement)
- Unsupported module type rejection
- Input preparation (local tensor -> DTensor, redistribute)
- Output preparation (redistribute, to_local)
- Integration with ``distribute_module`` via ``apply``
"""
import os
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.tensor_parallel.style import ColwiseParallel
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestColwiseParallelInit(unittest.TestCase):
    """Tests for ColwiseParallel constructor defaults and overrides."""

    def test_default_parameters(self):
        """
        Feature: ColwiseParallel default constructor values
        Description: instantiate ColwiseParallel with no arguments
        Expectation: input_layouts is (Replicate(),), output_layouts is (Shard(-1),),
            desired_input_layouts is (Replicate(),), use_local_output is True
        """
        style = ColwiseParallel()
        self.assertEqual(style.input_layouts, (Replicate(),))
        self.assertEqual(style.output_layouts, (Shard(-1),))
        self.assertEqual(style.desired_input_layouts, (Replicate(),))
        self.assertTrue(style.use_local_output)

    def test_custom_input_layouts(self):
        """
        Feature: ColwiseParallel custom input_layouts
        Description: pass Shard(0) as input_layouts
        Expectation: input_layouts is (Shard(0),); desired_input_layouts remains (Replicate(),)
        """
        style = ColwiseParallel(input_layouts=Shard(0))
        self.assertEqual(style.input_layouts, (Shard(0),))
        self.assertEqual(style.desired_input_layouts, (Replicate(),))

    def test_custom_output_layouts(self):
        """
        Feature: ColwiseParallel custom output_layouts
        Description: pass Replicate() as output_layouts
        Expectation: output_layouts is (Replicate(),)
        """
        style = ColwiseParallel(output_layouts=Replicate())
        self.assertEqual(style.output_layouts, (Replicate(),))

    def test_use_local_output_false(self):
        """
        Feature: ColwiseParallel with use_local_output=False
        Description: set use_local_output to False
        Expectation: use_local_output is False
        """
        style = ColwiseParallel(use_local_output=False)
        self.assertFalse(style.use_local_output)

    def test_is_subclass_of_parallel_style(self):
        """
        Feature: ColwiseParallel inheritance
        Description: check ColwiseParallel is a ParallelStyle
        Expectation: isinstance check passes
        """
        from hyper_parallel.core.tensor_parallel.style import ParallelStyle
        style = ColwiseParallel()
        self.assertIsInstance(style, ParallelStyle)

    def test_src_data_rank_inherited(self):
        """
        Feature: ColwiseParallel inherits src_data_rank default
        Description: fresh ColwiseParallel instance
        Expectation: src_data_rank equals 0
        """
        style = ColwiseParallel()
        self.assertEqual(style.src_data_rank, 0)


class TestColwiseParallelApply(unittest.TestCase):
    """Tests for ColwiseParallel.apply with mocked distribute_module."""

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
    def test_apply_linear_calls_distribute_module(self, mock_style_platform, mock_mesh_platform):
        """
        Feature: ColwiseParallel.apply on Linear module
        Description: apply ColwiseParallel to nn.Linear with mocked distribute_module
        Expectation: distribute_module is called with correct partition_fn, input_fn, output_fn
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = True
        mock_style_platform.is_embedding_module.return_value = False
        mock_style_platform.Module = nn.Module

        style = ColwiseParallel()
        module = nn.Linear(8, 8)

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
    def test_apply_embedding_calls_distribute_module(self, mock_style_platform, mock_mesh_platform):
        """
        Feature: ColwiseParallel.apply on Embedding module
        Description: apply ColwiseParallel to nn.Embedding with mocked distribute_module
        Expectation: distribute_module is called once
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = False
        mock_style_platform.is_embedding_module.return_value = True
        mock_style_platform.Module = nn.Module

        style = ColwiseParallel()
        module = nn.Embedding(100, 64)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.return_value = module
            result = style.apply(module, mesh)
            mock_dist.assert_called_once()
            self.assertIs(result, module)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_unsupported_module_raises(self, mock_style_platform, mock_mesh_platform):
        """
        Feature: ColwiseParallel.apply rejects unsupported module types
        Description: apply ColwiseParallel to nn.LayerNorm
        Expectation: raises NotImplementedError mentioning supported types
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = False
        mock_style_platform.is_embedding_module.return_value = False
        mock_style_platform.Module = nn.Module

        style = ColwiseParallel()
        module = nn.LayerNorm(8)

        with self.assertRaises(NotImplementedError) as ctx:
            style.apply(module, mesh)
        self.assertIn("Linear", str(ctx.exception))
        self.assertIn("Embedding", str(ctx.exception))


class TestColwiseParallelPartition(unittest.TestCase):
    """Tests for ColwiseParallel partition functions."""

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
    def test_partition_linear_fn_uses_shard0(self, mock_platform):
        """
        Feature: ColwiseParallel._partition_linear_fn sharding strategy
        Description: call _partition_linear_fn on nn.Linear(8, 16)
        Expectation: distribute_tensor is called with [Shard(0)] for each parameter
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = ColwiseParallel()
        module = nn.Linear(8, 16)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_tensor") as mock_dt, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_param_source") as mock_src, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_new_parameter") as mock_new, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_set_param"):
            mock_src.side_effect = lambda p: p.data
            mock_dt_result = MagicMock()
            mock_dt.return_value = mock_dt_result
            mock_new.return_value = nn.Parameter(torch.empty(1))

            style._partition_linear_fn("linear", module, mesh)

            for call in mock_dt.call_args_list:
                self.assertEqual(call[0][2], [Shard(0)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partition_embedding_fn_uses_shard1(self, mock_platform):
        """
        Feature: ColwiseParallel._partition_embedding_fn sharding strategy
        Description: call _partition_embedding_fn on nn.Embedding(100, 64)
        Expectation: distribute_tensor is called with [Shard(1)]
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = ColwiseParallel()
        module = nn.Embedding(100, 64)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_tensor") as mock_dt, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_param_source") as mock_src, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_new_parameter") as mock_new, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_set_param"):
            mock_src.side_effect = lambda p: p.data
            mock_dt.return_value = MagicMock()
            mock_new.return_value = nn.Parameter(torch.empty(1))

            style._partition_embedding_fn("embedding", module, mesh)

            for call in mock_dt.call_args_list:
                self.assertEqual(call[0][2], [Shard(1)])


class TestColwiseParallelIO(unittest.TestCase):
    """Tests for ColwiseParallel input/output preparation functions."""

    def test_prepare_input_fn_wraps_plain_tensor(self):
        """
        Feature: _prepare_input_fn wraps local tensor
        Description: pass plain torch.Tensor as input[0]
        Expectation: DTensor.from_local is called with correct device_mesh and input_layouts
        """
        from hyper_parallel.core.dtensor.dtensor import DTensor

        input_layouts = (Replicate(),)
        desired_layouts = (Replicate(),)
        mesh = MagicMock()
        mod = MagicMock()
        local_tensor = torch.randn(4, 8)
        inputs = (local_tensor,)

        with patch.object(DTensor, "from_local") as mock_from_local:
            mock_dtensor = MagicMock(spec=DTensor)
            mock_from_local.return_value = mock_dtensor

            ColwiseParallel._prepare_input_fn(
                input_layouts, desired_layouts, mod, inputs, mesh
            )
            mock_from_local.assert_called_once_with(local_tensor, mesh, input_layouts)

    def test_prepare_input_fn_skips_wrap_if_dtensor(self):
        """
        Feature: _prepare_input_fn passes through DTensor
        Description: pass DTensor as input[0] with matching layouts
        Expectation: DTensor.from_local is NOT called
        """
        from hyper_parallel.core.dtensor.dtensor import DTensor

        input_layouts = (Replicate(),)
        desired_layouts = (Replicate(),)
        mesh = MagicMock()
        mod = MagicMock()
        mock_dtensor = MagicMock(spec=DTensor)
        inputs = (mock_dtensor,)

        with patch.object(DTensor, "from_local") as mock_from_local:
            ColwiseParallel._prepare_input_fn(
                input_layouts, desired_layouts, mod, inputs, mesh
            )
            mock_from_local.assert_not_called()

    def test_prepare_input_fn_redistributes_if_layouts_differ(self):
        """
        Feature: _prepare_input_fn redistributes on layout mismatch
        Description: input_layouts=(Shard(0),), desired=(Replicate(),)
        Expectation: redistribute is called on the DTensor
        """
        from hyper_parallel.core.dtensor.dtensor import DTensor

        input_layouts = (Shard(0),)
        desired_layouts = (Replicate(),)
        mesh = MagicMock()
        mod = MagicMock()
        mock_dtensor = MagicMock(spec=DTensor)
        mock_from_local = MagicMock(return_value=mock_dtensor)

        with patch.object(DTensor, "from_local", mock_from_local):
            ColwiseParallel._prepare_input_fn(
                input_layouts, desired_layouts, mod, (torch.randn(4, 8),), mesh
            )
            mock_dtensor.redistribute.assert_called_once_with(mesh, desired_layouts)

    def test_prepare_output_fn_to_local(self):
        """
        Feature: _prepare_output_fn converts to local tensor
        Description: use_local_output=True, matching placements
        Expectation: to_local() is called on outputs
        """
        output_layouts = (Shard(-1),)
        mock_outputs = MagicMock()
        mock_outputs.placements = output_layouts
        mock_outputs.to_local.return_value = torch.randn(4, 4)
        mesh = MagicMock()

        result = ColwiseParallel._prepare_output_fn(
            output_layouts, True, MagicMock(), mock_outputs, mesh
        )
        mock_outputs.to_local.assert_called_once()
        self.assertIsInstance(result, torch.Tensor)

    def test_prepare_output_fn_keeps_dtensor(self):
        """
        Feature: _prepare_output_fn keeps DTensor when use_local_output=False
        Description: use_local_output=False, matching placements
        Expectation: to_local() is NOT called
        """
        output_layouts = (Shard(-1),)
        mock_outputs = MagicMock()
        mock_outputs.placements = output_layouts
        mesh = MagicMock()

        result = ColwiseParallel._prepare_output_fn(
            output_layouts, False, MagicMock(), mock_outputs, mesh
        )
        mock_outputs.to_local.assert_not_called()
        self.assertIs(result, mock_outputs)

    def test_prepare_output_fn_redistributes_if_needed(self):
        """
        Feature: _prepare_output_fn redistributes when placements differ
        Description: output has Shard(-1) but desired is Replicate()
        Expectation: redistribute is called with desired output_layouts
        """
        output_layouts = (Replicate(),)
        mock_outputs = MagicMock()
        mock_outputs.placements = (Shard(-1),)
        redistributed = MagicMock()
        redistributed.to_local.return_value = torch.randn(4, 8)
        redistributed.placements = output_layouts
        mock_outputs.redistribute.return_value = redistributed
        mesh = MagicMock()

        ColwiseParallel._prepare_output_fn(
            output_layouts, True, MagicMock(), mock_outputs, mesh
        )
        mock_outputs.redistribute.assert_called_once_with(mesh, output_layouts)


if __name__ == "__main__":
    unittest.main()
