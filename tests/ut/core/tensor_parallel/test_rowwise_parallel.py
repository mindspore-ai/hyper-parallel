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
"""Unit tests for ``hyper_parallel.core.tensor_parallel.style.RowwiseParallel``.

Tests cover:
- Default and custom constructor parameters
- Linear partition logic (weight Shard(1), bias Replicate)
- Embedding partition logic (weight Shard(0))
- Unsupported module type rejection
- Input preparation (local tensor -> DTensor, redistribute)
- Output preparation (redistribute, to_local)
- Integration with ``distribute_module`` via ``apply``
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.core.tensor_parallel.api import parallelize_module
from hyper_parallel.core.tensor_parallel.style import ParallelStyle, RowwiseParallel, ColwiseParallel
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


def _patch_dist_rank_for_ut(world_size: int = 1):
    """Patch ``torch.distributed`` rank helpers for CPU-only DTensor redistribute (see ``test_style``)."""
    return patch.multiple(
        "torch.distributed",
        get_rank=MagicMock(return_value=0),
        get_world_size=MagicMock(return_value=world_size),
    )


class TestRowwiseParallelInit(unittest.TestCase):
    """Tests for RowwiseParallel constructor defaults and overrides."""

    def test_default_parameters(self):
        """
        Feature: RowwiseParallel default constructor values
        Description: instantiate RowwiseParallel with no arguments
        Expectation: input_layouts is (Shard(-1),), output_layouts is (Replicate(),),
            desired_input_layouts is (Shard(-1),), use_local_output is True
        """
        style = RowwiseParallel()
        self.assertEqual(style.input_layouts, (Shard(-1),))
        self.assertEqual(style.output_layouts, (Replicate(),))
        self.assertEqual(style.desired_input_layouts, (Shard(-1),))
        self.assertTrue(style.use_local_output)

    def test_custom_input_layouts(self):
        """
        Feature: RowwiseParallel custom input_layouts
        Description: pass Replicate() as input_layouts
        Expectation: input_layouts is (Replicate(),)
        """
        style = RowwiseParallel(input_layouts=Replicate())
        self.assertEqual(style.input_layouts, (Replicate(),))

    def test_custom_output_layouts(self):
        """
        Feature: RowwiseParallel custom output_layouts
        Description: pass Shard(0) as output_layouts
        Expectation: output_layouts is (Shard(0),)
        """
        style = RowwiseParallel(output_layouts=Shard(0))
        self.assertEqual(style.output_layouts, (Shard(0),))

    def test_use_local_output_false(self):
        """
        Feature: RowwiseParallel with use_local_output=False
        Description: set use_local_output to False
        Expectation: use_local_output is False
        """
        style = RowwiseParallel(use_local_output=False)
        self.assertFalse(style.use_local_output)

    def test_is_subclass_of_parallel_style(self):
        """
        Feature: RowwiseParallel inheritance
        Description: check RowwiseParallel is a ParallelStyle
        Expectation: isinstance check passes
        """
        style = RowwiseParallel()
        self.assertIsInstance(style, ParallelStyle)

    def test_src_data_rank_inherited(self):
        """
        Feature: RowwiseParallel inherits src_data_rank default
        Description: fresh RowwiseParallel instance
        Expectation: src_data_rank equals 0
        """
        style = RowwiseParallel()
        self.assertEqual(style.src_data_rank, 0)

    def test_repr_contains_key_fields(self):
        """
        Feature: RowwiseParallel.__repr__
        Description: instantiate with custom layouts
        Expectation: repr includes class name and layout fields
        """
        style = RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(0),
            use_local_output=False,
        )
        r = repr(style)
        self.assertIn("RowwiseParallel", r)
        self.assertIn("use_local_output=False", r)


class TestRowwiseParallelApply(unittest.TestCase):
    """Tests for RowwiseParallel.apply with mocked distribute_module."""

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
    def test_apply_linear_calls_distribute_module(self, mock_style_platform, mock_mesh_platform):
        """
        Feature: RowwiseParallel.apply on Linear module
        Description: apply RowwiseParallel to nn.Linear with mocked distribute_module
        Expectation: distribute_module is called; desired_input_layouts is (Shard(-1),)
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = True
        mock_style_platform.is_embedding_module.return_value = False
        mock_style_platform.Module = nn.Module

        style = RowwiseParallel()
        module = nn.Linear(8, 8)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.return_value = module
            result = style.apply(module, mesh)
            mock_dist.assert_called_once()
            self.assertIs(result, module)
            self.assertEqual(style.desired_input_layouts, (Shard(-1),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_linear_invokes_distribute_module_callbacks(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: RowwiseParallel.apply registers callable hooks on distribute_module
        Description: capture and invoke partition_fn, input_fn, output_fn for Linear
        Expectation: partition_fn calls _partition_linear_fn; I/O hooks delegate to static helpers
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = True
        mock_style_platform.is_embedding_module.return_value = False
        mock_style_platform.Module = nn.Module

        style = RowwiseParallel()
        module = nn.Linear(4, 4)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist, \
             patch.object(style, "_partition_linear_fn") as mock_partition, \
             patch.object(RowwiseParallel, "_prepare_input_fn", return_value="inp") as mock_in, \
             patch.object(RowwiseParallel, "_prepare_output_fn", return_value="out") as mock_out:
            mock_dist.return_value = module
            style.apply(module, mesh)
            partition_fn, input_fn, output_fn = mock_dist.call_args[0][2:5]
            partition_fn("linear", module, mesh)
            mock_partition.assert_called_once_with(module, mesh)
            inp = input_fn(module, (torch.randn(2, 2),), mesh)
            out = output_fn(module, MagicMock(spec=DTensor), mesh)
            mock_in.assert_called_once()
            mock_out.assert_called_once()
            self.assertEqual(inp, "inp")
            self.assertEqual(out, "out")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_embedding_calls_distribute_module(self, mock_style_platform, mock_mesh_platform):
        """
        Feature: RowwiseParallel.apply on Embedding module
        Description: apply RowwiseParallel to nn.Embedding with mocked distribute_module
        Expectation: distribute_module is called; desired_input_layouts is (Replicate(),)
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = False
        mock_style_platform.is_embedding_module.return_value = True
        mock_style_platform.Module = nn.Module

        style = RowwiseParallel()
        module = nn.Embedding(100, 64)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.return_value = module
            result = style.apply(module, mesh)
            mock_dist.assert_called_once()
            self.assertIs(result, module)
            self.assertEqual(style.desired_input_layouts, (Replicate(),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_embedding_invokes_partition_fn(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: RowwiseParallel.apply partition_fn for Embedding
        Description: invoke captured partition_fn after apply on Embedding module
        Expectation: _partition_embedding_fn is called once
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = False
        mock_style_platform.is_embedding_module.return_value = True
        mock_style_platform.Module = nn.Module

        style = RowwiseParallel()
        module = nn.Embedding(8, 4)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist, \
             patch.object(style, "_partition_embedding_fn") as mock_partition:
            mock_dist.return_value = module
            style.apply(module, mesh)
            partition_fn = mock_dist.call_args[0][2]
            partition_fn("emb", module, mesh)
            mock_partition.assert_called_once_with(module, mesh)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_unsupported_module_raises(self, mock_style_platform, mock_mesh_platform):
        """
        Feature: RowwiseParallel.apply rejects unsupported module types
        Description: apply RowwiseParallel to nn.LayerNorm
        Expectation: raises NotImplementedError mentioning supported types
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = False
        mock_style_platform.is_embedding_module.return_value = False
        mock_style_platform.Module = nn.Module

        style = RowwiseParallel()
        module = nn.LayerNorm(8)

        with self.assertRaises(NotImplementedError) as ctx:
            style.apply(module, mesh)
        self.assertIn("Linear", str(ctx.exception))
        self.assertIn("Embedding", str(ctx.exception))


class TestRowwiseParallelPartition(unittest.TestCase):
    """Tests for RowwiseParallel partition functions."""

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
    def test_partition_linear_fn_weight_shard1_bias_replicate(self, mock_platform):
        """
        Feature: RowwiseParallel._partition_linear_fn sharding strategy
        Description: call _partition_linear_fn on nn.Linear(8, 16, bias=True)
        Expectation: weight uses [Shard(1)], bias uses [Replicate()]
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = RowwiseParallel()
        module = nn.Linear(8, 16, bias=True)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_tensor") as mock_dt, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_param_source") as mock_src, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_new_parameter") as mock_new, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_set_param"):
            mock_src.side_effect = lambda p: p.data
            mock_dt.return_value = MagicMock()
            mock_new.return_value = nn.Parameter(torch.empty(1))

            style._partition_linear_fn(module, mesh)

            self.assertEqual(mock_dt.call_count, 2)
            placements_used = [call[0][2] for call in mock_dt.call_args_list]
            self.assertEqual(placements_used[0], [Shard(1)])
            self.assertEqual(placements_used[1], [Replicate()])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partition_linear_fn_no_bias(self, mock_platform):
        """
        Feature: RowwiseParallel._partition_linear_fn with no bias
        Description: call _partition_linear_fn on nn.Linear(8, 16, bias=False)
        Expectation: distribute_tensor called once for weight only with [Shard(1)]
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = RowwiseParallel()
        module = nn.Linear(8, 16, bias=False)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_tensor") as mock_dt, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_param_source") as mock_src, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_new_parameter") as mock_new, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_set_param"):
            mock_src.side_effect = lambda p: p.data
            mock_dt.return_value = MagicMock()
            mock_new.return_value = nn.Parameter(torch.empty(1))

            style._partition_linear_fn(module, mesh)

            self.assertEqual(mock_dt.call_count, 1)
            self.assertEqual(mock_dt.call_args[0][2], [Shard(1)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partition_embedding_fn_uses_shard0(self, mock_platform):
        """
        Feature: RowwiseParallel._partition_embedding_fn sharding strategy
        Description: call _partition_embedding_fn on nn.Embedding(100, 64)
        Expectation: distribute_tensor is called with [Shard(0)]
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = RowwiseParallel()
        module = nn.Embedding(100, 64)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_tensor") as mock_dt, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_param_source") as mock_src, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_new_parameter") as mock_new, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_set_param"):
            mock_src.side_effect = lambda p: p.data
            mock_dt.return_value = MagicMock()
            mock_new.return_value = nn.Parameter(torch.empty(1))

            style._partition_embedding_fn(module, mesh)

            for call in mock_dt.call_args_list:
                self.assertEqual(call[0][2], [Shard(0)])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_partition_embedding_fn_skips_none_param(self, mock_platform):
        """
        Feature: RowwiseParallel._partition_embedding_fn skips None parameters
        Description: _distribute_module_iter_params yields a None parameter slot
        Expectation: distribute_tensor is not called for the None entry
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = RowwiseParallel()
        module = nn.Embedding(8, 4)
        weight = nn.Parameter(torch.randn(8, 4))

        with patch(
            "hyper_parallel.core.tensor_parallel.style._distribute_module_iter_params",
            return_value=[("weight", weight), ("unused", None)],
        ), patch("hyper_parallel.core.tensor_parallel.style.distribute_tensor") as mock_dt, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_param_source") as mock_src, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_new_parameter") as mock_new, \
             patch("hyper_parallel.core.tensor_parallel.style._distribute_module_set_param"):
            mock_src.side_effect = lambda p: p.data
            mock_dt.return_value = MagicMock()
            mock_new.return_value = nn.Parameter(torch.empty(1))

            style._partition_embedding_fn(module, mesh)

            self.assertEqual(mock_dt.call_count, 1)


class TestRowwiseParallelIO(unittest.TestCase):
    """Tests for RowwiseParallel input/output preparation functions."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 1):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size: int = 1):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    def test_prepare_input_fn_wraps_plain_tensor(self):
        """
        Feature: _prepare_input_fn wraps local tensor
        Description: pass plain torch.Tensor as input[0]
        Expectation: DTensor.from_local is called with correct device_mesh and input_layouts
        """
        input_layouts = (Shard(-1),)
        desired_layouts = (Shard(-1),)
        mesh = MagicMock()
        local_tensor = torch.randn(4, 8)
        inputs = (local_tensor,)

        with patch.object(DTensor, "from_local") as mock_from_local:
            mock_dtensor = MagicMock(spec=DTensor)
            mock_from_local.return_value = mock_dtensor

            RowwiseParallel._prepare_input_fn(
                input_layouts, desired_layouts, inputs, mesh
            )
            mock_from_local.assert_called_once_with(local_tensor, mesh, input_layouts)

    def test_prepare_input_fn_redistributes_if_layouts_differ(self):
        """
        Feature: _prepare_input_fn redistributes on layout mismatch
        Description: input_layouts=(Replicate(),), desired=(Shard(-1),)
        Expectation: redistribute is called
        """
        input_layouts = (Replicate(),)
        desired_layouts = (Shard(-1),)
        mesh = MagicMock()
        mock_dtensor = MagicMock(spec=DTensor)

        with patch.object(DTensor, "from_local", return_value=mock_dtensor):
            RowwiseParallel._prepare_input_fn(
                input_layouts, desired_layouts, (torch.randn(4, 8),), mesh
            )
            mock_dtensor.redistribute.assert_called_once_with(mesh, desired_layouts)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_prepare_output_fn_to_local(self, mock_platform):
        """
        Feature: _prepare_output_fn converts to local tensor
        Description: use_local_output=True, matching placements on a real DTensor
        Expectation: returns local torch.Tensor
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        local = torch.randn(4, 8)
        dt_out = DTensor.from_local(local, mesh, [Replicate()])
        with _patch_dist_rank_for_ut(world_size=1):
            result = RowwiseParallel._prepare_output_fn(
                (Replicate(),), True, dt_out, mesh
            )
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.allclose(result, local))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_prepare_output_fn_keeps_dtensor(self, mock_platform):
        """
        Feature: _prepare_output_fn keeps DTensor when use_local_output=False
        Description: use_local_output=False, matching placements
        Expectation: same DTensor instance is returned
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        dt_out = DTensor.from_local(torch.randn(4, 8), mesh, [Replicate()])
        result = RowwiseParallel._prepare_output_fn(
            (Replicate(),), False, dt_out, mesh
        )
        self.assertIs(result, dt_out)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_prepare_output_fn_redistributes_if_needed(self, mock_platform):
        """
        Feature: _prepare_output_fn redistributes when placements differ
        Description: DTensor has Shard(-1); desired output_layouts is Replicate()
        Expectation: result equals redistribute(..., Replicate()).to_local()
        """
        mesh = self._make_1d_mesh(mock_platform, size=1)
        local = torch.randn(4, 8)
        dt_sharded = DTensor.from_local(local, mesh, [Shard(-1)])
        with _patch_dist_rank_for_ut(world_size=1):
            expected = dt_sharded.redistribute(mesh, [Replicate()]).to_local()
            result = RowwiseParallel._prepare_output_fn(
                (Replicate(),), True, dt_sharded, mesh
            )
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.allclose(result, expected))

    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_prepare_output_fn_embedding_wraps_partial(self, mock_platform):
        """
        Feature: RowwiseParallel._prepare_output_fn Embedding partial path
        Description: plain tensor output from Embedding with embedding module passed in
        Expectation: DTensor.from_local is called with Partial(sum) before redistribute
        """
        mock_platform.is_embedding_module.return_value = True
        mesh = MagicMock()
        module = nn.Embedding(10, 4)
        local_out = torch.randn(2, 4)
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Partial("sum"),)
        redistributed = MagicMock(spec=DTensor)
        redistributed.placements = (Replicate(),)
        mock_dt.redistribute.return_value = redistributed
        redistributed.to_local.return_value = local_out

        with patch.object(DTensor, "from_local", return_value=mock_dt) as mock_from_local:
            result = RowwiseParallel._prepare_output_fn(
                (Replicate(),),
                True,
                local_out,
                mesh,
                module,
            )
            mock_from_local.assert_called_once()
            placements = mock_from_local.call_args[0][2]
            self.assertEqual(len(placements), 1)
            self.assertTrue(placements[0].is_partial("sum"))
            mock_dt.redistribute.assert_called_once_with(mesh, (Replicate(),))
            self.assertIs(result, local_out)

    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_prepare_output_fn_non_embedding_plain_tensor_raises(self, mock_platform):
        """
        Feature: RowwiseParallel._prepare_output_fn rejects plain tensors for Linear
        Description: non-DTensor output without an Embedding module
        Expectation: TypeError mentioning DTensor / unsupported module
        """
        mock_platform.is_embedding_module.return_value = False
        with self.assertRaises(TypeError) as ctx:
            RowwiseParallel._prepare_output_fn(
                (Replicate(),),
                True,
                torch.randn(2, 4),
                MagicMock(),
                nn.Linear(4, 4),
            )
        self.assertIn("DTensor", str(ctx.exception))


class TestColRowComposition(unittest.TestCase):
    """Tests for composing ColwiseParallel and RowwiseParallel on an MLP."""

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
    def test_mlp_colwise_rowwise_composition(self, mock_style_platform, mock_mesh_platform):
        """
        Feature: ColwiseParallel + RowwiseParallel composition on MLP
        Description: parallelize_module with linear1=ColwiseParallel, linear2=RowwiseParallel
        Expectation: distribute_module called twice (once per submodule)
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.is_linear_module.return_value = True
        mock_style_platform.is_embedding_module.return_value = False
        mock_style_platform.Module = nn.Module

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 16)
                self.linear2 = nn.Linear(16, 8)

        model = MLP()

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.side_effect = lambda m, *args, **kwargs: m
            parallelize_module(
                model, mesh,
                {"linear1": ColwiseParallel(), "linear2": RowwiseParallel()},
            )
            self.assertEqual(mock_dist.call_count, 2)


if __name__ == "__main__":
    unittest.main()
