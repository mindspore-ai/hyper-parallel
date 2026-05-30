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
"""Unit tests for ``hyper_parallel.core.tensor_parallel.style.SequenceParallel``."""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.tensor_parallel.style import ParallelStyle, SequenceParallel
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestSequenceParallelInit(unittest.TestCase):
    """Constructor and metadata for :class:`SequenceParallel`."""

    def test_default_parameters(self):
        """
        Feature: SequenceParallel default constructor values
        Description: instantiate with no arguments
        Expectation: sequence_sharding is (Shard(1),); use_local_output is False
        """
        style = SequenceParallel()
        self.assertEqual(style.sequence_sharding, (Shard(1),))
        self.assertFalse(style.use_local_output)

    def test_custom_sequence_dim(self):
        """
        Feature: SequenceParallel custom sequence_dim
        Description: pass sequence_dim=2
        Expectation: sequence_sharding is (Shard(2),)
        """
        style = SequenceParallel(sequence_dim=2)
        self.assertEqual(style.sequence_sharding, (Shard(2),))

    def test_use_local_output_true(self):
        """
        Feature: SequenceParallel use_local_output=True
        Description: set use_local_output to True
        Expectation: use_local_output is True
        """
        style = SequenceParallel(use_local_output=True)
        self.assertTrue(style.use_local_output)

    def test_is_subclass_of_parallel_style(self):
        """
        Feature: SequenceParallel inheritance
        Description: isinstance check against ParallelStyle
        Expectation: passes
        """
        style = SequenceParallel()
        self.assertIsInstance(style, ParallelStyle)

    def test_src_data_rank_inherited(self):
        """
        Feature: SequenceParallel inherits src_data_rank default
        Expectation: src_data_rank equals 0
        """
        style = SequenceParallel()
        self.assertEqual(style.src_data_rank, 0)

    def test_repr_contains_sequence_dim_and_flag(self):
        """
        Feature: SequenceParallel.__repr__
        Description: repr of default instance
        Expectation: contains class name, sequence_dim=1, use_local_output
        """
        r = repr(SequenceParallel(sequence_dim=1, use_local_output=False))
        self.assertIn("SequenceParallel", r)
        self.assertIn("sequence_dim=1", r)
        self.assertIn("use_local_output=False", r)


class TestSequenceParallelApply(unittest.TestCase):
    """``apply`` wires ``distribute_module`` like other parallel styles."""

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
    def test_apply_layernorm_calls_distribute_module(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: SequenceParallel.apply on LayerNorm
        Description: mock distribute_module
        Expectation: distribute_module called once; returns module from mock
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.Module = nn.Module

        style = SequenceParallel()
        module = nn.LayerNorm(8)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist:
            mock_dist.return_value = module
            result = style.apply(module, mesh)
            mock_dist.assert_called_once()
            self.assertIs(result, module)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_apply_invokes_distribute_module_callbacks(
        self, mock_style_platform, mock_mesh_platform
    ):
        """
        Feature: SequenceParallel.apply registers partition/input/output callbacks
        Description: invoke callbacks captured from mocked distribute_module
        Expectation: partition_fn is no-op; input_fn/output_fn delegate to static helpers
        """
        mesh = self._make_1d_mesh(mock_mesh_platform)
        mock_style_platform.Module = nn.Module

        style = SequenceParallel(sequence_dim=1)
        module = nn.LayerNorm(8)

        with patch("hyper_parallel.core.tensor_parallel.style.distribute_module") as mock_dist, \
             patch.object(SequenceParallel, "_prepare_input_fn", return_value="inp") as mock_in, \
             patch.object(SequenceParallel, "_prepare_output_fn", return_value="out") as mock_out:
            mock_dist.return_value = module
            style.apply(module, mesh)
            partition_fn, input_fn, output_fn = mock_dist.call_args[0][2:5]
            self.assertIsNone(partition_fn("", module, mesh))
            inp = input_fn(module, (torch.randn(2, 4, 8),), mesh)
            out = output_fn(module, MagicMock(), mesh)
            mock_in.assert_called_once()
            mock_out.assert_called_once()
            self.assertEqual(inp, "inp")
            self.assertEqual(out, "out")


class TestSequenceParallelIO(unittest.TestCase):
    """Input/output hooks for sequence-sharded activations."""

    def test_prepare_input_fn_wraps_plain_tensor(self):
        """
        Feature: _prepare_input_fn wraps local tensor with from_local
        Description: first input is torch.Tensor
        Expectation: DTensor.from_local called with sequence_sharding
        """
        sequence_sharding = (Shard(1),)
        mesh = MagicMock()
        local_tensor = torch.randn(2, 4, 8)
        inputs = (local_tensor,)
        mod = nn.LayerNorm(8)

        with patch.object(DTensor, "from_local") as mock_from_local:
            mock_dtensor = MagicMock(spec=DTensor)
            mock_from_local.return_value = mock_dtensor

            SequenceParallel._prepare_input_fn(
                sequence_sharding, mod, inputs, mesh
            )
            mock_from_local.assert_called_once_with(
                local_tensor, mesh, sequence_sharding
            )

    def test_prepare_input_fn_dtensor_matching_placements_no_redistribute(self):
        """
        Feature: _prepare_input_fn leaves DTensor unchanged when placements match
        """
        sequence_sharding = (Shard(1),)
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Shard(1),)
        inputs = (mock_dt,)
        mod = nn.LayerNorm(8)

        out = SequenceParallel._prepare_input_fn(
            sequence_sharding, mod, inputs, mesh
        )
        self.assertIs(out[0], mock_dt)
        mock_dt.redistribute.assert_not_called()

    def test_prepare_input_fn_dtensor_redistributes_when_mismatch(self):
        """
        Feature: _prepare_input_fn redistributes DTensor to sequence_sharding
        """
        sequence_sharding = (Shard(1),)
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        mock_dt.placements = (Shard(0),)
        redistributed = MagicMock(spec=DTensor)
        redistributed.placements = (Shard(1),)
        mock_dt.redistribute.return_value = redistributed
        inputs = (mock_dt,)
        mod = nn.LayerNorm(8)

        out = SequenceParallel._prepare_input_fn(
            sequence_sharding, mod, inputs, mesh
        )
        mock_dt.redistribute.assert_called_once_with(mesh, sequence_sharding)
        self.assertIs(out[0], redistributed)

    def test_prepare_input_fn_invalid_type_raises(self):
        """
        Feature: _prepare_input_fn rejects non-tensor non-DTensor
        """
        sequence_sharding = (Shard(1),)
        mesh = MagicMock()
        mod = nn.LayerNorm(8)
        with self.assertRaises(ValueError) as ctx:
            SequenceParallel._prepare_input_fn(
                sequence_sharding, mod, (object(),), mesh
            )
        self.assertIn("tensor or DTensor", str(ctx.exception))

    def test_prepare_output_fn_to_local(self):
        """
        Feature: _prepare_output_fn with use_local_output=True
        """
        mock_outputs = MagicMock()
        mock_outputs.to_local.return_value = torch.randn(2, 4, 8)
        result = SequenceParallel._prepare_output_fn(True, mock_outputs)
        mock_outputs.to_local.assert_called_once()
        self.assertIsInstance(result, torch.Tensor)

    def test_prepare_output_fn_keeps_dtensor(self):
        """
        Feature: _prepare_output_fn with use_local_output=False
        """
        mock_outputs = MagicMock()
        result = SequenceParallel._prepare_output_fn(False, mock_outputs)
        mock_outputs.to_local.assert_not_called()
        self.assertIs(result, mock_outputs)


if __name__ == "__main__":
    unittest.main()
