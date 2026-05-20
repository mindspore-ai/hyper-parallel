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
"""Unit tests for hyper_parallel.core.shard.custom_shard"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np

from hyper_parallel.core.dtensor.dtensor import DTensor, _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.shard.custom_shard import custom_shard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


def _make_mesh(mock_platform, mesh_shape, mesh_dim_names):
    EXISTING_COMM_GROUPS.clear()
    _DEVICE_MESH_MAP.clear()
    mock_platform.get_rank.return_value = 0
    mock_platform.get_world_size.return_value = int(np.prod(mesh_shape))
    mock_platform.tensor_to_numpy.side_effect = (
        lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
    )
    return init_device_mesh(
        device_type="npu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        init_backend=False,
    )


def _make_dtensor(layout):
    """Create a minimal DTensor mock with given layout."""
    dt = MagicMock(spec=DTensor)
    dt.layout = layout
    local_np = np.array([1.0, 2.0, 3.0, 4.0])
    dt.to_local.return_value = local_np
    dt.redistribute.return_value = dt
    return dt


class TestCustomShardNoDistributed(unittest.TestCase):
    """
    Feature: custom_shard with plain (non-DTensor) inputs
    Description: When no DTensor inputs, function runs natively without wrapping.
    Expectation: Output is the raw function result with no DTensor conversion.
    """

    def test_no_dtensor_returns_plain_output(self):
        """No DTensor inputs → func called normally, output returned as-is."""
        def add(x, y):
            return x + y

        wrapped = custom_shard(
            func=add,
            device_mesh=MagicMock(),
            out_placements=((Replicate(),),),
            in_placements=(None, None),
        )
        result = wrapped(10, 20)
        self.assertEqual(result, 30)

    def test_no_dtensor_no_in_placements(self):
        """No DTensor inputs, in_placements=None → func runs normally."""
        def identity(x):
            return x

        wrapped = custom_shard(
            func=identity,
            device_mesh=MagicMock(),
            out_placements=((Replicate(),),),
            in_placements=None,
        )
        result = wrapped(42)
        self.assertEqual(result, 42)


class TestCustomShardInPlacementsValidation(unittest.TestCase):
    """
    Feature: custom_shard input validation
    Description: Validate in_placements length and type requirements.
    Expectation: ValueError/RuntimeError/TypeError for mismatches.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_in_placements_length_mismatch_raises(self, mock_platform):
        """in_placements length != args count raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        wrapped = custom_shard(
            func=lambda x, y: x,
            device_mesh=mesh,
            out_placements=((Replicate(),),),
            in_placements=((Replicate(),),),  # 1 placement, 2 args
        )
        with self.assertRaises(ValueError):
            wrapped(dt, dt)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dtensor_input_without_in_placements_raises(self, mock_platform):
        """DTensor input when in_placements is None raises RuntimeError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        wrapped = custom_shard(
            func=lambda x: x,
            device_mesh=mesh,
            out_placements=((Replicate(),),),
            in_placements=None,
        )
        with self.assertRaises(RuntimeError):
            wrapped(dt)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dtensor_at_none_placement_position_raises(self, mock_platform):
        """DTensor input at position where in_placements[i] is None raises TypeError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        wrapped = custom_shard(
            func=lambda x: x,
            device_mesh=mesh,
            out_placements=((Replicate(),),),
            in_placements=(None,),
        )
        with self.assertRaises(TypeError):
            wrapped(dt)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_non_dtensor_with_placement_raises(self, mock_platform):
        """Non-DTensor input at a position that expects Placement raises TypeError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))

        wrapped = custom_shard(
            func=lambda x: x,
            device_mesh=mesh,
            out_placements=((Replicate(),),),
            in_placements=((Replicate(),),),
        )
        with self.assertRaises(TypeError):
            wrapped(42)  # plain int, not DTensor


class TestCustomShardOutputValidation(unittest.TestCase):
    """
    Feature: custom_shard output wrapping
    Description: Validate output_placements length and type requirements.
    Expectation: ValueError/TypeError for mismatches between output count and placements.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_output_count_mismatch_raises(self, mock_platform):
        """Output count != out_placements count raises ValueError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        # func returns 1 tensor, but out_placements has 2
        wrapped = custom_shard(
            func=lambda x: np.array([1.0]),
            device_mesh=mesh,
            out_placements=((Replicate(),), (Replicate(),)),
            in_placements=((Replicate(),),),
        )
        with self.assertRaises(ValueError):
            wrapped(dt)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tensor_output_with_none_placement_raises(self, mock_platform):
        """Tensor output with None out_placements raises TypeError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        # Patch Tensor to numpy.ndarray so test is platform-agnostic
        with patch("hyper_parallel.core.shard.custom_shard.Tensor", np.ndarray):
            local_array = np.array([1.0])
            wrapped = custom_shard(
                func=lambda x: local_array,
                device_mesh=mesh,
                out_placements=(None,),
                in_placements=((Replicate(),),),
            )
            with self.assertRaises(TypeError):
                wrapped(dt)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_non_tensor_output_with_placement_raises(self, mock_platform):
        """Non-tensor output (e.g., int) with non-None placement raises TypeError."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        wrapped = custom_shard(
            func=lambda x: 42,
            device_mesh=mesh,
            out_placements=((Replicate(),),),
            in_placements=((Replicate(),),),
        )
        with self.assertRaises(TypeError):
            wrapped(dt)


class TestCustomShardRedistributeInputs(unittest.TestCase):
    """
    Feature: custom_shard redistribute_inputs flag
    Description: When redistribute_inputs=False, DTensor.redistribute is not called.
    Expectation: to_local() is called; redistribute() is not called.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_no_redistribute_when_flag_false(self, mock_platform):
        """redistribute_inputs=False skips redistribution."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        called = []

        def func(x):
            called.append(x)
            return x

        wrapped = custom_shard(
            func=func,
            device_mesh=mesh,
            out_placements=(None,),
            in_placements=((Replicate(),),),
            redistribute_inputs=False,
        )
        # With non-tensor output and None placement, this will raise TypeError
        # because the output validation runs after - that's ok, we check redistribute was not called
        try:
            wrapped(dt)
        except TypeError:
            pass
        dt.redistribute.assert_not_called()


class TestCustomShardTupleOutput(unittest.TestCase):
    """
    Feature: custom_shard with tuple output
    Description: When the wrapped function returns a tuple, each element is
                 handled according to its out_placement.
    Expectation: Tuple of DTensors (or pass-throughs) is returned.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_tuple_output_multiple_dtensors(self, mock_mesh_platform):
        """Tuple of two Tensor outputs yields two DTensors."""
        mesh = _make_mesh(mock_mesh_platform, (1,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        out1 = np.array([1.0])
        out2 = np.array([2.0])

        with patch("hyper_parallel.core.shard.custom_shard.Tensor", np.ndarray):
            with patch.object(DTensor, "from_local", return_value=MagicMock(spec=DTensor)):
                wrapped = custom_shard(
                    func=lambda x: (out1, out2),
                    device_mesh=mesh,
                    out_placements=((Replicate(),), (Replicate(),)),
                    in_placements=((Replicate(),),),
                )
                result = wrapped(dt)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_single_tensor_output_returns_dtensor(self, mock_mesh_platform):
        """Single Tensor output with out_placements wraps in DTensor."""
        mesh = _make_mesh(mock_mesh_platform, (1,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        local_out = np.array([1.0, 2.0])

        with patch("hyper_parallel.core.shard.custom_shard.Tensor", np.ndarray):
            with patch.object(DTensor, "from_local", return_value=MagicMock(spec=DTensor)):
                wrapped = custom_shard(
                    func=lambda x: local_out,
                    device_mesh=mesh,
                    out_placements=((Replicate(),),),
                    in_placements=((Replicate(),),),
                )
                result = wrapped(dt)
                self.assertIsNotNone(result)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_non_tensor_output_with_none_placement_passes_through(self, mock_platform):
        """Non-tensor output with None out_placement passes through unchanged."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        layout = _build_layout(mesh, (Replicate(),), 1)
        dt = _make_dtensor(layout)

        wrapped = custom_shard(
            func=lambda x: (99,),
            device_mesh=mesh,
            out_placements=(None,),
            in_placements=((Replicate(),),),
        )
        result = wrapped(dt)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], 99)


if __name__ == "__main__":
    unittest.main()
