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
"""parallel_vstack test"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.shard.ops.parallel_vstack import (
    VstackDistributedOp,
    _normalize_vstack_args,
    _promote_tensor_map,
)
from hyper_parallel.core.dtensor.device_mesh import (
    init_device_mesh,
    _DEVICE_MESH_MAP,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS

op = VstackDistributedOp("vstack")


class TestVstackDistributedOp(unittest.TestCase):
    """Unit tests for VstackDistributedOp."""

    def setUp(self) -> None:
        """Clear global caches before each test to ensure isolation."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Restore global cache state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=4):
        """Configure common mock-platform attributes used across tests."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
        )
        mock_platform.platform_type = MagicMock()

    def _make_2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=4)
        return init_device_mesh(
            device_type="cpu", mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"), init_backend=False,
        )

    def _make_2x2x2_mesh(self, mock_platform):
        """Set up mock and return a standard 2x2x2 mesh via init_device_mesh."""
        self._setup_mock_platform(mock_platform, world_size=8)
        return init_device_mesh(
            device_type="cpu", mesh_shape=(2, 2, 2),
            mesh_dim_names=("dp", "cp", "tp"), init_backend=False,
        )

    def _run_infer(self, layouts):
        """Helper to run infer_layout and return output layout."""
        output_layouts, extra_info = op.infer_layout(layouts)
        self.assertIsNone(extra_info, f"extra_info should be None, got {extra_info}")
        return output_layouts[0]

    # ---- infer_layout success cases ----

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_2d_all_replicated(self, mock_platform):
        """
        Feature: 2D all-replicated inputs on 2x2 mesh.
        Description: Two 2D DTensors with tensor_map (-1,-1).
        Expectation: output tensor_map is (-1,-1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        output_layout = self._run_infer([layout, layout])
        expected_map = (-1, -1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_2d_non_cat_dim_sharded(self, mock_platform):
        """
        Feature: 2D inputs with non-cat dim sharded.
        Description: tensor_map (-1, 0), dim=0 Replicate, dim=1 sharded on tp.
        Expectation: output tensor_map is (-1, 0).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(1)), 2)
        output_layout = self._run_infer([layout, layout])
        expected_map = (-1, 0)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_3d_mixed(self, mock_platform):
        """
        Feature: 3D inputs on 2x2x2 mesh with mixed sharding.
        Description: tensor_map (-1, 1, 0), dim=0 Replicate.
        Expectation: output tensor_map is (-1, 1, 0).
        """
        mesh = self._make_2x2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Shard(1), Shard(2)), 3)
        output_layout = self._run_infer([layout, layout])
        expected_map = (-1, 1, 0)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_0d_scalar(self, mock_platform):
        """
        Feature: 0D scalar inputs promoted to 2D.
        Description: tensor_map (), promoted to (-1,-1).
        Expectation: output tensor_map is (-1,-1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 0)
        output_layout = self._run_infer([layout, layout])
        expected_map = (-1, -1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_1d_replicated(self, mock_platform):
        """
        Feature: 1D replicated inputs promoted to 2D.
        Description: tensor_map (-1,), promoted to (-1,-1).
        Expectation: output tensor_map is (-1,-1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 1)
        output_layout = self._run_infer([layout, layout])
        expected_map = (-1, -1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_1d_sharded(self, mock_platform):
        """
        Feature: 1D sharded inputs promoted to 2D.
        Description: tensor_map (1,), promoted to (-1,1).
        Expectation: output tensor_map is (-1, 1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 1)
        output_layout = self._run_infer([layout, layout])
        expected_map = (-1, 1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_0d_plus_2d(self, mock_platform):
        """
        Feature: Mixed 0D + 2D inputs, promoted layouts match.
        Description: 0D () + 2D (-1,-1), both promoted to (-1,-1).
        Expectation: output tensor_map is (-1,-1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout_0d = _build_layout(mesh, (Replicate(), Replicate()), 0)
        layout_2d = _build_layout(mesh, (Replicate(), Replicate()), 2)
        output_layout = self._run_infer([layout_0d, layout_2d])
        expected_map = (-1, -1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_1d_plus_2d(self, mock_platform):
        """
        Feature: Mixed 1D + 2D inputs, promoted layouts match.
        Description: 1D (1,) + 2D (-1,1), both promoted to (-1,1).
        Expectation: output tensor_map is (-1, 1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout_1d = _build_layout(mesh, (Shard(0), Replicate()), 1)
        layout_2d = _build_layout(mesh, (Shard(1), Replicate()), 2)
        output_layout = self._run_infer([layout_1d, layout_2d])
        expected_map = (-1, 1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_single_input(self, mock_platform):
        """
        Feature: Single input is valid.
        Description: One 2D DTensor, tensor_map (-1,-1).
        Expectation: output tensor_map is (-1,-1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        output_layout = self._run_infer([layout])
        expected_map = (-1, -1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_multi_input(self, mock_platform):
        """
        Feature: Three inputs with identical layout.
        Description: Three 2D DTensors with tensor_map (-1,-1).
        Expectation: output tensor_map is (-1,-1).
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        output_layout = self._run_infer([layout, layout, layout])
        expected_map = (-1, -1)
        self.assertEqual(output_layout.tensor_map, expected_map,
                         f"Expected {expected_map}, got {output_layout.tensor_map}")

    # ---- infer_layout error cases ----

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_error_partial(self, mock_platform):
        """
        Feature: Partial input is rejected before promotion.
        Description: 2D layout with Partial on dp axis.
        Expectation: ValueError with "Partial status".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        layout.set_partial_by_dev_axis("dp", "sum")
        with self.assertRaisesRegex(ValueError, "Partial status"):
            op.infer_layout([layout, layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_error_promoted_mismatch(self, mock_platform):
        """
        Feature: Promoted layouts mismatch.
        Description: 1D (0,) → promoted (-1,0) vs 2D (-1,1) → promoted (-1,1).
        Expectation: ValueError with "input tensors must have the same layout".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        # 1D (Replicate(), Shard(0)) → tensor_map (0,) → promoted (-1,0)
        layout_1d = _build_layout(mesh, (Replicate(), Shard(0)), 1)
        # 2D (Shard(1), Replicate()) → tensor_map (-1,1) → promoted (-1,1)
        layout_2d = _build_layout(mesh, (Shard(1), Replicate()), 2)
        with self.assertRaisesRegex(ValueError, "input tensors must have the same layout"):
            op.infer_layout([layout_1d, layout_2d])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_error_dim0_sharded(self, mock_platform):
        """
        Feature: dim=0 sharded is rejected.
        Description: 2D (0, -1), dim=0 is sharded.
        Expectation: ValueError with "sharded dimension".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Shard(0), Replicate()), 2)
        with self.assertRaisesRegex(ValueError, "sharded dimension"):
            op.infer_layout([layout, layout])

    # ---- preprocess cases ----

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_preprocess_out_not_none(self, mock_platform):
        """
        Feature: out keyword is rejected.
        Description: out=some_tensor passed to preprocess.
        Expectation: ValueError with "out keyword is not supported".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        mock_local = Mock(name="local_tensor")
        fake_dtensor = SimpleNamespace(layout=layout, _local_tensor=mock_local)
        fake_dtensor.to_local = lambda: mock_local

        with self.assertRaisesRegex(ValueError, "out keyword is not supported"):
            op.preprocess(((fake_dtensor, fake_dtensor),), {"out": mock_local})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_preprocess_plain_tensor_rejected(self, mock_platform):
        """
        Feature: Plain Tensor mixed with DTensor is rejected.
        Description: One DTensor and one plain Tensor.
        Expectation: ValueError with "all inputs must be DTensor".
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        mock_local = Mock(name="local_tensor")
        fake_dtensor = SimpleNamespace(layout=layout, _local_tensor=mock_local)
        fake_dtensor.to_local = lambda: mock_local
        plain_tensor = Mock(name="plain_tensor")

        with self.assertRaisesRegex(ValueError, "all inputs must be DTensor"):
            op.preprocess(((fake_dtensor, plain_tensor),), {"out": None})

    @patch("hyper_parallel.core.shard.ops.parallel_vstack.DTensor", new=SimpleNamespace)
    def test_vstack_preprocess_cache_original_layouts(self):
        """
        Feature: cache_values stores original layouts, not promoted.
        Description: Two DTensors with different ndim, cache_values should contain
            original layouts without promotion.
        Expectation: cache_values has 2 elements, both are original Layout objects.
        """
        mock_local = Mock(name="local_tensor")
        mock_layout_0d = Mock(name="layout_0d")
        mock_layout_0d.alias_tensor_map = ()
        mock_layout_1d = Mock(name="layout_1d")
        mock_layout_1d.alias_tensor_map = (1,)

        fake_dtensor_0d = SimpleNamespace(layout=mock_layout_0d, _local_tensor=mock_local)
        fake_dtensor_0d.to_local = lambda: mock_local
        fake_dtensor_1d = SimpleNamespace(layout=mock_layout_1d, _local_tensor=mock_local)
        fake_dtensor_1d.to_local = lambda: mock_local

        local_args, local_kwargs, cache_values = op.preprocess(
            ((fake_dtensor_0d, fake_dtensor_1d),), {"out": None}
        )
        self.assertEqual(len(cache_values), 2,
                         f"Expected 2 cache entries, got {len(cache_values)}")
        self.assertIs(cache_values[0], mock_layout_0d,
                      "cache_values[0] should be the original 0D layout")
        self.assertIs(cache_values[1], mock_layout_1d,
                      "cache_values[1] should be the original 1D layout")

    # ---- normalize function ----

    def test_normalize_vstack_args(self):
        """
        Feature: Argument normalization.
        Description: Verify _normalize_vstack_args returns correct args/kwargs split.
        Expectation: tensors in args, out in kwargs.
        """
        x, y = object(), object()
        args, kwargs = _normalize_vstack_args((x, y), out=None)
        self.assertEqual(args, ((x, y),),
                         f"Expected ((x, y),), got {args}")
        self.assertEqual(kwargs, {"out": None},
                         f"Expected {{'out': None}}, got {kwargs}")

    # ---- promote_tensor_map function ----

    def test_promote_tensor_map_0d(self):
        """
        Feature: atleast_2d promotion of 0D tensor_map.
        Description: () → ("None", "None").
        Expectation: promoted map is ("None", "None").
        """
        result = _promote_tensor_map(())
        self.assertEqual(result, ("None", "None"),
                         f"Expected ('None', 'None'), got {result}")

    def test_promote_tensor_map_1d(self):
        """
        Feature: atleast_2d promotion of 1D tensor_map.
        Description: (1,) → ("None", 1).
        Expectation: promoted map is ("None", 1).
        """
        result = _promote_tensor_map((1,))
        self.assertEqual(result, ("None", 1),
                         f"Expected ('None', 1), got {result}")

    def test_promote_tensor_map_2d_unchanged(self):
        """
        Feature: atleast_2d promotion of 2D tensor_map unchanged.
        Description: (-1, 0) → (-1, 0).
        Expectation: promoted map is (-1, 0).
        """
        result = _promote_tensor_map((-1, 0))
        self.assertEqual(result, (-1, 0),
                         f"Expected (-1, 0), got {result}")

    # ---- get_expand_impl ----

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_vstack_get_expand_impl_returns_none(self, mock_platform):
        """
        Feature: get_expand_impl returns None.
        Description: vstack does not need expand logic.
        Expectation: get_expand_impl returns None.
        """
        mesh = self._make_2x2_mesh(mock_platform)
        layout = _build_layout(mesh, (Replicate(), Replicate()), 2)
        self.assertIsNone(
            op.get_expand_impl(None, ((layout,), None), [layout, layout]),
            "get_expand_impl should return None",
        )


if __name__ == "__main__":
    unittest.main()
