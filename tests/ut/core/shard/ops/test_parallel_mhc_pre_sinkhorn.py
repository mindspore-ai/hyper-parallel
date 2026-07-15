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
"""Unit tests for NpuMhcPreSinkhornDistributedOp."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from hyper_parallel.core.dtensor.dtensor import _build_layout, _LAYOUT_CACHE
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.shard.ops.parallel_mhc_pre_sinkhorn import (
    NpuMhcPreClampSinkhornDistributedOp,
    _normalize_mhc_pre_sinkhorn_args,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


class TestNpuMhcPreSinkhorn(unittest.TestCase):
    """Unit tests for NpuMhcPreSinkhornDistributedOp."""

    def setUp(self) -> None:
        """Clear global state before each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        """Clear global state after each test."""
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _LAYOUT_CACHE.clear()

    def _setup_mock_platform(self, mock_platform, world_size=8):
        """Configure mock platform for mesh creation."""
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.split_group.return_value = MagicMock()
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.asnumpy() if hasattr(t, "asnumpy") else np.array(t)
        )

    def _make_1d_mesh(self, mock_platform, size=8, name="dp"):
        """Return a 1-D mesh of the given size."""
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(device_type="npu", mesh_shape=(size,), mesh_dim_names=(name,))

    def _make_2d_mesh(self, mock_platform, shape=(2, 2), names=("dp", "cp_tp")):
        """Return a 2-D mesh; cp_tp is mesh dim index 1 by default."""
        self._setup_mock_platform(mock_platform, world_size=shape[0] * shape[1])
        return init_device_mesh(device_type="npu", mesh_shape=shape, mesh_dim_names=names)

    def _make_3d_mesh(self, mock_platform, shape=(2, 2, 2), names=("dp", "cp", "tp")):
        """Return a 3-D mesh."""
        world_size = shape[0] * shape[1] * shape[2]
        self._setup_mock_platform(mock_platform, world_size=world_size)
        return init_device_mesh(device_type="npu", mesh_shape=shape, mesh_dim_names=names)

    @staticmethod
    def _get_op():
        return get_distributed_op("npu_mhc_pre_sinkhorn")

    # ------------------------------------------------------------------
    # _normalize_mhc_pre_sinkhorn_args
    # ------------------------------------------------------------------

    def test_normalize_args_positional_defaults_1(self):
        """
        Feature: _normalize_mhc_pre_sinkhorn_args fills in default values.
        Description: Call with only the 4 mandatory positional args.
        Expectation: hc_mult=4, num_iters=20; kwargs empty.
        """
        x, phi, alpha, bias = object(), object(), object(), object()
        args, kwargs = _normalize_mhc_pre_sinkhorn_args(x, phi, alpha, bias)
        self.assertIs(args[0], x)
        self.assertIs(args[1], phi)
        self.assertIs(args[2], alpha)
        self.assertIs(args[3], bias)
        self.assertEqual(args[4], 4, msg=f"hc_mult default should be 4, got {args[4]}")
        self.assertEqual(args[5], 20, msg=f"num_iters default should be 20, got {args[5]}")
        self.assertEqual(kwargs, {})

    def test_normalize_args_custom_params_2(self):
        """
        Feature: _normalize_mhc_pre_sinkhorn_args accepts custom parameters.
        Description: Override hc_mult and num_iters.
        Expectation: Custom values land at the expected positions.
        """
        x, phi, alpha, bias = object(), object(), object(), object()
        args, _ = _normalize_mhc_pre_sinkhorn_args(x, phi, alpha, bias, hc_mult=8, num_iters=10)
        self.assertEqual(args[4], 8, msg=f"hc_mult should be 8, got {args[4]}")
        self.assertEqual(args[5], 10, msg=f"num_iters should be 10, got {args[5]}")

    # ------------------------------------------------------------------
    # YAML registration
    # ------------------------------------------------------------------

    def test_yaml_registration_3(self):
        """
        Feature: YAML registry resolves npu_mhc_pre_sinkhorn to a DistributedOp.
        Description: Call get_distributed_op with the op name.
        Expectation: Returns a non-None DistributedOp instance.
        """
        op = self._get_op()
        self.assertIsNotNone(op)

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.shard.ops.parallel_mhc_pre_sinkhorn.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_ms_path_4(self, mock_device_platform, mock_op_platform):
        """
        Feature: preprocess builds 9 positional args on MindSpore path.
        Description: All 4 tensor inputs are DTensor mocks; platform is MINDSPORE.
        Expectation: local_args has 9 elements; local_kwargs is empty.
        """
        mock_device_platform.get_rank.return_value = 0
        mock_device_platform.get_world_size.return_value = 8
        mock_device_platform.split_group.return_value = MagicMock()
        mock_device_platform.tensor_to_numpy.side_effect = (
            lambda t: t.asnumpy() if hasattr(t, "asnumpy") else np.array(t)
        )
        mock_op_platform.platform_type = PlatformType.MINDSPORE

        def _mk():
            dt = MagicMock(spec=DTensor)
            dt.to_local.return_value = MagicMock()
            dt.layout = MagicMock()
            return dt

        dtensor_x, dtensor_phi, dtensor_alpha, dtensor_bias = _mk(), _mk(), _mk(), _mk()

        op = self._get_op()
        local_args, local_kwargs, cache_values = op.preprocess(
            (dtensor_x, dtensor_phi, dtensor_alpha, dtensor_bias,
             4, 20, 1e-6, 1e-6, True),
            {}
        )

        assert len(local_args) == 9, (
            f"MindSpore path: expected 9 local_args, got {len(local_args)}"
        )
        assert local_kwargs == {}, f"Expected empty kwargs, got {local_kwargs}"

    @patch("hyper_parallel.core.shard.ops.parallel_mhc_pre_sinkhorn.platform")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_preprocess_torch_path_5(self, mock_device_platform, mock_op_platform):
        """
        Feature: preprocess builds 4 positional + kwargs on PyTorch path.
        Description: All 4 tensor inputs are DTensor mocks; platform is TORCH.
        Expectation: local_args has 4 elements; local_kwargs has 5 scalar entries.
        """
        mock_device_platform.get_rank.return_value = 0
        mock_device_platform.get_world_size.return_value = 8
        mock_device_platform.split_group.return_value = MagicMock()
        mock_device_platform.tensor_to_numpy.side_effect = (
            lambda t: t.asnumpy() if hasattr(t, "asnumpy") else np.array(t)
        )
        mock_op_platform.platform_type = PlatformType.PYTORCH

        def _mk():
            dt = MagicMock(spec=DTensor)
            dt.to_local.return_value = MagicMock()
            dt.layout = MagicMock()
            return dt

        dtensor_x, dtensor_phi, dtensor_alpha, dtensor_bias = _mk(), _mk(), _mk(), _mk()

        op = self._get_op()
        local_args, local_kwargs, cache_values = op.preprocess(
            (dtensor_x, dtensor_phi, dtensor_alpha, dtensor_bias,
             4, 20, 1e-6, 1e-6, True),
            {}
        )

        assert len(local_args) == 4, (
            f"Torch path: expected 4 local_args, got {len(local_args)}"
        )
        assert "hc_mult" in local_kwargs, f"hc_mult missing from kwargs: {local_kwargs}"
        assert "num_iters" in local_kwargs, f"num_iters missing from kwargs: {local_kwargs}"

    # ------------------------------------------------------------------
    # infer_layout — success cases
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_all_replicated_6(self, mock_platform):
        """
        Feature: infer_layout with fully replicated inputs.
        Description: All 4 inputs replicated on 1-D mesh; 8 outputs expected.
        Expectation: Returns 8 output layouts; tensor_map all -1; extra is None.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Replicate(),), 4)
        phi_layout = _build_layout(mesh, (Replicate(),), 2)
        alpha_layout = _build_layout(mesh, (Replicate(),), 1)
        bias_layout = _build_layout(mesh, (Replicate(),), 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, phi_layout, alpha_layout, bias_layout]
        )

        assert len(out_layouts) == 8, (
            f"npu_mhc_pre_sinkhorn emits 8 outputs; expected 8 layouts, got {len(out_layouts)}"
        )
        assert extra is None, f"Expected extra=None, got {extra}"
        # get_expand_impl is not overridden — verify once here.
        assert op.get_expand_impl(None, (out_layouts, extra), [x_layout]) is None, (
            f"get_expand_impl should return None, "
            f"got {op.get_expand_impl(None, (out_layouts, extra), [x_layout])}"
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_data_parallel_7(self, mock_platform):
        """
        Feature: infer_layout with B-dim sharded (data parallel).
        Description: x sharded on dim 0 (batch); N replicated.  Outputs
            preserve B/S/C sharding according to their own rank.
        Expectation: 3-D outputs carry (b_map, s_map, …); 4-D/5-D outputs
            prefix the iteration axis as replicated.
        """
        mesh = self._make_1d_mesh(mock_platform, size=4, name="dp")

        x_layout = _build_layout(mesh, (Shard(0),), 4)
        phi_layout = _build_layout(mesh, (Replicate(),), 2)
        alpha_layout = _build_layout(mesh, (Replicate(),), 1)
        bias_layout = _build_layout(mesh, (Replicate(),), 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, phi_layout, alpha_layout, bias_layout]
        )

        assert len(out_layouts) == 8, (
            f"Expected 8 output layouts, got {len(out_layouts)}"
        )
        # x_tm = (0, -1, -1, -1)  →  b_map=0, c_map=-1
        expected = [
            (0, -1, -1),          # 0: h_in           (B, S, C)
            (0, -1, -1),          # 1: h_post         (B, S, N)
            (0, -1, -1),          # 2: h_res          (B, S, N*N)
            (0, -1, -1),          # 3: h_pre          (B, S, N)
            (0, -1, -1),          # 4: hc_before_norm (B, S, fusion)
            (0, -1, -1),          # 5: inv_rms        (B, S, 1)
            (-1, 0, -1, -1),      # 6: sum_out        (I, B, S, N)
            (-1, 0, -1, -1, -1),  # 7: norm_out       (I, B, S, N, N)
        ]
        for i, exp_tm in enumerate(expected):
            actual_tm = out_layouts[i].tensor_map
            assert actual_tm == exp_tm, (
                f"Output {i}: expected tensor_map {exp_tm}, got {actual_tm}"
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_dp_cp_tp_mixed_maps_13(self, mock_platform):
        """
        Feature: infer_layout with 3-D mesh (dp, cp, tp).
        Description: x placements (Shard(0), Shard(1), Shard(1));
            mesh (2,2,2).  Outputs must preserve B/S sharding and
            replicate kernel-internal axes.
        Expectation: 3-D outputs (B,S,…) use (b_map, s_map, -1);
            4-D sum_out uses (-1, b_map, s_map, -1);
            5-D norm_out uses (-1, b_map, s_map, -1, -1).
        """
        mesh = self._make_3d_mesh(mock_platform)
        replicated = (Replicate(), Replicate(), Replicate())

        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(1)), 4)
        phi_layout = _build_layout(mesh, replicated, 2)
        alpha_layout = _build_layout(mesh, replicated, 1)
        bias_layout = _build_layout(mesh, replicated, 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, phi_layout, alpha_layout, bias_layout]
        )

        assert len(out_layouts) == 8, (
            f"Expected 8 output layouts, got {len(out_layouts)}"
        )
        # Derive expected maps from the input tensor_map (reverse-indexed).
        b_map, s_map = x_layout.tensor_map[0], x_layout.tensor_map[1]
        tm_3d = (b_map, s_map, -1)
        tm_4d = (-1, b_map, s_map, -1)
        tm_5d = (-1, b_map, s_map, -1, -1)

        expected = [
            tm_3d,  # 0: h_in           (B, S, C)
            tm_3d,  # 1: h_post         (B, S, N)
            tm_3d,  # 2: h_res          (B, S, N*N)
            tm_3d,  # 3: h_pre          (B, S, N)
            tm_3d,  # 4: hc_before_norm (B, S, fusion)
            tm_3d,  # 5: inv_rms        (B, S, 1)
            tm_4d,  # 6: sum_out        (I, B, S, N)
            tm_5d,  # 7: norm_out       (I, B, S, N, N)
        ]
        for i, exp_tm in enumerate(expected):
            actual_tm = out_layouts[i].tensor_map
            assert actual_tm == exp_tm, (
                f"DP+CP+TP output {i}: expected {exp_tm}, got {actual_tm}"
            )
        assert extra is None, f"Expected extra=None, got {extra}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_partial_input_raises_8(self, mock_platform):
        """
        Feature: infer_layout rejects inputs in Partial state.
        Description: x_layout.is_partial() returns True.
        Expectation: ValueError is raised.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Replicate(),), 4)
        x_layout.is_partial = MagicMock(return_value=True)

        op = self._get_op()
        with self.assertRaises(ValueError):
            op.infer_layout([x_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_sharded_n_dim_raises_9(self, mock_platform):
        """
        Feature: infer_layout rejects N-dimension sharding for BSND format.
        Description: x sharded on dim 2 (N); N must be replicated per spec.
        Expectation: ValueError mentioning 'N dimension'.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Shard(2),), 4)
        phi_layout = _build_layout(mesh, (Replicate(),), 2)
        alpha_layout = _build_layout(mesh, (Replicate(),), 1)
        bias_layout = _build_layout(mesh, (Replicate(),), 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N dimension"):
            op.infer_layout([x_layout, phi_layout, alpha_layout, bias_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_replicated_11(self, mock_platform):
        """
        Feature: infer_layout with TND format (3D x), all replicated.
        Description: x is [T,N,C] (3D tensor_map); phi/alpha/bias Replicate.
        Expectation: Returns 8 output layouts with per-output ranks;
            every entry is -1 because no mesh axis is used.
        """
        mesh = self._make_1d_mesh(mock_platform, size=8)

        x_layout = _build_layout(mesh, (Replicate(),), 3)
        phi_layout = _build_layout(mesh, (Replicate(),), 2)
        alpha_layout = _build_layout(mesh, (Replicate(),), 1)
        bias_layout = _build_layout(mesh, (Replicate(),), 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        out_layouts, extra = op.infer_layout(
            [x_layout, phi_layout, alpha_layout, bias_layout]
        )

        assert len(out_layouts) == 8, (
            f"TND replicated: expected 8 output layouts, got {len(out_layouts)}"
        )
        # x_tm = (-1, -1, -1)  →  t_map=-1, c_map=-1
        expected = [
            (-1, -1),              # 0: h_in           (T, C)       — 2-D
            (-1, -1),              # 1: h_post         (T, N)       — 2-D
            (-1, -1),              # 2: h_res          (T, N*N)     — 2-D
            (-1, -1),              # 3: h_pre          (T, N)       — 2-D
            (-1, -1),              # 4: hc_before_norm (T, fusion)  — 2-D
            (-1, -1),              # 5: inv_rms        (T, 1)       — 2-D
            (-1, -1, -1),          # 6: sum_out        (I, T, N)    — 3-D
            (-1, -1, -1, -1),      # 7: norm_out       (I, T, N, N) — 4-D
        ]
        for i, exp_tm in enumerate(expected):
            actual_tm = out_layouts[i].tensor_map
            assert actual_tm == exp_tm, (
                f"TND output {i}: expected tensor_map {exp_tm}, got {actual_tm}"
            )
        assert extra is None, f"Expected extra=None, got {extra}"

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_n_axis_raises_12(self, mock_platform):
        """
        Feature: infer_layout rejects N-axis sharding on TND format.
        Description: x [T,N,C] sharded on N (dim 1); N must be replicated.
        Expectation: ValueError mentioning 'N dimension'.
        """
        mesh = self._make_2d_mesh(mock_platform)

        x_layout = _build_layout(mesh, (Replicate(), Shard(1)), 3)
        phi_layout = _build_layout(mesh, (Replicate(),), 2)
        alpha_layout = _build_layout(mesh, (Replicate(),), 1)
        bias_layout = _build_layout(mesh, (Replicate(),), 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "N dimension"):
            op.infer_layout([x_layout, phi_layout, alpha_layout, bias_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_bsnd_c_axis_raises_15(self, mock_platform):
        """Test that BSND input cannot shard the C dimension."""
        mesh = self._make_1d_mesh(mock_platform, size=2, name="tp")

        x_layout = _build_layout(mesh, (Shard(3),), 4)
        phi_layout = _build_layout(mesh, (Replicate(),), 2)
        alpha_layout = _build_layout(mesh, (Replicate(),), 1)
        bias_layout = _build_layout(mesh, (Replicate(),), 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "C dimension"):
            op.infer_layout([x_layout, phi_layout, alpha_layout, bias_layout])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_infer_layout_tnd_c_axis_raises_16(self, mock_platform):
        """Test that TND input cannot shard the C dimension."""
        mesh = self._make_1d_mesh(mock_platform, size=2, name="tp")

        x_layout = _build_layout(mesh, (Shard(2),), 3)
        phi_layout = _build_layout(mesh, (Replicate(),), 2)
        alpha_layout = _build_layout(mesh, (Replicate(),), 1)
        bias_layout = _build_layout(mesh, (Replicate(),), 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        op = self._get_op()
        with self.assertRaisesRegex(ValueError, "C dimension"):
            op.infer_layout([x_layout, phi_layout, alpha_layout, bias_layout])


    # ------------------------------------------------------------------
    # clamp variant
    # ------------------------------------------------------------------

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_clamp_infer_layout_9_outputs_14(self, mock_platform):
        """
        Feature: clamp variant infer_layout returns 9 output layouts.
        Description: 3-D mesh (dp, cp, tp); the 9th output h_res_logits
            shares norm_out's tensor_map.
        Expectation: 9 layouts; last one is 5-D with iter-axis replicated.
        """
        mesh = self._make_3d_mesh(mock_platform)
        replicated = (Replicate(), Replicate(), Replicate())

        x_layout = _build_layout(mesh, (Shard(0), Shard(1), Shard(1)), 4)
        phi_layout = _build_layout(mesh, replicated, 2)
        alpha_layout = _build_layout(mesh, replicated, 1)
        bias_layout = _build_layout(mesh, replicated, 1)

        for layout in (x_layout, phi_layout, alpha_layout, bias_layout):
            layout.is_partial = MagicMock(return_value=False)

        clamp_op = NpuMhcPreClampSinkhornDistributedOp("npu_mhc_pre_clamp_sinkhorn")
        out_layouts, extra = clamp_op.infer_layout(
            [x_layout, phi_layout, alpha_layout, bias_layout]
        )

        assert len(out_layouts) == 9, (
            f"Expected 9 layouts, got {len(out_layouts)}"
        )
        # Derive expected maps from the input tensor_map (reverse-indexed).
        b_map, s_map = x_layout.tensor_map[0], x_layout.tensor_map[1]
        tm_3d = (b_map, s_map, -1)
        tm_4d = (-1, b_map, s_map, -1)
        tm_5d = (-1, b_map, s_map, -1, -1)
        # First 8 match the non-clamp variant.
        assert out_layouts[0].tensor_map == tm_3d, (
            f"h_in: expected {tm_3d}, got {out_layouts[0].tensor_map}"
        )
        assert out_layouts[6].tensor_map == tm_4d, (
            f"sum_out: expected {tm_4d}, got {out_layouts[6].tensor_map}"
        )
        assert out_layouts[7].tensor_map == tm_5d, (
            f"norm_out: expected {tm_5d}, got {out_layouts[7].tensor_map}"
        )
        # 9th output: h_res_logits — same as norm_out
        assert out_layouts[8].tensor_map == tm_5d, (
            f"h_res_logits: expected {tm_5d}, got {out_layouts[8].tensor_map}"
        )
        assert extra is None, f"Expected extra=None, got {extra}"


if __name__ == "__main__":
    unittest.main()
