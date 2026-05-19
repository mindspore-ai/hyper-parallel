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
"""Unit tests for DSA context-parallel boundary hook styles."""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.context_parallel import (
    DSAContextParallel,
    DSAIndexerContextParallel,
    DSAIndexerLossContextParallel,
    DSASparseAttentionContextParallel,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


def _patch_torch_dist_rank(world_size=1):
    """Patch rank helpers for DTensor redistribution in CPU-only UT."""
    return patch.multiple(
        "torch.distributed",
        get_rank=MagicMock(return_value=0),
        get_world_size=MagicMock(return_value=world_size),
    )


class _PlainDsaAttentionBoundary(nn.Module):
    """Minimal attention-like module with ``(q, k, v, topk)`` boundary."""

    def forward(self, query, key, value, topk_indices, query_rope=None, key_rope=None):
        return query, key, value, topk_indices, query_rope, key_rope


class _IdentityModule(nn.Module):
    """Leaf module that returns its arguments unchanged."""

    def forward(self, *args, **kwargs):
        if kwargs:
            return args, kwargs
        return args


class TestDsaContextParallel(unittest.TestCase):
    """CPU-only tests for DSA CP hook wiring."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size=1):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_cp_mesh(self, mock_platform, size=1):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("cp",),
            init_backend=False,
        )

    def test_rejects_non_colossal_mode(self):
        """Only Colossal-style CP is supported in the first implementation."""
        with self.assertRaises(ValueError):
            DSAContextParallel(mode="ulysses")

    def test_repr_includes_key_config(self):
        """repr should expose user-visible configuration."""
        style = DSAContextParallel(layout="TND", use_local_output=False)
        text = repr(style)
        self.assertIn("DSAContextParallel", text)
        self.assertIn("layout='TND'", text)
        self.assertIn("use_local_output=False", text)

    def test_low_level_dsa_style_is_not_sparse_attention_subclass(self):
        """Low-level DSA boundary and sparse FA boundary are separate public styles."""
        self.assertNotIsInstance(DSAContextParallel(), DSASparseAttentionContextParallel)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_boundary_attention_inputs_are_rewritten(self, mock_mesh_platform):
        """Low-level boundary module should receive sharded q/topk and replicated k/v."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSAContextParallel(use_local_output=False)
        module = _PlainDsaAttentionBoundary()
        style.apply(module, mesh)

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 1, 16)
        value = torch.randn(2, 4, 1, 16)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)
        q_rope = torch.randn(2, 4, 8, 8)
        k_rope = torch.randn(2, 4, 1, 8)

        with _patch_torch_dist_rank():
            out = module(query, key, value, topk, q_rope, k_rope)
        self.assertIsInstance(out[0], DTensor)
        self.assertIsInstance(out[1], DTensor)
        self.assertIsInstance(out[2], DTensor)
        self.assertIsInstance(out[3], DTensor)
        self.assertIsInstance(out[4], DTensor)
        self.assertIsInstance(out[5], DTensor)
        self.assertEqual(out[0].placements, (Shard(1),))
        self.assertEqual(out[1].placements, (Replicate(),))
        self.assertEqual(out[2].placements, (Replicate(),))
        self.assertEqual(out[3].placements, (Shard(1),))
        self.assertEqual(out[4].placements, (Shard(1),))
        self.assertEqual(out[5].placements, (Replicate(),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_indexer_boundary_inputs_are_rewritten(self, mock_mesh_platform):
        """Indexer boundary should receive q/w sharded and k replicated."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSAIndexerContextParallel(layout="BSND", use_local_output=False)
        module = _IdentityModule()
        style.apply(module, mesh)

        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 1, 16)
        w = torch.randn(2, 4, 8)

        with _patch_torch_dist_rank():
            out = module(q, k, w)
        self.assertIsInstance(out[0], DTensor)
        self.assertIsInstance(out[1], DTensor)
        self.assertIsInstance(out[2], DTensor)
        self.assertEqual(out[0].placements, (Shard(1),))
        self.assertEqual(out[1].placements, (Replicate(),))
        self.assertEqual(out[2].placements, (Shard(1),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_sparse_attention_boundary_inputs_are_rewritten(self, mock_mesh_platform):
        """Sparse FA boundary should receive q/topk shard and k/v replicate."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSASparseAttentionContextParallel(layout="BSND", use_local_output=False)
        module = _IdentityModule()
        style.apply(module, mesh)

        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 1, 16)
        v = torch.randn(2, 4, 1, 16)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)
        q_rope = torch.randn(2, 4, 8, 8)
        k_rope = torch.randn(2, 4, 1, 8)

        with _patch_torch_dist_rank():
            out = module(q, k, v, topk, q_rope, k_rope)
        self.assertEqual(out[0].placements, (Shard(1),))
        self.assertEqual(out[1].placements, (Replicate(),))
        self.assertEqual(out[2].placements, (Replicate(),))
        self.assertEqual(out[3].placements, (Shard(1),))
        self.assertEqual(out[4].placements, (Shard(1),))
        self.assertEqual(out[5].placements, (Replicate(),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_indexer_loss_boundary_inputs_are_rewritten(self, mock_mesh_platform):
        """Indexer-loss boundary should shard query-side tensors and replicate key-side tensors."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSAIndexerLossContextParallel(layout="BSND", use_local_output=False)
        module = _IdentityModule()

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 1, 16)
        query_index = torch.randn(2, 4, 8, 16)
        key_index = torch.randn(2, 4, 1, 16)
        weights = torch.randn(2, 4, 8)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)
        softmax_max = torch.randn(2, 4, 8, 1)
        softmax_sum = torch.randn(2, 4, 8, 1)
        q_rope = torch.randn(2, 4, 8, 8)
        k_rope = torch.randn(2, 4, 1, 8)

        with _patch_torch_dist_rank():
            style.apply(module, mesh)
            out = module(
                query, key, query_index, key_index, weights,
                topk, softmax_max, softmax_sum, q_rope, k_rope
            )
        self.assertEqual(out[0].placements, (Shard(1),))
        self.assertEqual(out[1].placements, (Replicate(),))
        self.assertEqual(out[2].placements, (Shard(1),))
        self.assertEqual(out[3].placements, (Replicate(),))
        self.assertEqual(out[4].placements, (Shard(1),))
        self.assertEqual(out[5].placements, (Shard(1),))
        self.assertIs(out[6], softmax_max)
        self.assertIs(out[7], softmax_sum)
        self.assertEqual(out[8].placements, (Shard(1),))
        self.assertEqual(out[9].placements, (Replicate(),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_public_output_hook_returns_local_tensor(self, mock_mesh_platform):
        """Public output should default to local tensor conversion."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSAContextParallel(use_local_output=True)
        module = _PlainDsaAttentionBoundary()
        style.apply(module, mesh)

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 1, 16)
        value = torch.randn(2, 4, 1, 16)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)

        with _patch_torch_dist_rank():
            out = module(query, key, value, topk)
        self.assertIsInstance(out[0], torch.Tensor)
        self.assertNotIsInstance(out[0], DTensor)
