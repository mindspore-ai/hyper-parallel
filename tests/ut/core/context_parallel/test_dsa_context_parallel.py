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
    AsyncDSAIndexerContextParallel,
    AsyncDSAIndexerLossContextParallel,
    AsyncDSASparseAttentionContextParallel,
    DSAIndexerContextParallel,
    DSAIndexerLossContextParallel,
    DSASparseAttentionContextParallel,
)
from hyper_parallel.core.context_parallel.context_parallel import (
    ContextParallel,
    _OUTPUT_LAYOUT_STACK_ATTR,
    _drop_cp_from_output,
    _non_cp_dtensor_layout,
    _to_cp_dtensor,
)
from hyper_parallel.core.context_parallel.async_dsa_context_parallel import _AsyncSequenceReplicateSlot
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


def _patch_torch_dist_rank(world_size=1):
    """Patch rank helpers for DTensor redistribution in CPU-only UT."""
    return patch.multiple(
        "torch.distributed",
        get_rank=MagicMock(return_value=0),
        get_world_size=MagicMock(return_value=world_size),
    )


class _IdentityModule(nn.Module):
    """Leaf module that returns its arguments unchanged."""

    def forward(self, *args, **kwargs):
        if kwargs:
            return args, kwargs
        return args


class _SingleIdentityModule(nn.Module):
    """Single-input handoff module."""

    def forward(self, value):
        return value


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

    def _make_cp_tp_meshes(self, mock_platform):
        self._setup_mock_platform(mock_platform, world_size=1)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(1, 1),
            mesh_dim_names=("cp", "tp"),
            init_backend=False,
        )
        return root["cp"], root["tp"]

    def _make_tp_cp_meshes(self, mock_platform):
        self._setup_mock_platform(mock_platform, world_size=1)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(1, 1),
            mesh_dim_names=("tp", "cp"),
            init_backend=False,
        )
        return root["cp"], root["tp"]

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_non_cp_dtensor_enters_cp_on_composed_mesh_and_drops_cp_on_exit(self, mock_mesh_platform):
        """TP DTensor inputs should run inside CP+TP and leave CP with TP preserved."""
        cp_mesh, tp_mesh = self._make_cp_tp_meshes(mock_mesh_platform)
        tp_local = torch.randn(2, 4, 8, 16)
        tp_dtensor = DTensor.from_local(tp_local, tp_mesh, (Shard(2),))

        with _patch_torch_dist_rank():
            cp_tp_dtensor = _to_cp_dtensor(
                tp_dtensor,
                cp_mesh,
                (Shard(1),),
                (Shard(1),),
                seq_dim=1,
            )

        self.assertEqual(cp_tp_dtensor.device_mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(cp_tp_dtensor.placements, (Shard(1), Shard(2)))

        layout = _non_cp_dtensor_layout(cp_tp_dtensor, cp_mesh, seq_dim=1)
        with _patch_torch_dist_rank():
            output = _drop_cp_from_output(cp_tp_dtensor, layout, (Shard(1),))

        self.assertIsInstance(output, DTensor)
        self.assertEqual(output.device_mesh.mesh_dim_names, ("tp",))
        self.assertEqual(output.placements, (Shard(2),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cp_head_shard_uses_strided_shard_when_tp_already_shards_head(self, mock_mesh_platform):
        """CP Shard(seq)->Shard(head) should preserve right-to-left head split order with TP."""
        self._setup_mock_platform(mock_mesh_platform, world_size=2)
        root = init_device_mesh(
            device_type="cpu",
            mesh_shape=(1, 2),
            mesh_dim_names=("cp", "tp"),
            init_backend=False,
        )
        cp_mesh = root["cp"]
        tp_mesh = root["tp"]
        tp_local = torch.randn(2, 4, 4, 16)
        tp_dtensor = DTensor.from_local(tp_local, tp_mesh, (Shard(2),))

        with _patch_torch_dist_rank(world_size=2):
            cp_tp_dtensor = _to_cp_dtensor(
                tp_dtensor,
                cp_mesh,
                (Shard(1),),
                (Shard(2),),
                seq_dim=1,
            )

        self.assertEqual(cp_tp_dtensor.device_mesh.mesh_dim_names, ("cp", "tp"))
        self.assertEqual(cp_tp_dtensor.placements, (StridedShard(2, 2), Shard(2)))
        self.assertEqual(cp_tp_dtensor._layout.tensor_map, (-1, -1, (0, 1), -1))

        layout = _non_cp_dtensor_layout(cp_tp_dtensor, cp_mesh, seq_dim=1)
        with _patch_torch_dist_rank(world_size=2):
            output = _drop_cp_from_output(cp_tp_dtensor, layout, (Shard(2),))

        self.assertIsInstance(output, DTensor)
        self.assertEqual(output.device_mesh.mesh_dim_names, ("tp",))
        self.assertEqual(output.placements, (Shard(2),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_context_parallel_output_policy_follows_q_input_boundary(self, mock_mesh_platform):
        """With DTensor output enabled, CP mirrors local/CP/TP input boundaries."""
        cp_mesh, tp_mesh = self._make_cp_tp_meshes(mock_mesh_platform)
        module = _SingleIdentityModule()
        style = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1, use_local_output=False)
        local = torch.randn(2, 4, 8, 16)

        with _patch_torch_dist_rank():
            cp_internal = DTensor.from_local(local, cp_mesh, (Shard(1),))

            style._record_q_output_tp_layout(module, [local], {}, cp_mesh)
            local_output = style._post_hook_colossal(module, (), cp_internal, cp_mesh)
            self.assertNotIsInstance(local_output, DTensor)
            self.assertFalse(hasattr(module, _OUTPUT_LAYOUT_STACK_ATTR))

            cp_input = DTensor.from_local(local, cp_mesh, (Shard(1),))
            style._record_q_output_tp_layout(module, [cp_input], {}, cp_mesh)
            cp_output = style._post_hook_colossal(module, (), cp_internal, cp_mesh)
            self.assertIsInstance(cp_output, DTensor)
            self.assertEqual(cp_output.device_mesh.mesh_dim_names, ("cp",))
            self.assertEqual(cp_output.placements, (Shard(1),))

            tp_input = DTensor.from_local(local, tp_mesh, (Shard(2),))
            composed_mesh = _non_cp_dtensor_layout(tp_input, cp_mesh, seq_dim=1)[2]
            cp_tp_internal = DTensor.from_local(local, composed_mesh, (Shard(1), Shard(2)))
            style._record_q_output_tp_layout(module, [tp_input], {}, cp_mesh)
            tp_output = style._post_hook_colossal(module, (), cp_tp_internal, cp_mesh)
            self.assertIsInstance(tp_output, DTensor)
            self.assertEqual(tp_output.device_mesh.mesh_dim_names, ("tp",))
            self.assertEqual(tp_output.placements, (Shard(2),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_output_layout_record_uses_stack_for_reentrant_forward(self, mock_mesh_platform):
        """Nested CP forwards should pop the most recent Q layout first."""
        cp_mesh = self._make_cp_mesh(mock_mesh_platform)
        module = _SingleIdentityModule()
        style = ContextParallel(seq_dim=1, head_dim=2, ulysses_degree=1, use_local_output=False)
        local = torch.randn(2, 4, 8, 16)

        with _patch_torch_dist_rank():
            cp_input = DTensor.from_local(local, cp_mesh, (Shard(1),))
            cp_internal = DTensor.from_local(local, cp_mesh, (Shard(1),))

            style._record_q_output_tp_layout(module, [local], {}, cp_mesh)
            style._record_q_output_tp_layout(module, [cp_input], {}, cp_mesh)
            inner_output = style._post_hook_colossal(module, (), cp_internal, cp_mesh)
            outer_output = style._post_hook_colossal(module, (), cp_internal, cp_mesh)

        self.assertIsInstance(inner_output, DTensor)
        self.assertNotIsInstance(outer_output, DTensor)
        self.assertFalse(hasattr(module, _OUTPUT_LAYOUT_STACK_ATTR))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_composed_mesh_follows_root_order_when_cp_is_after_tp(self, mock_mesh_platform):
        """TP+CP composition should preserve placements when root order is TP,CP."""
        cp_mesh, tp_mesh = self._make_tp_cp_meshes(mock_mesh_platform)
        tp_local = torch.randn(2, 4, 8, 16)
        tp_dtensor = DTensor.from_local(tp_local, tp_mesh, (Shard(2),))

        with _patch_torch_dist_rank():
            tp_cp_dtensor = _to_cp_dtensor(
                tp_dtensor,
                cp_mesh,
                (Shard(1),),
                (Shard(1),),
                seq_dim=1,
            )

        self.assertEqual(tp_cp_dtensor.device_mesh.mesh_dim_names, ("tp", "cp"))
        self.assertEqual(tp_cp_dtensor.placements, (Shard(2), Shard(1)))

        layout = _non_cp_dtensor_layout(tp_cp_dtensor, cp_mesh, seq_dim=1)
        with _patch_torch_dist_rank():
            output = _drop_cp_from_output(tp_cp_dtensor, layout, (Shard(1),))

        self.assertIsInstance(output, DTensor)
        self.assertEqual(output.device_mesh.mesh_dim_names, ("tp",))
        self.assertEqual(output.placements, (Shard(2),))

    def test_rejects_non_colossal_mode(self):
        """Only Colossal-style CP is supported in the first implementation."""
        with self.assertRaises(ValueError):
            DSASparseAttentionContextParallel(mode="ulysses")

    def test_repr_includes_key_config(self):
        """repr should expose user-visible configuration."""
        style = DSASparseAttentionContextParallel(layout="TND", use_local_output=False)
        text = repr(style)
        self.assertIn("DSASparseAttentionContextParallel", text)
        self.assertIn("layout='TND'", text)
        self.assertIn("use_local_output=False", text)

    def test_async_dsa_styles_are_public(self):
        """Async DSA CP exposes CP-level styles without raw collective handles."""
        self.assertIsInstance(AsyncDSAIndexerContextParallel(), DSAIndexerContextParallel)
        self.assertIsInstance(AsyncDSAIndexerLossContextParallel(), DSAIndexerLossContextParallel)
        self.assertIsInstance(AsyncDSASparseAttentionContextParallel(), DSASparseAttentionContextParallel)

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
    def test_indexer_boundary_kwargs_are_rewritten(self, mock_mesh_platform):
        """Indexer boundary should also rewrite configured keyword arguments."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSAIndexerContextParallel(
            layout="BSND",
            query_index=None,
            key_index=None,
            weights_index=None,
            query_kwarg_name="query",
            key_kwarg_name="key",
            weights_kwarg_name="weights",
            use_local_output=False,
        )
        module = _IdentityModule()
        style.apply(module, mesh)

        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 1, 16)
        w = torch.randn(2, 4, 8)

        with _patch_torch_dist_rank():
            out_args, out_kwargs = module(query=q, key=k, weights=w, keep="unchanged")
        self.assertEqual(out_args, ())
        self.assertEqual(out_kwargs["query"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["key"].placements, (Replicate(),))
        self.assertEqual(out_kwargs["weights"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["keep"], "unchanged")

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
    def test_sparse_attention_boundary_kwargs_are_rewritten(self, mock_mesh_platform):
        """Sparse FA boundary should also rewrite configured keyword arguments."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSASparseAttentionContextParallel(
            layout="BSND",
            query_index=None,
            key_index=None,
            value_index=None,
            topk_index=None,
            query_rope_index=None,
            key_rope_index=None,
            query_kwarg_name="query",
            key_kwarg_name="key",
            value_kwarg_name="value",
            topk_kwarg_name="topk",
            query_rope_kwarg_name="query_rope",
            key_rope_kwarg_name="key_rope",
            use_local_output=False,
        )
        module = _IdentityModule()
        style.apply(module, mesh)

        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 1, 16)
        v = torch.randn(2, 4, 1, 16)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)
        q_rope = torch.randn(2, 4, 8, 8)
        k_rope = torch.randn(2, 4, 1, 8)

        with _patch_torch_dist_rank():
            out_args, out_kwargs = module(
                query=q,
                key=k,
                value=v,
                topk=topk,
                query_rope=q_rope,
                key_rope=k_rope,
            )
        self.assertEqual(out_args, ())
        self.assertEqual(out_kwargs["query"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["key"].placements, (Replicate(),))
        self.assertEqual(out_kwargs["value"].placements, (Replicate(),))
        self.assertEqual(out_kwargs["topk"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["query_rope"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["key_rope"].placements, (Replicate(),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_async_sparse_attention_boundary_inputs_are_rewritten(self, mock_mesh_platform):
        """Async sparse FA boundary preserves the same public DSA CP placements."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = AsyncDSASparseAttentionContextParallel(layout="BSND", use_local_output=False)
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
    def test_async_indexer_waits_prelaunched_key_handoff(self, mock_mesh_platform):
        """Async indexer CP should consume producer-launched key-side tensor."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = AsyncDSAIndexerContextParallel(layout="BSND", use_local_output=False)
        module = _IdentityModule()
        key_handoff = _SingleIdentityModule()
        style.apply(module, mesh, key_handoff=key_handoff)

        q = torch.randn(2, 4, 8, 16)
        producer_key = torch.randn(2, 4, 1, 16)
        stale_key = torch.zeros_like(producer_key)
        w = torch.randn(2, 4, 8)

        with _patch_torch_dist_rank():
            _ = key_handoff(producer_key)
            out = module(q, stale_key, w)
        self.assertEqual(out[1].placements, (Replicate(),))
        self.assertTrue(torch.equal(out[1].to_local(), producer_key))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_async_sparse_attention_waits_prelaunched_handoffs(self, mock_mesh_platform):
        """Async sparse FA CP should wait producer hooks for key/value/key_rope."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = AsyncDSASparseAttentionContextParallel(layout="BSND", use_local_output=False)
        module = _IdentityModule()
        key_handoff = _SingleIdentityModule()
        value_handoff = _SingleIdentityModule()
        key_rope_handoff = _SingleIdentityModule()
        style.apply(
            module,
            mesh,
            key_handoff=key_handoff,
            value_handoff=value_handoff,
            key_rope_handoff=key_rope_handoff,
        )

        q = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 1, 16)
        value = torch.randn(2, 4, 1, 16)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)
        q_rope = torch.randn(2, 4, 8, 8)
        key_rope = torch.randn(2, 4, 1, 8)

        with _patch_torch_dist_rank():
            _ = key_handoff(key)
            _ = value_handoff(value)
            _ = key_rope_handoff(key_rope)
            out = module(q, torch.zeros_like(key), torch.zeros_like(value), topk, q_rope, torch.zeros_like(key_rope))
        self.assertTrue(torch.equal(out[1].to_local(), key))
        self.assertTrue(torch.equal(out[2].to_local(), value))
        self.assertTrue(torch.equal(out[5].to_local(), key_rope))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_async_slot_waits_backward_handle_at_producer_boundary(self, mock_mesh_platform):
        """Async DSA should defer all-gather backward wait to the producer boundary."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        slot = _AsyncSequenceReplicateSlot(mesh, seq_dim=1)
        expected_grad = torch.randn(2, 4, 8, 16)
        out_perm = expected_grad.permute(1, 0, 2, 3).contiguous()
        work = MagicMock()
        grad_output = (torch.zeros_like(expected_grad),)

        result = slot._producer_bwd_pre_hook(grad_output, [(work, out_perm, 1)])

        work.wait.assert_called_once()
        self.assertTrue(torch.equal(result[0], expected_grad))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_async_indexer_loss_waits_prelaunched_handoffs(self, mock_mesh_platform):
        """Async indexer-loss CP should wait key/key-index/key-rope producer hooks."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = AsyncDSAIndexerLossContextParallel(layout="BSND", use_local_output=False)
        module = _IdentityModule()
        key_handoff = _SingleIdentityModule()
        key_indexer_handoff = _SingleIdentityModule()
        key_rope_handoff = _SingleIdentityModule()

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 1, 16)
        query_index = torch.randn(2, 4, 8, 16)
        key_index = torch.randn(2, 4, 1, 16)
        weights = torch.randn(2, 4, 8)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)
        softmax_max = torch.randn(2, 4, 8, 1)
        softmax_sum = torch.randn(2, 4, 8, 1)
        query_rope = torch.randn(2, 4, 8, 8)
        key_rope = torch.randn(2, 4, 1, 8)

        with _patch_torch_dist_rank():
            style.apply(
                module,
                mesh,
                key_handoff=key_handoff,
                key_indexer_handoff=key_indexer_handoff,
                key_rope_handoff=key_rope_handoff,
            )
            _ = key_handoff(key)
            _ = key_indexer_handoff(key_index)
            _ = key_rope_handoff(key_rope)
            out = module(
                query, torch.zeros_like(key), query_index, torch.zeros_like(key_index),
                weights, topk, softmax_max, softmax_sum, query_rope, torch.zeros_like(key_rope)
            )
        self.assertTrue(torch.equal(out[1].to_local(), key))
        self.assertTrue(torch.equal(out[3].to_local(), key_index))
        self.assertTrue(torch.equal(out[9].to_local(), key_rope))

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
    def test_indexer_loss_boundary_kwargs_are_rewritten(self, mock_mesh_platform):
        """Indexer-loss boundary should also rewrite configured keyword arguments."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSAIndexerLossContextParallel(
            layout="BSND",
            query_index=None,
            key_index=None,
            query_indexer_index=None,
            key_indexer_index=None,
            weights_index=None,
            topk_index=None,
            query_rope_index=None,
            key_rope_index=None,
            query_kwarg_name="query",
            key_kwarg_name="key",
            query_indexer_kwarg_name="query_index",
            key_indexer_kwarg_name="key_index",
            weights_kwarg_name="weights",
            topk_kwarg_name="topk",
            query_rope_kwarg_name="query_rope",
            key_rope_kwarg_name="key_rope",
            use_local_output=False,
        )
        module = _IdentityModule()

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 1, 16)
        query_index = torch.randn(2, 4, 8, 16)
        key_index = torch.randn(2, 4, 1, 16)
        weights = torch.randn(2, 4, 8)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)
        q_rope = torch.randn(2, 4, 8, 8)
        k_rope = torch.randn(2, 4, 1, 8)

        with _patch_torch_dist_rank():
            style.apply(module, mesh)
            out_args, out_kwargs = module(
                query=query,
                key=key,
                query_index=query_index,
                key_index=key_index,
                weights=weights,
                topk=topk,
                query_rope=q_rope,
                key_rope=k_rope,
            )
        self.assertEqual(out_args, ())
        self.assertEqual(out_kwargs["query"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["key"].placements, (Replicate(),))
        self.assertEqual(out_kwargs["query_index"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["key_index"].placements, (Replicate(),))
        self.assertEqual(out_kwargs["weights"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["topk"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["query_rope"].placements, (Shard(1),))
        self.assertEqual(out_kwargs["key_rope"].placements, (Replicate(),))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_public_output_hook_returns_local_tensor(self, mock_mesh_platform):
        """Public output should default to local tensor conversion."""
        mesh = self._make_cp_mesh(mock_mesh_platform)
        style = DSASparseAttentionContextParallel(use_local_output=True)
        module = _IdentityModule()
        style.apply(module, mesh)

        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 4, 1, 16)
        value = torch.randn(2, 4, 1, 16)
        topk = torch.randint(0, 4, (2, 4, 1, 2), dtype=torch.int32)

        with _patch_torch_dist_rank():
            out = module(query, key, value, topk)
        self.assertIsInstance(out[0], torch.Tensor)
        self.assertNotIsInstance(out[0], DTensor)
