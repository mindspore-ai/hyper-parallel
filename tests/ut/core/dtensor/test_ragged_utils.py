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
"""Unit tests for RaggedShard geometry and its DTensor integration."""
import os
import unittest
from unittest.mock import patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor._ragged_utils import (
    _compute_ragged_all_to_all_splits,
    _compute_ragged_slice,
)
from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import (
    DTensor,
    _build_layout,
    distribute_tensor,
)
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import RaggedShard, Replicate, Shard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestRaggedDTensor(unittest.TestCase):
    """Verify phase-one RaggedShard construction without distributed hardware."""

    def setUp(self):
        """Use mesh rank zero and isolate global mesh caches."""
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()
        self.rank_patcher = patch(
            "hyper_parallel.core.dtensor.device_mesh.platform.get_rank",
            return_value=0,
        )
        self.rank_patcher.start()

    def tearDown(self):
        """Restore rank lookup and clear global mesh caches."""
        self.rank_patcher.stop()
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()

    @staticmethod
    def _mesh(mesh_shape=(2,), names=("ragged",)):
        """Create an uninitialized-backend mesh suitable for local slicing."""
        return Layout(mesh_shape, names, init_backend=False).mesh

    def test_distribute_tensor_matches_vescale_flat_storage(self):
        """Split ``(6, 4, 8)`` into rank-zero's 64-element flat shard."""
        mesh = self._mesh()
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))
        global_tensor = torch.arange(6 * 4 * 8).reshape(6, 4, 8)

        result = distribute_tensor(
            global_tensor,
            mesh,
            (ragged,),
            src_data_rank=None,
        )

        self.assertEqual(tuple(result.shape), (6, 4, 8))
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.numel(), 192)
        self.assertEqual(tuple(result.local_shape), (64,))
        self.assertTrue(torch.equal(result.to_local(), torch.arange(64)))

    def test_ragged_slice_geometry_covers_global_storage(self):
        """Produce the same two flat intervals observed in the veScale smoke test."""
        mesh = self._mesh()
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))
        layout = _build_layout(mesh, (ragged,), tensor_dim=3)

        rank_zero = _compute_ragged_slice((6, 4, 8), layout, local_rank=0)
        rank_one = _compute_ragged_slice((6, 4, 8), layout, local_rank=1)

        self.assertEqual((rank_zero.flat_start, rank_zero.flat_end), (0, 64))
        self.assertEqual((rank_one.flat_start, rank_one.flat_end), (64, 192))

    def test_from_local_requires_and_preserves_global_shape(self):
        """Require explicit shape and preserve it through detach and dtype conversion."""
        mesh = self._mesh()
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))
        local = torch.arange(64)

        with self.assertRaisesRegex(ValueError, "requires an explicit global shape"):
            DTensor.from_local(local, mesh, (ragged,))

        result = DTensor.from_local(local, mesh, (ragged,), shape=(6, 4, 8))
        detached = result.detach()
        converted = result.to(dtype=torch.float32)

        self.assertEqual(tuple(detached.shape), (6, 4, 8))
        self.assertEqual(tuple(detached.local_shape), (64,))
        self.assertEqual(tuple(converted.shape), (6, 4, 8))
        self.assertEqual(tuple(converted.local_shape), (64,))

    def test_empty_placeholder_uses_one_dimensional_replicated_layout(self):
        """Accept NPU FA's empty auxiliary output with a one-dimensional map."""
        mesh = self._mesh()
        layout = Layout.from_device_mesh(mesh)
        layout.set_placements((Replicate(),))
        layout.set_tensor_map((-1,))

        result = DTensor.from_local_with_layout(torch.empty(0), layout)

        self.assertEqual(tuple(result.local_shape), (0,))
        self.assertEqual(tuple(result.shape), (0,))

    def test_from_local_rejects_non_flat_or_wrong_size_storage(self):
        """Reject local tensors that violate the phase-one flat storage contract."""
        mesh = self._mesh()
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))

        with self.assertRaisesRegex(ValueError, "one-dimensional flat storage"):
            DTensor.from_local(torch.ones(8, 8), mesh, (ragged,), shape=(6, 4, 8))
        with self.assertRaisesRegex(ValueError, "numel does not match"):
            DTensor.from_local(torch.ones(63), mesh, (ragged,), shape=(6, 4, 8))

    def test_phase_one_rejects_other_non_replicate_placements(self):
        """Reject ordinary Shard composition until its ordering semantics are implemented."""
        mesh = self._mesh((2, 2), ("ragged", "tp"))
        placements = (
            RaggedShard(dims=(0,), local_units=(1, 1)),
            Shard(1),
        )

        with self.assertRaisesRegex(NotImplementedError, "only supports Replicate"):
            DTensor.from_local(
                torch.ones(16),
                mesh,
                placements,
                shape=(4, 8),
            )

    def test_replicate_can_coexist_on_other_mesh_dimensions(self):
        """Allow Replicate alongside the single RaggedShard placement."""
        mesh = self._mesh((2, 2), ("ragged", "replicate"))
        placements = (
            RaggedShard(dims=(0,), local_units=(1, 1)),
            Replicate(),
        )

        result = DTensor.from_local(
            torch.arange(16),
            mesh,
            placements,
            shape=(4, 8),
        )

        self.assertEqual(tuple(result.shape), (4, 8))
        self.assertEqual(tuple(result.local_shape), (16,))

    def test_ragged_to_normal_gathers_flat_storage(self):
        """Gather weighted flat shards and restore the logical global shape."""
        mesh = self._mesh()
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))
        source = DTensor.from_local(
            torch.arange(64),
            mesh,
            (ragged,),
            shape=(6, 4, 8),
        )
        gathered = torch.arange(6 * 4 * 8)

        with patch.object(source.layout.mesh, "get_group", return_value="ragged_group"), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "platform.differentiable_variable_all_gather",
            return_value=gathered,
        ) as mock_all_gather:
            result = source.redistribute(mesh, (Replicate(),))

        self.assertEqual(tuple(result.shape), (6, 4, 8))
        self.assertEqual(tuple(result.local_shape), (6, 4, 8))
        self.assertTrue(torch.equal(result.to_local(), gathered.reshape(6, 4, 8)))
        _, output_splits, group = mock_all_gather.call_args.args
        self.assertEqual(output_splits, (64, 128))
        self.assertEqual(group, "ragged_group")

    def test_full_tensor_uses_global_shape_rank_for_replicate_layout(self):
        """Build the full-tensor target map from logical rank, not flat local rank."""
        mesh = self._mesh()
        source = DTensor.from_local(
            torch.arange(64),
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 2)),),
            shape=(6, 4, 8),
        )
        gathered = torch.arange(6 * 4 * 8)

        with patch.object(source.layout.mesh, "get_group", return_value="ragged_group"), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "platform.differentiable_variable_all_gather",
            return_value=gathered,
        ), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "_tensor_redistribution._redistribution_normal"
        ) as mock_normal_redistribution:
            result = source.full_tensor()

        self.assertTrue(torch.equal(result, gathered.reshape(6, 4, 8)))
        mock_normal_redistribution.assert_not_called()

    def test_elementwise_chain_preserves_ragged_layout_and_backward(self):
        """Run local elementwise kernels and inherit Ragged metadata and autograd edges."""
        mesh = self._mesh()
        local = torch.linspace(0.0, 1.0, 64, requires_grad=True)
        source = DTensor.from_local(
            local,
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 2)),),
            shape=(6, 4, 8),
        )

        result = torch.sigmoid(source * 1.5 - 0.25) + torch.sigmoid(source)
        expected = torch.sigmoid(local * 1.5 - 0.25) + torch.sigmoid(local)

        self.assertEqual(tuple(result.shape), (6, 4, 8))
        self.assertEqual(tuple(result.local_shape), (64,))
        self.assertEqual(tuple(result.placements), tuple(source.placements))
        self.assertTrue(torch.equal(result.to_local(), expected))
        result.to_local().sum().backward()
        self.assertIsNotNone(local.grad)

    def test_elementwise_does_not_prevalidate_misaligned_ragged_inputs(self):
        """Whitelisted elementwise ops do not add a separate Ragged validation layer."""
        mesh = self._mesh((3,), ("ragged",))
        first = DTensor.from_local(
            torch.arange(32),
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 2, 3)),),
            shape=(6, 4, 8),
        )
        second = DTensor.from_local(
            torch.arange(32),
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 3, 2)),),
            shape=(6, 4, 8),
        )

        result = first + second
        self.assertEqual(tuple(result.placements), tuple(first.placements))
        self.assertTrue(torch.equal(result.to_local(), first.to_local() + second.to_local()))

    def test_non_elementwise_op_remains_fail_closed(self):
        """Shape-changing and reduction ops still require explicit Ragged support."""
        mesh = self._mesh()
        source = DTensor.from_local(
            torch.arange(64, dtype=torch.float32),
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 2)),),
            shape=(6, 4, 8),
        )

        with self.assertRaisesRegex(RuntimeError, "does not support RaggedShard"):
            source.mean(dim=-1)

    def test_ragged_to_normal_continues_with_normal_redistribution(self):
        """Run the legacy normal path when the requested target is not the normal view."""
        mesh = self._mesh()
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))
        source = DTensor.from_local(
            torch.arange(64),
            mesh,
            (ragged,),
            shape=(6, 4, 8),
        )
        sentinel = object()

        with patch.object(source.layout.mesh, "get_group", return_value="ragged_group"), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "platform.differentiable_variable_all_gather",
            return_value=torch.arange(6 * 4 * 8),
        ), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "_tensor_redistribution._redistribution_normal",
            return_value=sentinel,
        ) as mock_normal_redistribution:
            result = source.redistribute(mesh, (Shard(1),))

        self.assertIs(result, sentinel)
        normal, target_layout = mock_normal_redistribution.call_args.args
        self.assertEqual(tuple(normal.placements), (Replicate(),))
        self.assertEqual(tuple(normal.local_shape), (6, 4, 8))
        self.assertEqual(tuple(target_layout.placements), (Shard(1),))

    def test_normal_to_ragged_slices_flat_storage(self):
        """Slice the target rank's weighted interval from a Replicate tensor."""
        mesh = self._mesh()
        global_tensor = torch.arange(6 * 4 * 8).reshape(6, 4, 8)
        source = DTensor.from_local(
            global_tensor,
            mesh,
            (Replicate(),),
            shape=(6, 4, 8),
        )
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))

        result = source.redistribute(mesh, (ragged,))

        self.assertEqual(tuple(result.shape), (6, 4, 8))
        self.assertEqual(tuple(result.local_shape), (64,))
        self.assertTrue(torch.equal(result.to_local(), torch.arange(64)))

    def test_normal_to_ragged_first_matches_target_normal_view(self):
        """Run normal redistribution before slicing when source layout differs."""
        mesh = self._mesh()
        source = DTensor.from_local(
            torch.arange(6 * 2 * 8).reshape(6, 2, 8),
            mesh,
            (Shard(1),),
            shape=(6, 4, 8),
        )
        normal = DTensor.from_local(
            torch.arange(6 * 4 * 8).reshape(6, 4, 8),
            mesh,
            (Replicate(),),
            shape=(6, 4, 8),
        )
        ragged = RaggedShard(dims=(0, 1), local_units=(1, 2))

        with patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "_tensor_redistribution._redistribution_normal",
            return_value=normal,
        ) as mock_normal_redistribution:
            result = source.redistribute(mesh, (ragged,))

        self.assertEqual(tuple(result.local_shape), (64,))
        self.assertTrue(torch.equal(result.to_local(), torch.arange(64)))
        _, target_normal_layout = mock_normal_redistribution.call_args.args
        self.assertEqual(tuple(target_normal_layout.placements), (Replicate(),))

    def test_ragged_all_to_all_splits_follow_flat_interval_overlaps(self):
        """Derive rank-zero send and receive lengths from changed local units."""
        mesh = self._mesh()
        source_layout = _build_layout(
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 2)),),
            tensor_dim=3,
        )
        target_layout = _build_layout(
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(2, 1)),),
            tensor_dim=3,
        )

        input_splits, output_splits = _compute_ragged_all_to_all_splits(
            (6, 4, 8),
            source_layout,
            target_layout,
        )

        self.assertEqual(input_splits, (64, 0))
        self.assertEqual(output_splits, (64, 64))

        with patch.object(source_layout.mesh, "get_local_rank", return_value=1), patch.object(
            target_layout.mesh,
            "get_local_rank",
            return_value=1,
        ):
            input_splits, output_splits = _compute_ragged_all_to_all_splits(
                (6, 4, 8),
                source_layout,
                target_layout,
            )

        self.assertEqual(input_splits, (64, 64))
        self.assertEqual(output_splits, (0, 64))

    def test_ragged_to_ragged_uses_variable_all_to_all(self):
        """Exchange flat interval overlaps when only local units change."""
        mesh = self._mesh()
        source = DTensor.from_local(
            torch.arange(64),
            mesh,
            (RaggedShard(dims=(0, 1), local_units=(1, 2)),),
            shape=(6, 4, 8),
        )
        target_placement = RaggedShard(dims=(0, 1), local_units=(2, 1))
        expected = torch.arange(128)

        with patch.object(source.layout.mesh, "get_group", return_value="ragged_group"), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "platform.differentiable_all_to_all_single",
            return_value=expected,
        ) as mock_all_to_all:
            result = source.redistribute(mesh, (target_placement,))

        self.assertEqual(tuple(result.shape), (6, 4, 8))
        self.assertEqual(tuple(result.local_shape), (128,))
        self.assertTrue(torch.equal(result.to_local(), expected))
        flat_input, input_splits, output_splits, group = mock_all_to_all.call_args.args
        self.assertTrue(torch.equal(flat_input, source.to_local()))
        self.assertEqual(input_splits, (64, 0))
        self.assertEqual(output_splits, (64, 64))
        self.assertEqual(group, "ragged_group")

    def test_ragged_to_ragged_dims_change_uses_replicate(self):
        """Gather to Replicate before slicing along different ragged dimensions."""
        mesh = self._mesh()
        source = DTensor.from_local(
            torch.arange(64),
            mesh,
            (RaggedShard(dims=(0,), local_units=(1, 2)),),
            shape=(6, 4, 8),
        )
        gathered = torch.arange(6 * 4 * 8)

        with patch.object(source.layout.mesh, "get_group", return_value="ragged_group"), patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "platform.differentiable_variable_all_gather",
            return_value=gathered,
        ) as mock_all_gather, patch(
            "hyper_parallel.core.dtensor.tensor_redistribution."
            "platform.differentiable_all_to_all_single",
        ) as mock_all_to_all:
            result = source.redistribute(
                mesh,
                (RaggedShard(dims=(0, 1), local_units=(2, 1)),),
            )

        self.assertEqual(tuple(result.shape), (6, 4, 8))
        self.assertEqual(tuple(result.local_shape), (128,))
        self.assertTrue(torch.equal(result.to_local(), gathered[:128]))
        _, output_splits, group = mock_all_gather.call_args.args
        self.assertEqual(output_splits, (64, 128))
        self.assertEqual(group, "ragged_group")
        mock_all_to_all.assert_not_called()


if __name__ == "__main__":
    unittest.main()
