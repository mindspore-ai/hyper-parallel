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
"""Independent routing and packed snapshot checks."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.distributed.context_parallel import halo
from hyper_parallel.components.modules.shared_compressed_dsa_attention import SharedCompressedPackedSequence
from tests.common.mark_utils import arg_mark


class TestSequenceHalo(unittest.TestCase):
    """Check routing against explicit token enumeration."""

    def test_prepared_batch_reuses_forward_geometry_and_detects_mutation(self):
        """Repeated forwards reuse prepared tensors; original input updates create a new snapshot."""
        boundaries = torch.tensor([0, 4, 8])
        packed = SharedCompressedPackedSequence(boundaries, 2, 4, 8)
        with patch("torch.bucketize", wraps=torch.bucketize) as bucketize:
            first = packed.prepare(torch.device("cpu"), (1, 2))
            mask = first.segment_start_mask(torch.device("cpu"))
            window = first.sliding_window_indices(torch.device("cpu"), 3, 0)
            for _ in range(4):
                self.assertIs(packed.prepare(torch.device("cpu"), (1, 2)), first)
                self.assertIs(first.prepare(torch.device("cpu"), (1, 2)), first)
                self.assertIs(first.segment_start_mask(torch.device("cpu")), mask)
                self.assertIs(first.sliding_window_indices(torch.device("cpu"), 3, 0), window)
            self.assertEqual(bucketize.call_count, 1)
        self.assertEqual(first.indexer_segments(2), ((0, 2, 0, 2, 0), (2, 4, 2, 3, 0)))
        boundaries[1] = 6
        second = packed.prepare(torch.device("cpu"), (1, 2))
        self.assertIsNot(second, first)
        torch.testing.assert_close(first.segment_start_mask(torch.device("cpu")),
                                   torch.tensor([[False, False, True, False]]))
        self.assertFalse(bool(second.segment_start_mask(torch.device("cpu")).any()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_packed_snapshot_reuses_geometry_and_isolates_batches(self):
        """Prepared geometry survives input mutation and is reused across layers."""
        boundaries = torch.tensor([0, 4, 8])
        packed = SharedCompressedPackedSequence(boundaries, 0, 8, 8)
        with patch("torch.bucketize", wraps=torch.bucketize) as bucketize:
            prepared = packed.prepare(torch.device("cpu"), (0, 1, 2, 2))
            for _ in range(40):
                starts = prepared.local_segment_starts(torch.device("cpu"))
                minima = prepared.minimum_key_indices(torch.device("cpu"), 2)
            self.assertEqual(bucketize.call_count, 1)
        boundaries[1] = 6
        torch.testing.assert_close(starts, torch.tensor([[0, 0, 0, 0, 4, 4, 4, 4]]))
        torch.testing.assert_close(minima, starts // 2)
        fresh = packed.prepare(torch.device("cpu"), (2,))
        torch.testing.assert_close(
            fresh.local_segment_starts(torch.device("cpu")), torch.tensor([[0, 0, 0, 0, 0, 0, 6, 6]])
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_packed_rejects_cross_sample_compression(self):
        """A prepared snapshot must retain compression-boundary validation."""
        packed = SharedCompressedPackedSequence(torch.tensor([0, 3, 8]), 0, 8, 8)
        with self.assertRaisesRegex(ValueError, "align"):
            packed.prepare(torch.device("cpu"), (2,))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_routes_match_causal_remote_tokens(self):
        """Feature: Contiguous CP halo and shared packed geometry.
        Description: All token owners agree with every receiver, including multi-owner halos.
        Expectation: Every receiver obtains exactly the enumerated causal remote tokens.
        """
        for size in (1, 2, 4, 8):
            for length in (1, 2, 8, 128):
                for window in (1, 2, 8, 128):
                    routes = [halo.build_sequence_halo_route(length, window, size, rank) for rank in range(size)]
                    for consumer, route in enumerate(routes):
                        wanted = list(range(max(0, consumer*length-window+1), consumer*length))
                        actual = []
                        for owner in range(size):
                            count = routes[owner].input_splits[consumer]
                            self.assertEqual(count, route.output_splits[owner])
                            offset = routes[owner].send_offsets[consumer]
                            actual.extend(owner*length+offset+i for i in range(count))
                        self.assertEqual(actual, wanted)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_route_inputs_fail(self):
        """Feature: Contiguous CP halo and shared packed geometry.
        Description: Reject invalid geometry before launching any collective.
        Expectation: Invalid route arguments raise ValueError before communication.
        """
        for args in ((0, 8, 2, 0), (8, 0, 2, 0), (8, 8, 0, 0), (8, 8, 2, 2), (8, 8, 2, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                halo.build_sequence_halo_route(*args)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp1_and_window1_keep_local_gradient(self):
        """Feature: Contiguous CP halo and shared packed geometry.
        Description: Globally communication-free cases preserve input order and autograd.
        Expectation: Output and gradients match the local identity without group access.
        """
        for size, rank, window in ((1, 0, 128), (4, 2, 1)):
            mesh = MagicMock()
            mesh.size.return_value, mesh.get_local_rank.return_value = size, rank
            value = torch.randn(2, 4, 3).transpose(0, 1).requires_grad_()
            handle = halo.async_cp_halo_launch(value, 0, window, mesh)
            output = handle.wait()
            torch.testing.assert_close(output, value)
            output.sum().backward()
            torch.testing.assert_close(value.grad, torch.ones_like(value))
            mesh.get_group.assert_not_called()
            with self.assertRaises(RuntimeError):
                handle.wait()
