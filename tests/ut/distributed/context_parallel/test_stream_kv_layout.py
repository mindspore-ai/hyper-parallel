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

"""Independent causal oracle for mirrored Q, paired KV and owner gradient return."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.context_parallel import _stream_kv_layout as layout
from hyper_parallel.distributed.context_parallel import stream_kv as implementation
from tests.ut.distributed.context_parallel.test_stream_kv import DenseNative, Mesh, dense_reference


def _indices(degree, local, owner):
    half = local // 2
    return torch.cat((torch.arange(owner * half, (owner + 1) * half),
                      torch.arange((2 * degree - 1 - owner) * half, (2 * degree - owner) * half)))


class TestStreamKVLayout(unittest.TestCase):
    """Use global token IDs and global autograd, never a balanced reference kernel."""

    @classmethod
    def setUpClass(cls) -> None:
        """Avoid excessive CPU threads for small mathematical fixtures."""
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_permutation_and_inverse(self):
        """Check sparse peer splits using independent global token IDs.

        Feature: Causal mirror permutation and its inverse.
        Description: Mock peer halves for even and odd group sizes.
        Expectation: Forward/inverse results and all send splits match exact token IDs.
        """
        for degree in (1, 2, 3, 4, 7, 8):
            local = 6
            full = torch.arange(degree * local * 4).view(1, 2, degree * local, 2).float()
            for inverse in (False, True):
                contiguous = [torch.arange(rank * local, (rank + 1) * local) for rank in range(degree)]
                mirrored = [_indices(degree, local, rank) for rank in range(degree)]
                source, target = (mirrored, contiguous) if inverse else (contiguous, mirrored)
                for owner in range(degree):
                    with self.subTest(degree=degree, inverse=inverse, owner=owner):

                        def exchange(received: torch.Tensor, send: torch.Tensor, output_split_sizes: list,
                                     input_split_sizes: list, group: Mesh) -> None:
                            """Supply independently indexed peer halves and check exact send splits."""
                            self.assertEqual(group.owner, owner)
                            send_parts, receive_parts = [], []
                            for peer in range(degree):
                                sending = source[owner][torch.isin(source[owner], target[peer])]
                                receiving = source[peer][torch.isin(source[peer], target[owner])]
                                self.assertEqual(input_split_sizes[peer], len(sending))
                                self.assertEqual(output_split_sizes[peer], len(receiving))
                                send_parts.append(full[:, :, sending].permute(2, 0, 1, 3))
                                receive_parts.append(full[:, :, receiving].permute(2, 0, 1, 3))
                            torch.testing.assert_close(send, torch.cat(send_parts), rtol=0, atol=0)
                            received.copy_(torch.cat(receive_parts))

                        with patch.object(layout.dist, "all_to_all_single", side_effect=exchange):
                            actual = layout._redistribute(full[:, :, source[owner]], Mesh(degree, owner), inverse)
                        torch.testing.assert_close(actual, full[:, :, target[owner]], rtol=0, atol=0)

    def test_balanced_global_forward_and_vjp(self):
        """Compare both Q halves' contributions with global dense autograd.

        Feature: Mirrored Q and paired KV global forward/backward.
        Description: Assemble every query owner's partial KV gradients over ragged panels.
        Expectation: Output, dQ and summed dK/dV match the independent FP64 reference.
        """
        generator = torch.Generator().manual_seed(941)
        for degree, local, panel, block, chunk in ((1, 10, 3, 2, 3), (2, 10, 6, 5, None),
                                                   (3, 10, 4, 7, 2), (4, 6, 99, 2, 1)):
            with self.subTest(degree=degree, panel=panel, block=block):
                dim, heads, kv_heads = 8, 8, 2
                query = torch.randn(1, heads, degree * local, dim, generator=generator)
                key, value = [torch.randn(1, kv_heads, degree * local, dim, generator=generator) for _ in range(2)]
                incoming = torch.randn(query.shape, generator=generator)
                reference, gradients = dense_reference(query, key, value, incoming)
                outputs, dquery = torch.empty_like(query), torch.empty_like(query)
                dkey, dvalue = torch.zeros_like(key), torch.zeros_like(value)
                indices = [_indices(degree, local, rank) for rank in range(degree)]
                config = implementation.StreamKVConfig(panel, block, chunk, causal_load_balance=True)
                half = local // 2
                bounds = [(start, min(start + panel // 2, half)) for start in range(0, half, panel // 2)]
                contributions = []
                for owner in range(degree):
                    gather_bounds, reduce_bounds = iter(bounds * 2), iter(bounds)
                    owner_contribution = torch.zeros(degree, local, 2, 1, kv_heads, dim)

                    def gather(received: torch.Tensor, send: torch.Tensor, group: Mesh) -> None:
                        """Supply canonical global tokens for paired owner stripes."""
                        start, end = next(gather_bounds)
                        selected = [torch.cat((tokens[start:end], tokens[half + start:half + end]))
                                    for tokens in indices]
                        packed = torch.stack([torch.stack((key[:, :, tokens], value[:, :, tokens]), 2).permute(
                            3, 2, 0, 1, 4) for tokens in selected])
                        torch.testing.assert_close(send, packed[group.owner], rtol=0, atol=0)
                        received.copy_(packed.flatten(0, 1))

                    def reduce(received: torch.Tensor, send: torch.Tensor, op: object, group: Mesh) -> None:
                        """Capture partial KV gradients before independently summing query owners."""
                        self.assertEqual(op, torch.distributed.ReduceOp.SUM)
                        self.assertEqual(send.dtype, torch.float32)
                        start, end = next(reduce_bounds)
                        packed = send.view(degree, 2 * (end - start), 2, 1, kv_heads, dim)
                        owner_contribution[:, start:end] = packed[:, :end - start]
                        owner_contribution[:, half + start:half + end] = packed[:, end - start:]
                        received.copy_(packed[group.owner])

                    tokens = indices[owner]
                    with patch.object(implementation, "_npu_ops", return_value=DenseNative), (
                            patch.object(layout.dist, "all_gather_into_tensor", side_effect=gather)), (
                            patch.object(layout.dist, "reduce_scatter_tensor", side_effect=reduce)):
                        out, maximum, total = implementation._forward(query[:, :, tokens], key[:, :, tokens],
                            value[:, :, tokens], Mesh(degree, owner), dim**-.5, config)
                        dq, dk, dv = implementation._backward(query[:, :, tokens], key[:, :, tokens],
                            value[:, :, tokens], out, maximum, total, incoming[:, :, tokens],
                            Mesh(degree, owner), dim**-.5, config)
                    outputs[:, :, tokens], dquery[:, :, tokens] = out, dq
                    contributions.append(owner_contribution)
                    if degree == 1:
                        dkey[:, :, tokens], dvalue[:, :, tokens] = dk, dv
                if degree > 1:
                    summed = torch.stack(contributions).sum(0)
                    for owner, tokens in enumerate(indices):
                        dkey[:, :, tokens] = summed[owner, :, 0].permute(1, 2, 0, 3)
                        dvalue[:, :, tokens] = summed[owner, :, 1].permute(1, 2, 0, 3)
                for actual, expected in zip((outputs, dquery, dkey, dvalue), (reference, *gradients)):
                    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-6)

    def test_equal_pairs_per_round_at_large_degree(self):
        """Count scheduled causal pairs through logical Pg1280.

        Feature: Per-panel causal load balancing at large logical group sizes.
        Description: Count scheduled FULL/CAUSAL areas including tail panels.
        Expectation: Each panel is rank-balanced and totals cover the global triangle.
        """
        local, budget = 18, 8
        for degree in (1, 3, 8, 128, 1280):
            totals = [0] * degree
            for start, end in layout._panel_bounds(local, budget, True):
                work = []
                for owner in range(degree):
                    pairs = 0
                    for begin, finish, q_start, q_end, causal, _ in layout._pieces(
                            owner, degree, end - start, start, local, 2**20, True):
                        keys, queries = finish - begin, q_end - q_start
                        pairs += queries * keys - (keys * (keys - 1) // 2 if causal else 0)
                    work.append(pairs)
                    totals[owner] += pairs
                self.assertEqual(min(work), max(work), f"Expected equal per-panel work at Pg={degree}")
            expected = (degree * local) * (degree * local + 1) // (2 * degree)
            self.assertEqual(totals, [expected] * degree)

    def test_invalid_balanced_config(self):
        """Check balanced configuration at the input boundary.

        Feature: Nonempty paired KV stripes and explicit balance enablement.
        Description: Pass insufficient token budgets and non-boolean balance values.
        Expectation: Every invalid configuration raises ValueError.
        """
        for kwargs in ({"owner_panel_tokens": 1, "causal_load_balance": True},
                       {"owner_panel_tokens": 4, "causal_load_balance": 1},
                       {"owner_panel_tokens": None, "causal_load_balance": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                implementation.StreamKVConfig(key_block_tokens=2, **kwargs)
