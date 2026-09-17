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
"""Ordered KDA summaries, local-transfer ownership, and subgroup contracts."""
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional import kimi_delta_attention_cp as cp
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_mesh import build_kda_boundary
from tests.common.mark_utils import arg_mark
from tests.ut.components.functional.kda_cp_test_utils import Mailbox, serial_reference


class TestKDAAllGather(unittest.TestCase):
    """Exercise production CPU collectives against an FP64 autograd oracle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_noncommuting_summaries_and_reverse_invocations(self):
        """Feature: KDA boundary ordering and invocation ownership.

        Description: Run three live forwards and reverse backwards on noncontiguous peers.
        Expectation: Results match the FP64 adjoint oracle within FP32 composition error.
        """
        for size, width in ((1, 1), (3, 3), (4, 4), (4, 2), (6, 2), (6, 3), (8, 2)):
            with self.subTest(size=size, width=width):
                self._check(size, width)

    def _check(self, size, width):
        generator = torch.Generator().manual_seed(741 + size)
        shape = (1, 2, 128, 128)
        states = [torch.randn(shape, generator=generator) for _ in range(size)]
        matrices = [torch.eye(128) * .9 + torch.randn(shape, generator=generator) * .003 for _ in range(size)]
        gradients = [torch.randn(shape, generator=generator) for _ in range(size)]
        ranks = tuple(2 * rank + 1 for rank in range(size))
        protocols = []
        for rank in range(size):
            group = (ranks[rank], ranks)
            start = rank // width * width
            intra = (ranks[rank], ranks[start:start + width])
            protocols.append(cp.AllGatherBoundary(group, rank, size) if width == size else
                             cp.GroupedAllGatherBoundary(group, intra, rank, ranks, width))
        mailbox = Mailbox(2 * size)
        inputs, outputs = [], []
        with patch.object(cp, "dist", mailbox), ThreadPoolExecutor(size) as pool:
            for step in range(3):
                inputs.append([(s * (1 + .1 * step), m * (1 - .03 * step))
                               for s, m in zip(states, matrices)])
                outputs.append(list(pool.map(lambda r: protocols[r].forward(*inputs[step][r]), range(size))))
            for step in reversed(range(3)):
                snapshots = [row[1].clone() for row in inputs[step]]
                actual = list(pool.map(lambda r: protocols[r].backward(gradients[r], inputs[step][r][1]), range(size)))
                expected, adjoints = serial_reference(states, matrices, gradients, step)
                self._assert_ordered_feedback(inputs[step], gradients, outputs[step], actual)
                for rank in range(size):
                    self._assert_fp32_composition(outputs[step][rank], expected[rank])
                    self._assert_fp32_composition(actual[rank], adjoints[rank])
                    self.assertTrue(torch.equal(inputs[step][rank][1], snapshots[rank]))
                    self.assertFalse(any(isinstance(value, torch.Tensor) for value in vars(protocols[rank]).values()))
        self.assertTrue(all(queue.empty() for queue in mailbox.queues.values()))
        self.assertTrue(all(queue.empty() for queue in mailbox.collective_queues.values()))

    def _assert_ordered_feedback(self, summaries, gradients, outputs, adjoints):
        """Catch group reassociation even when an FP64 norm check would accept it."""
        state = torch.zeros_like(summaries[0][0])
        for index, (offset, matrix) in enumerate(summaries):
            self.assertTrue(torch.equal(outputs[index], state))
            state = matrix @ state + offset
        state = torch.zeros_like(gradients[0])
        for index in reversed(range(len(summaries))):
            self.assertTrue(torch.equal(adjoints[index], state))
            state = summaries[index][1].transpose(-1, -2) @ state + gradients[index]

    def _assert_fp32_composition(self, actual, expected):
        """Bound norm and peak errors without dividing by cancelling coordinates."""
        expected = expected.detach()
        delta = actual.double() - expected
        if expected.norm() == 0:
            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
            return
        # Near-zero entries need a tensor-scale bound alongside a normwise bound
        # when comparing the FP32 recurrence with the independent FP64 oracle.
        self.assertLessEqual(float(delta.norm() / expected.norm()), 1e-6)
        self.assertLessEqual(float(delta.abs().max() / expected.abs().max()), 2e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_contracts_rejected_before_transport(self):
        """Feature: KDA boundary validation.

        Description: Pass incompatible shapes, dtypes, rank orders and options.
        Expectation: Invalid calls fail before transport is invoked.
        """
        state = torch.zeros(1, 2, 128, 128)
        protocols = [cp.AllGatherBoundary(None, 0, 4),
                     cp.GroupedAllGatherBoundary(None, None, 0, (0, 2, 4, 6), 2)]
        with patch.object(cp, "dist") as transport:
            for protocol in protocols:
                with self.assertRaises(ValueError):
                    protocol.forward(state, state.double())
                with self.assertRaises(ValueError):
                    protocol.backward(state, state[..., :64])
            transport.all_gather_into_tensor.assert_not_called()
            transport.irecv.assert_not_called()
        with self.assertRaisesRegex(ValueError, "ordered"):
            cp.GroupedAllGatherBoundary(None, None, 0, (2, 0, 6, 4), 2)
        with self.assertRaisesRegex(ValueError, "only valid"):
            build_kda_boundary(None, "allgather", 2)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            build_kda_boundary(None, "unknown", 1)
