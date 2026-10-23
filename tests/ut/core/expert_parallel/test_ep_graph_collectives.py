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
"""CPU contracts for EP count exchange and runtime split sizes."""

import unittest
from unittest.mock import patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel.distributed.expert_parallel import collectives
from hyper_parallel.distributed.expert_parallel.collectives import ep_exchange_counts, ep_split_sizes


def _receive_buffer(counts):
    """Keep the output row count dependent on this invocation's metadata."""
    return counts.new_empty((sum(ep_split_sizes(counts)), 3))


class TestEpGraphCollectives(unittest.TestCase):
    """Preserve eager collectives and symbolize routing counts only in graphs."""

    def test_eager_exchange_and_zero_splits(self):
        """The eager helper invokes the original count collective unchanged."""
        counts = torch.tensor([0, 3])
        with patch("hyper_parallel.distributed.expert_parallel.collectives.dist") as backend:
            backend.all_to_all_single.side_effect = lambda out, value, **kwargs: out.copy_(value.flip(0))
            received = ep_exchange_counts(counts, "ep")
        self.assertEqual(ep_split_sizes(received), [3, 0])
        self.assertEqual(backend.all_to_all_single.call_args.kwargs, {"group": "ep"})

    def test_one_graph_keeps_runtime_split_sizes(self):
        """A fixed-size count vector can describe empty or changing receives."""
        with patch.object(collectives, "_is_tracing_or_compiling", return_value=True):
            graph = make_fx(_receive_buffer, tracing_mode="symbolic")(torch.tensor([1, 2]))
        for counts in ([0, 0], [0, 5], [3, 1]):
            actual = graph(torch.tensor(counts))
            self.assertEqual(tuple(actual.shape), (sum(counts), 3))

    def test_capture_collectives_preserve_payload_autograd(self):
        """Wait for count metadata while leaving payload waits to its consumer."""
        counts = torch.tensor([0, 3])
        inputs = torch.randn(3, 4, requires_grad=True)
        with patch.object(collectives, "_is_tracing_or_compiling", return_value=True), \
                patch.object(collectives, "_functional_collectives") as backend:
            backend.all_to_all_single.return_value = counts.flip(0)
            backend.wait_tensor.side_effect = lambda value: value
            backend.all_to_all_single_autograd.side_effect = lambda value, *_args: value * 2
            received = ep_exchange_counts(counts, "ep")
            output = collectives.ep_all_to_all(inputs, [0, 3], [3, 0], "ep")
            gradient, = torch.autograd.grad(output.sum(), inputs)
            backend.all_to_all_single.assert_called_once_with(counts, None, None, "ep")
            backend.wait_tensor.assert_called_once_with(received)
            backend.all_to_all_single_autograd.assert_called_once_with(inputs, [3, 0], [0, 3], "ep")
        torch.testing.assert_close(received, torch.tensor([3, 0]))
        torch.testing.assert_close(gradient, torch.full_like(inputs, 2))

    def test_capture_split_sizes_validate_receive_shape(self):
        """Zero segments are legal; negative sizes and inconsistent totals fail."""
        with patch.object(collectives, "_is_tracing_or_compiling", return_value=True):
            self.assertEqual(ep_split_sizes(torch.tensor([0, 3]), total=3), [0, 3])
            with self.assertRaises(RuntimeError):
                ep_split_sizes(torch.tensor([-1, 3]))
            with self.assertRaises(RuntimeError):
                ep_split_sizes(torch.tensor([0, 3]), total=4)
