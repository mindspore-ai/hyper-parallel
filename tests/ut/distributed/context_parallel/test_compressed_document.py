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
"""CPU coverage for compact compressed-bank all-gather gradients."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch.nn import functional

from hyper_parallel.distributed.context_parallel.compressed_document import document_compression_plan


class TestCompactBankGather(unittest.TestCase):
    """De-padding must return each gradient once and discard padded rows."""

    def test_uneven_owners_and_noncontiguous_gradient(self):
        """Match independent owner banks, including empty owners and BF16 gradients."""
        for counts in ((3, 0, 2, 1), (1, 0, 1, 1)):
            for batch, dtype in ((1, torch.bfloat16), (2, torch.bfloat16), (2, torch.float32)):
                with self.subTest(counts=counts, batch=batch, dtype=dtype):
                    capacity = max(counts)
                    banks = [torch.arange(batch * count * 4, dtype=dtype).reshape(batch, count, 4)
                             .requires_grad_() for count in counts]
                    references = [value.detach().clone().requires_grad_() for value in banks]
                    rank = 1
                    plan = replace(document_compression_plan((0, 8), 2, 2, 4, rank), counts=counts)
                    padded = [functional.pad(bank, (0, 0, 0, capacity - count))
                              for bank, count in zip(banks, counts)]
                    gathered = torch.cat(padded, dim=1)
                    gathered.retain_grad()
                    handle = Mock()
                    handle.wait.return_value = gathered
                    context = SimpleNamespace(launch=Mock(return_value=handle))
                    pending = plan.launch_bank(banks[rank], context)
                    handle.wait.assert_not_called()
                    actual = pending.wait()
                    handle.wait.assert_called_once_with()
                    torch.testing.assert_close(context.launch.call_args.args[0], padded[rank], rtol=0, atol=0)
                    self.assertEqual(context.launch.call_args.args[1], 1)
                    expected = torch.cat(references, dim=1)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    gradient = torch.arange(actual.numel() * 2, dtype=torch.float32).reshape(
                        batch, sum(counts), 8).to(dtype)[..., ::2]
                    self.assertFalse(gradient.is_contiguous())
                    # Two consumers exercise accumulation before the compaction inverse.
                    (actual + actual).backward(gradient)
                    (expected + expected).backward(gradient)
                    for bank, reference in zip(banks, references):
                        torch.testing.assert_close(bank.grad, reference.grad, rtol=0, atol=0)
                    for owner, count in enumerate(counts):
                        padding = gathered.grad[:, owner * capacity + count:(owner + 1) * capacity]
                        torch.testing.assert_close(padding, torch.zeros_like(padding), rtol=0, atol=0)

    def test_equal_and_empty_owners_keep_fast_paths(self):
        """Equal capacities need no compaction; all-empty banks need no communication."""
        for count in (0, 2):
            with self.subTest(count=count):
                bank = torch.ones(1, count, 4, requires_grad=True)
                plan = replace(document_compression_plan((0, 8), 2, 2, 4, 0), counts=(count,) * 4)
                gathered = bank.repeat(1, 4, 1)
                handle = Mock()
                handle.wait.return_value = gathered
                context = SimpleNamespace(launch=Mock(return_value=handle))
                actual = plan.launch_bank(bank, context).wait()
                self.assertIs(actual, gathered if count else bank)
                if count:
                    context.launch.assert_called_once_with(bank, 1)
                else:
                    context.launch.assert_not_called()
                actual.sum().backward()
                torch.testing.assert_close(bank.grad, torch.full_like(bank, 4), rtol=0, atol=0)
