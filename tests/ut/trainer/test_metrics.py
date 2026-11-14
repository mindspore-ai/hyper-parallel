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
"""Token-weighted loss reductions use only distinct DP/CP targets."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hyper_parallel.trainer.runtime import metrics


class TestLossMetrics(unittest.TestCase):
    """Check collective domains and gradients without process groups."""

    def test_tp_replicas_do_not_reduce_token_counts(self) -> None:
        """Feature: Token-weighted SP loss.
        Description: Aggregate LM/MTP losses with replicated TP counts.
        Expectation: Only DP/CP numerators reduce; local gradients retain their weight.
        """
        for tracing in (False, True):
            for sequence_parallel in (False, True):
                for count in (0, 3):
                    with self.subTest(tracing=tracing, sp=sequence_parallel, count=count):
                        group = object()
                        mesh = SimpleNamespace(
                            dp_cp_mesh=Mock(), dp_size=2, cp_size=2,
                            sequence_parallel=sequence_parallel, device_mesh=MagicMock(),
                        )
                        mesh.dp_cp_mesh.get_group.return_value = group
                        mesh.dp_cp_mesh.size.return_value = 4
                        parameter = torch.tensor(2.0, requires_grad=True)
                        counts = {"foundation_tokens": torch.tensor(count)}

                        def reduce_numerator(value, op, group):
                            self.assertEqual(op, "sum")
                            self.assertIs(group, mesh.dp_cp_mesh.get_group())
                            # The remote ranks contribute a fixed numerator of 10.
                            return value + 10

                        with patch.object(metrics, "_is_tracing_or_compiling", return_value=tracing), \
                                patch.object(metrics, "all_reduce", side_effect=reduce_numerator) as eager, \
                                patch.object(metrics, "differentiable_all_reduce",
                                             side_effect=reduce_numerator) as graph:
                            losses = metrics.mean_global_loss(
                                {"foundation_loss": parameter.square(), "foundation_loss_mtp": parameter * 2},
                                counts, mesh,
                            )
                        for loss in losses.values():
                            torch.testing.assert_close(loss, torch.tensor(4.0 * count + 10))
                        sum(losses.values()).backward()
                        torch.testing.assert_close(parameter.grad, torch.tensor(24.0 * count))
                        self.assertEqual(eager.call_count, 0 if tracing else 2)
                        self.assertEqual(graph.call_count, 2 if tracing else 0)
                        self.assertEqual(counts["foundation_tokens"].item(), count)
                        mesh.device_mesh.__getitem__.assert_not_called()

    def test_empty_shard_masks_nan_mean_before_weighting(self) -> None:
        """Feature: Empty loss shards.
        Description: Reduce cross entropy when every target is ignored.
        Expectation: Report a finite zero and keep zero gradients in both execution paths.
        """
        for tracing in (False, True):
            with self.subTest(tracing=tracing):
                logits = torch.randn(2, 4, requires_grad=True)
                loss = torch.nn.functional.cross_entropy(logits, torch.full((2,), -100))
                self.assertTrue(torch.isnan(loss))
                mesh = SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, sequence_parallel=False)
                with patch.object(metrics, "_is_tracing_or_compiling", return_value=tracing):
                    reduced = metrics.mean_global_loss(loss, {"foundation_tokens": torch.tensor(0)}, mesh)
                torch.testing.assert_close(reduced["foundation_loss"], torch.tensor(0.0))
                reduced["foundation_loss"].backward()
                torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
