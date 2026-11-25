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
"""Evaluation range and tail coverage must not depend on training samplers."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from hyper_parallel.trainer.config.tasks import EvaluationConfig
from hyper_parallel.trainer.runtime.eval_data import EvaluationBatchSampler, build_evaluation_dataloader
from hyper_parallel.trainer.runtime.eval_step import LossEvalStep
from hyper_parallel.evaluation import Evaluator, LossMetric, PerplexityMetric, ReductionContext
from tests.ut.evaluation.test_loss_eval_step import _batch, _FakeTrainer, _MeanLoss


class TestEvaluationData(unittest.TestCase):
    """Global caps precede sharding and all-ignored targets never produce NaN metrics."""

    def test_non_divisible_global_range_has_no_dropped_or_duplicated_samples(self):
        samplers = [EvaluationBatchSampler(7, 2, rank, 3) for rank in range(3)]
        indices = [index for sampler in samplers for batch in sampler for index in batch]
        self.assertEqual(sorted(indices), list(range(7)))
        self.assertEqual(list(samplers[0]), [[0, 3], [6]])
        self.assertEqual(list(samplers[0]), [[0, 3], [6]])

    def test_empty_lane_has_a_forward_shape_without_contributing_samples(self):
        sampler = EvaluationBatchSampler(1, 2, 1, 2)
        self.assertFalse(sampler.has_samples)
        self.assertEqual(list(sampler), [[0]])

    def test_empty_lane_is_read_from_the_split_batch_sampler(self):
        """The step derives lane emptiness from the sampler, not a trainer flag."""
        trainer = _FakeTrainer([_batch([[1, 2, 3]], [[1, 2, 3]])], _MeanLoss())
        trainer.valid_batch_sampler = SimpleNamespace(has_samples=False)
        self.assertIsNone(LossEvalStep(trainer)(0))

    def test_max_samples_applies_to_custom_dataset_builder(self):
        trainer = SimpleNamespace(config=SimpleNamespace(evaluation=EvaluationConfig(split="dataset", max_samples=5),
                                                        training=SimpleNamespace(seed=7)),
                                  valid_dataset=list(range(12)), mesh=SimpleNamespace(dp_rank=1, dp_size=2),
                                  collate_fn=list, data_transform=None)
        target = Mock(load_balance=None)
        build_evaluation_dataloader(trainer, target)
        self.assertEqual(list(trainer.valid_batch_sampler), [[1], [3]])
        self.assertFalse(target.build.call_args.kwargs["drop_last"])
        self.assertEqual(target.build.call_args.kwargs["sampler_type"], "single")

    def test_whole_masked_batch_does_not_poison_next_batch(self):
        batches = [_batch([[1, 2, 3]], [[-100, -100, -100]]), _batch([[1, 2, 3]], [[1, 2, 3]])]
        trainer = _FakeTrainer(batches, _MeanLoss())
        report = Evaluator(LossEvalStep(trainer), [LossMetric(), PerplexityMetric()],
                           ReductionContext(), torch.device("cpu")).run()
        self.assertTrue(torch.isfinite(torch.tensor(report.metrics["loss"])))
        self.assertEqual(report.num_samples, 1)
        self.assertEqual(report.num_tokens, 2)

    def test_shifted_first_token_counts_as_a_real_sample(self):
        labels = torch.tensor([[2, -100, -100]])
        count = LossEvalStep._real_sample_count({"labels": labels, "shift_labels": labels})
        self.assertEqual(count.item(), 1)

    def test_packed_samples_are_not_counted_as_one_tensor_row(self):
        labels = torch.tensor([[2, 3, -100, -100, 4, -100]])
        count = LossEvalStep._real_sample_count(
            {"labels": labels, "shift_labels": labels}, torch.tensor([0, 2, 4, 6]))
        self.assertEqual(count.item(), 2)
