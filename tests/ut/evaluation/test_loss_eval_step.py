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
"""Numerical tests for the Trainer-side evaluation step against a stub model."""

import math
import types
import unittest

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.evaluation import (
    Evaluator,
    LossMetric,
    PerplexityMetric,
    ReductionContext,
)
from hyper_parallel.trainer.runtime.evaluation import LossEvalStep


class _StubModel(torch.nn.Module):
    """Return a per-token mean cross-entropy, like a Transformers causal LM."""

    def __init__(self, vocab: int = 7) -> None:
        """Create a seeded embedding that doubles as the output projection."""
        super().__init__()
        self.vocab = vocab
        self.embed = torch.nn.Embedding(vocab, vocab)
        torch.nn.init.normal_(self.embed.weight, generator=torch.Generator().manual_seed(0))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor,
                use_cache: bool = False, **_: object) -> types.SimpleNamespace:
        """Return per-token mean cross-entropy and the logits it came from."""
        del use_cache
        logits = self.embed(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return types.SimpleNamespace(loss=loss, logits=logits)


class _MeanLoss(torch.nn.Module):
    """Loss module that declares its normalization, as the contract requires."""

    loss_normalization = "token_mean"

    def forward(self, *, model_output: types.SimpleNamespace, labels: torch.Tensor) -> torch.Tensor:
        """Read the model-computed mean loss."""
        del labels
        return model_output.loss


class _UndeclaredLoss(torch.nn.Module):
    """Loss module that omits the normalization declaration."""

    def forward(self, *, model_output: types.SimpleNamespace, labels: torch.Tensor) -> torch.Tensor:
        """Read the model-computed mean loss without declaring its normalization."""
        del labels
        return model_output.loss


def _batch(ids, labels):
    """Build the (model_inputs, loss_inputs) pair a get_batch runtime returns."""
    ids = torch.tensor(ids)
    labels = torch.tensor(labels)
    return {"input_ids": ids, "labels": labels}, {"labels": labels}


class _FakeTrainer:
    """Minimal Trainer surface the evaluation step reads."""

    def __init__(self, batches: list, loss_fn: torch.nn.Module) -> None:
        """Hold the stub model, loss, and one fixed validation DataLoader."""
        self.model = _StubModel()
        self.loss_fn = loss_fn
        self.valid_dataloader = list(batches)
        self._cursor = None

    def get_batch(self, iterator: object) -> tuple:
        """Return the next prepared (model_inputs, loss_inputs) pair."""
        return next(iterator)


def _reference_loss(model, batches):
    """Compute the exact token-weighted mean loss in one pass over all tokens."""
    total, count = 0.0, 0
    for model_inputs, _ in batches:
        logits = model.embed(model_inputs["input_ids"])[:, :-1].reshape(-1, model.vocab)
        targets = model_inputs["labels"][:, 1:].reshape(-1)
        valid = targets != IGNORE_INDEX
        per_token = torch.nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, reduction="none")
        total += float(per_token[valid].sum())
        count += int(valid.sum())
    return total / count, count


class TestLossEvalStep(unittest.TestCase):
    """The step's additive statistics reproduce a single-pass global mean."""

    def _run(self, batches, loss_fn=None, max_batches=None):
        trainer = _FakeTrainer(batches, loss_fn or _MeanLoss())
        step = LossEvalStep(trainer, split_name="valid", max_batches=max_batches)
        metrics = [LossMetric(), PerplexityMetric()]
        evaluator = Evaluator(step, metrics, ReductionContext(), torch.device("cpu"))
        return trainer, evaluator.run(max_batches)

    def test_unequal_batches_match_single_pass_reference(self):
        """Batches with different valid-token counts must be token weighted."""
        batches = [
            _batch([[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]),
            _batch([[2, 3, 1, 6, 0], [4, 4, 4, 4, 4]], [[2, 3, IGNORE_INDEX, IGNORE_INDEX, 0], [4, 4, 4, 4, 4]]),
        ]
        trainer, report = self._run(batches)
        expected, tokens = _reference_loss(trainer.model, batches)
        self.assertAlmostEqual(report.metrics["loss"], expected, places=6)
        self.assertAlmostEqual(report.metrics["perplexity"], math.exp(expected), places=5)
        self.assertEqual(int(report.num_tokens), tokens)
        self.assertEqual(report.num_batches, 2)

    def test_fully_masked_sample_is_not_counted(self):
        """A sample whose targets are all ignored adds neither loss nor weight."""
        batches = [
            _batch([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [IGNORE_INDEX] * 3]),
        ]
        trainer, report = self._run(batches)
        expected, _ = _reference_loss(trainer.model, batches)
        self.assertAlmostEqual(report.metrics["loss"], expected, places=6)
        self.assertEqual(report.num_samples, 1)

    def test_max_batches_bounds_the_loop(self):
        """The round cap limits forwards and the reference restricts to them."""
        batches = [_batch([[1, 2, 3]], [[1, 2, 3]]) for _ in range(5)]
        trainer, report = self._run(batches, max_batches=2)
        expected, _ = _reference_loss(trainer.model, batches[:2])
        self.assertEqual(report.num_batches, 2)
        self.assertAlmostEqual(report.metrics["loss"], expected, places=6)

    def test_undeclared_normalization_is_refused(self):
        """A loss that does not say how it is normalized cannot be weighted."""
        batches = [_batch([[1, 2, 3]], [[1, 2, 3]])]
        with self.assertRaisesRegex(ValueError, "loss_normalization"):
            self._run(batches, loss_fn=_UndeclaredLoss())

    def test_forward_runs_without_building_a_graph(self):
        """The step must not retain autograd state across batches."""
        batches = [_batch([[1, 2, 3]], [[1, 2, 3]])]
        trainer = _FakeTrainer(batches, _MeanLoss())
        step = LossEvalStep(trainer, split_name="valid")
        result = step(0)
        statistic = result.statistics["foundation"]
        self.assertFalse(statistic.numerator.requires_grad)
        self.assertEqual(statistic.numerator.dtype, torch.float64)

    def test_exhausted_dataloader_ends_the_run(self):
        """Running past the data returns None rather than raising."""
        batches = [_batch([[1, 2, 3]], [[1, 2, 3]])]
        trainer = _FakeTrainer(batches, _MeanLoss())
        step = LossEvalStep(trainer, split_name="valid")
        self.assertIsNotNone(step(0))
        self.assertIsNone(step(1))


if __name__ == "__main__":
    unittest.main()
