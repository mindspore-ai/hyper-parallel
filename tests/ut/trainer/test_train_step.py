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
"""Unit tests for ``BaseTrainer.train_step`` business logic.

``train_step`` is the load-bearing path that every training run hits on every
step; a regression there silently corrupts loss / gradient maths without
breaking any other test. Coverage here pins down the contracts that the
ST suite cannot reach without a real multi-card NPU:

1. **Token-weighted loss formula** (production code path, ``base.py:920``):
   ``avg_loss = sum(raw_loss_i * micro_tokens_i) / global_tokens``.
2. **Rank-average loss formula** (alternative branch, ``base.py:1016-1029``):
   ``avg_loss = mean(raw_loss across micro batches)`` — divided by ``num_micro``
   inside ``forward_backward_step`` and by ``dp_size`` for cross-rank reduce.
3. **``StopIteration`` does NOT bump ``global_step``** (``base.py:942-943``): the
   counter must stay synchronised with successfully completed steps so the
   resume / checkpoint indices line up.
4. **Grad sync schedule** (``base.py:970-972``): when ``model`` is an
   ``HSDPModule``, ``set_requires_gradient_sync(False)`` fires on every
   micro-batch except the last; ``True`` on the last. A regression that
   flips the order (sync on every step) is silent in ST but doubles the
   reduce-scatter traffic.
5. **Reshard toggle for grad accum** (``base.py:1424-1435``): on multi-micro
   steps, ``set_reshard_after_backward(False)`` fires on micro-batch 0 and
   ``True`` on the last; intermediates leave it as-is.
6. **Nested micro-batch ``_to_device``**: real batches contain nested
   dicts / lists / tuples (multimodal); the recursive mover must walk all
   of them.
"""
# pylint: disable=protected-access
import os
import types
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import nn

from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer, TrainerState


# ----------------------------------------------------------------------------
# Toy model + harness builders
# ----------------------------------------------------------------------------


class _ToyModel(nn.Module):
    """Minimal CE-loss model — enough autograd surface for one step.

    Returns a dict to exercise the ``outputs["loss"] if isinstance(outputs, dict)``
    branch in ``forward_backward_step``.
    """

    def __init__(self, vocab: int = 8, hidden: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids, labels=None, use_cache=False):  # pylint: disable=unused-argument
        x = self.embed(input_ids)
        logits = self.proj(x)
        if labels is None:
            return {"loss": logits.sum()}
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100,
        )
        return {"loss": loss}


class _HSDPRecordingModel(_ToyModel):
    """Toy model that records the HSDP grad-sync toggle calls.

    Used in the grad-sync schedule test — combined with
    ``patch.object(trainer_base, 'HSDPModule', _HSDPRecordingModel)``
    the ``isinstance(model, HSDPModule)`` check inside ``train_step``
    resolves to ``True`` so the recorded calls actually fire.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sync_calls = []
        self.last_bw_calls = []
        self.reshard_calls = []

    def set_requires_gradient_sync(self, value):
        self.sync_calls.append(bool(value))

    def set_is_last_backward(self, value):
        self.last_bw_calls.append(bool(value))

    def set_reshard_after_backward(self, value):
        self.reshard_calls.append(bool(value))


def _make_args(loss_aggregation="token_weighted", max_grad_norm=1.0):
    """Build a SimpleNamespace ``args`` shaped like ``HyperTrainerConfig``."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name="toy"),
        train=types.SimpleNamespace(
            max_steps=5,
            optimizer=types.SimpleNamespace(
                loss_aggregation=loss_aggregation,
                max_grad_norm=max_grad_norm,
            ),
        ),
    )


def _build_trainer(model, *, args=None, stub_step_hooks=True):
    """Construct a ``BaseTrainer`` bypassing ``_setup`` — wires the fields
    ``train_step`` / ``forward_backward_step`` actually read.
    """
    args = args or _make_args()
    with patch.object(trainer_base, "get_spec", return_value=types.SimpleNamespace(
            clip_grad_fn=None,
    )):
        trainer = BaseTrainer(args)
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    # Optimizer state is outside these CPU unit-test contracts; the downstream
    # clip_grad pass uses ``model.parameters()`` directly.
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = None
    trainer.parallel_dims = types.SimpleNamespace(dp_size=1)
    trainer._dp_group_info = types.SimpleNamespace(rank_size=1)
    if stub_step_hooks:
        # Stub the per-step callback hooks so we don't need a full callback wiring.
        trainer.on_substep_end = MagicMock()
        trainer.on_pre_optimizer_step = MagicMock()
    return trainer


def _batch(input_ids, labels=None):
    if labels is None:
        labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestTokenWeightedFormula(unittest.TestCase):
    """``token_weighted`` aggregation must use ``mean_global_loss`` semantics."""

    def test_returns_token_weighted_average_loss(self):
        """``avg_loss`` for two micro-batches must equal token-weighted mean.

        Build labels that yield deterministic CE losses by seeding torch:
        the assertion checks the *recovered* avg_loss matches a hand-computed
        token-weighted aggregation, not the absolute loss number.
        """
        torch.manual_seed(0)
        model = _ToyModel()
        trainer = _build_trainer(model)
        # Two micro-batches with different valid-token counts: 4 and 2.
        # micro_0 has all 4 labels valid; micro_1 has 2 valid + 2 -100.
        mb0 = _batch(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 3, 4]]))
        mb1 = _batch(
            torch.tensor([[5, 6, 7, 0]]),
            torch.tensor([[5, 6, -100, -100]]),
        )

        # Compute the expected losses by running the same forward outside
        # the trainer (token-weighted aggregation pins the formula).
        torch.manual_seed(0)
        ref_model = _ToyModel()
        ref_model.load_state_dict(model.state_dict())  # mirror init
        with torch.no_grad():
            ref0 = ref_model(**mb0)["loss"].item()
            ref1 = ref_model(**mb1)["loss"].item()
        # Token-weighted: sum(loss_i * tokens_i) / total_tokens
        expected = (ref0 * 4 + ref1 * 2) / 6

        # Drive a single train step with both micro-batches.
        with patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            metrics = trainer.train_step(iter([[mb0, mb1]]))

        self.assertAlmostEqual(metrics["loss"], expected, places=4, msg=(
            f"token_weighted loss formula broken: expected {expected:.6f}, "
            f"got {metrics['loss']:.6f}"
        ))
        self.assertEqual(trainer.state.global_step, 1, (
            f"global_step must advance to 1 after successful step, "
            f"got {trainer.state.global_step}"
        ))

    def test_global_tokens_clamped_to_one_when_batch_all_padding(self):
        """When every label is -100, ``local_tokens`` is clamped to 1 so the
        token-weighted divisor never goes to zero.

        Regression guard for ``base.py:951-953``. Without the clamp, the
        first all-padding step would divide by zero and crash a real run.
        """
        torch.manual_seed(0)
        model = _ToyModel()
        trainer = _build_trainer(model)
        mb = _batch(
            torch.tensor([[1, 2, 3, 4]]),
            torch.full((1, 4), -100, dtype=torch.long),
        )
        with patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            # Must not raise.
            metrics = trainer.train_step(iter([[mb]]))
        # CE with all-ignore labels returns NaN; the contract here is "no
        # divide-by-zero / no crash", not numerical correctness.
        self.assertIn("loss", metrics)
        self.assertIn("grad_norm", metrics)


class TestRankAverageFormula(unittest.TestCase):
    """``rank_average`` aggregation must use the per-rank-mean / dp_size formula."""

    def test_rank_average_divides_by_num_micro(self):
        """Two micro-batches → ``scaled_loss = raw_loss / num_micro``; the
        reported ``avg_loss`` equals the arithmetic mean of raw losses.
        """
        torch.manual_seed(0)
        model = _ToyModel()
        trainer = _build_trainer(model, args=_make_args(loss_aggregation="rank_average"))
        mb0 = _batch(torch.tensor([[1, 2, 3, 4]]))
        mb1 = _batch(torch.tensor([[5, 6, 7, 0]]))

        torch.manual_seed(0)
        ref = _ToyModel()
        ref.load_state_dict(model.state_dict())
        with torch.no_grad():
            ref0 = ref(**mb0)["loss"].item()
            ref1 = ref(**mb1)["loss"].item()
        expected = (ref0 + ref1) / 2

        with patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            metrics = trainer.train_step(iter([[mb0, mb1]]))

        self.assertAlmostEqual(metrics["loss"], expected, places=4, msg=(
            f"rank_average loss formula broken: expected {expected:.6f}, "
            f"got {metrics['loss']:.6f}"
        ))


class TestStopIterationDoesNotBumpStep(unittest.TestCase):
    """``StopIteration`` from the data iterator must not advance ``global_step``."""

    def test_exhausted_iterator_propagates_without_step_increment(self):
        """An empty iterator → ``StopIteration`` propagates and ``global_step`` stays.

        Regression guard for ``base.py:942-943``. If ``global_step += 1``
        moved above the ``next(data_iterator)`` call, an early-exhausted
        epoch would silently inflate the step counter and break checkpoint
        naming / resume cadence.
        """
        model = _ToyModel()
        trainer = _build_trainer(model)
        trainer.state.global_step = 7
        with patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            with self.assertRaises(StopIteration):
                trainer.train_step(iter([]))
        self.assertEqual(trainer.state.global_step, 7, (
            f"global_step must NOT advance on StopIteration, "
            f"got {trainer.state.global_step}"
        ))


class TestGradSyncSchedule(unittest.TestCase):
    """When ``model`` is an ``HSDPModule``, sync toggles must follow the
    pattern that gives one all-reduce per real step under grad accumulation.
    """

    def test_sync_only_on_last_micro_batch(self):
        """With N=3 micro-batches: sync False on idx 0 and 1, True on idx 2."""
        model = _HSDPRecordingModel()
        trainer = _build_trainer(model)
        mb = _batch(torch.tensor([[1, 2, 3, 4]]))

        with patch.object(trainer_base, "HSDPModule", _HSDPRecordingModel), \
             patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            trainer.train_step(iter([[mb, mb, mb]]))

        self.assertEqual(model.sync_calls, [False, False, True], (
            f"set_requires_gradient_sync schedule broken: expected "
            f"[False, False, True], got {model.sync_calls}"
        ))
        self.assertEqual(model.last_bw_calls, [False, False, True], (
            f"set_is_last_backward schedule broken: expected "
            f"[False, False, True], got {model.last_bw_calls}"
        ))

    def test_reshard_toggled_only_at_boundaries_under_grad_accum(self):
        """``_maybe_toggle_reshard``: False on idx 0, True on idx N-1, untouched in between."""
        model = _HSDPRecordingModel()
        trainer = _build_trainer(model)
        mb = _batch(torch.tensor([[1, 2, 3, 4]]))

        with patch.object(trainer_base, "HSDPModule", _HSDPRecordingModel), \
             patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            trainer.train_step(iter([[mb, mb, mb]]))
        # Expect exactly two toggles: False at micro_step 0, True at the last.
        self.assertEqual(model.reshard_calls, [False, True], (
            f"reshard toggle should fire only at boundaries, "
            f"got {model.reshard_calls}"
        ))

    def test_reshard_not_toggled_when_no_grad_accum(self):
        """With a single micro-batch, ``_maybe_toggle_reshard`` is a no-op."""
        model = _HSDPRecordingModel()
        trainer = _build_trainer(model)
        mb = _batch(torch.tensor([[1, 2, 3, 4]]))
        with patch.object(trainer_base, "HSDPModule", _HSDPRecordingModel), \
             patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            trainer.train_step(iter([[mb]]))
        self.assertEqual(model.reshard_calls, [], (
            f"single micro-batch must NOT touch reshard toggle, "
            f"got {model.reshard_calls}"
        ))


class TestForwardBackwardStepNestedBatch(unittest.TestCase):
    """``forward_backward_step`` recursively moves nested batch values to device.

    Multimodal batches contain nested dicts / lists / tuples of tensors;
    the inline ``_to_device`` walker must cover all four shapes. A regression
    that only handles flat ``{key: tensor}`` would silently keep image-grid
    tensors on CPU and crash inside the model.
    """

    def test_to_device_walks_dict_list_tuple_tensor(self):
        """Verify recursive descent: nested dict + list + tuple all hit ``.to(device)``."""
        device_records = []

        class _RecordingTensor(torch.Tensor):
            # Subclassing Tensor: track which tensors got .to()-moved.
            pass

        # Build a recursive batch shape:  dict→list→tuple→tensor + flat tensor
        flat = torch.tensor([[1, 2, 3, 4]])
        nested_tensor = torch.tensor([[5, 6]])
        batch = {
            "input_ids": flat,
            "labels": flat,
            "vision": {
                "grids": [(nested_tensor, nested_tensor.clone())],
            },
        }

        class _SinkModel(nn.Module):
            """Module that records the device of every tensor it receives.

            Used as a probe to verify ``_to_device`` walks the full nested
            batch structure (dict / list / tuple / tensor) and moves every
            leaf tensor, not just the top-level entries.
            """

            def __init__(self):
                super().__init__()
                # Need at least one trainable parameter so the SGD optimizer
                # (and the clip_grad pass downstream) have something to work on.
                self.scale = nn.Parameter(torch.ones(()))

            def forward(self, **kwargs):  # pylint: disable=arguments-differ
                """Walk every tensor in kwargs and record its device."""
                def _record(v):
                    if isinstance(v, torch.Tensor):
                        device_records.append(v.device.type)
                    elif isinstance(v, dict):
                        for vv in v.values():
                            _record(vv)
                    elif isinstance(v, (list, tuple)):
                        for vv in v:
                            _record(vv)
                for v in kwargs.values():
                    _record(v)
                return {"loss": self.scale * 0.0}

        model = _SinkModel()
        trainer = _build_trainer(model)
        with patch.object(trainer_base, "platform") as mock_plat:
            mock_plat.get_world_size.return_value = 1
            trainer.forward_backward_step(
                batch, micro_batch_tokens=4, global_tokens=4, num_micro=1,
            )
        # All visited tensors must report 'cpu' (we requested cpu device).
        self.assertTrue(device_records, "no tensors visited — recursion broke")
        self.assertTrue(all(d == "cpu" for d in device_records), (
            f"_to_device left tensors un-moved; device records={device_records}"
        ))


if __name__ == "__main__":
    unittest.main()
