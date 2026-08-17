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
"""Unit tests for the trainer's pipeline-parallel helpers.

Covers the hardware-agnostic surface of ``BaseTrainer``'s ``pp>1`` path: the
``pp_micro_batch_num`` config knob and the ``_pp_concat_micro_batches`` static
helper that rebuilds the global batch from the grad-accum group before handing
it to the pipeline schedule. The schedule-driven ``_pp_train_step`` itself runs
collectives and is exercised by the accuracy STs / trainer PP run.
"""
import os
import types
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config import AcceleratorConfig


def _validation_trainer(ac_mode, pp_applies_ac):
    """A bare trainer carrying just what ``_validate_pp_runtime_options`` reads."""
    trainer = object.__new__(BaseTrainer)
    trainer.args = types.SimpleNamespace(
        train=types.SimpleNamespace(
            optimizer=types.SimpleNamespace(loss_aggregation="token_weighted"),
            accelerator=types.SimpleNamespace(pp_micro_batch_num=2),
            gradient_checkpointing=types.SimpleNamespace(activation_checkpoint=ac_mode),
            checkpoint=types.SimpleNamespace(save_hf_weights=False),
            micro_batch_size=1,
            global_batch_size=2,
        ),
    )
    trainer.spec = types.SimpleNamespace(
        name="stub_model", pp_applies_activation_checkpoint=pp_applies_ac)
    return trainer


class TestPpActivationCheckpointGuard(unittest.TestCase):
    """``_validate_pp_runtime_options`` rejects AC for models that ignore it."""

    def test_rejects_ac_when_pipelining_fn_does_not_apply_it(self):
        """
        Feature: activation_checkpoint under pp>1 must fail loudly, not no-op.
        Description: the PP path bypasses ``parallelize_fn``, so a model whose
            ``pipelining_fn`` does not checkpoint the stage it builds would
            silently train without recompute at full activation memory.
        Expectation: NotImplementedError naming the model.
        """
        trainer = _validation_trainer("full", pp_applies_ac=False)
        with self.assertRaises(NotImplementedError) as ctx:
            trainer._validate_pp_runtime_options(types.SimpleNamespace(dp_size=1))
        self.assertIn("stub_model", str(ctx.exception))

    def test_allows_ac_when_the_model_declares_support(self):
        """
        Feature: models whose PP stage builder applies AC are not blocked.
        Description: ``pp_applies_activation_checkpoint=True`` declares that
            ``pipelining_fn`` checkpoints the stage, so the setting is honoured.
        Expectation: validation returns the micro-batch count, no raise.
        """
        trainer = _validation_trainer("full", pp_applies_ac=True)
        self.assertEqual(
            trainer._validate_pp_runtime_options(types.SimpleNamespace(dp_size=1)), 2)

    def test_ac_off_is_allowed_for_every_model(self):
        """
        Feature: the guard only fires when AC is actually requested.
        Description: with AC off there is nothing to silently drop.
        Expectation: no raise even when the model does not apply AC.
        """
        trainer = _validation_trainer("none", pp_applies_ac=False)
        self.assertEqual(
            trainer._validate_pp_runtime_options(types.SimpleNamespace(dp_size=1)), 2)


class TestPipelineConfig(unittest.TestCase):
    """``AcceleratorConfig.pp_micro_batch_num`` knob."""

    def test_pp_micro_batch_num_defaults_to_one(self):
        """
        Feature: ``pp_micro_batch_num`` defaults to a single micro-batch.
        Description: when unset, the pipeline schedule must not split the
            global batch, so the default has to be ``1``.
        Expectation: ``AcceleratorConfig().pp_micro_batch_num == 1``.
        """
        self.assertEqual(
            AcceleratorConfig().pp_micro_batch_num, 1,
            f"expected pp_micro_batch_num default 1, "
            f"got {AcceleratorConfig().pp_micro_batch_num}",
        )


class TestPpConcatMicroBatches(unittest.TestCase):
    """``BaseTrainer._pp_concat_micro_batches`` global-batch rebuild."""

    def test_single_micro_batch_is_copied(self):
        """
        Feature: a one-element grad-accum group passes through unchanged.
        Description: with ``grad_accum == 1`` the schedule already receives
            the full batch, so the helper returns the same tensors in a
            fresh dict (no concat).
        Expectation: identical tensor values for every key.
        """
        batch = {"input_ids": torch.arange(6).view(1, 6), "labels": torch.arange(6).view(1, 6)}
        merged = BaseTrainer._pp_concat_micro_batches([batch])
        self.assertEqual(set(merged.keys()), {"input_ids", "labels"})
        self.assertTrue(torch.equal(merged["input_ids"], batch["input_ids"]))

    def test_multiple_micro_batches_concatenated_on_dim0(self):
        """
        Feature: grad-accum micro-batches are concatenated into one batch.
        Description: under PP the schedule owns micro-batching, so the
            trainer rebuilds the global batch by concatenating the
            grad-accum group along dim 0; non-tensor values are taken from
            the first micro-batch.
        Expectation: stacked rows in order, non-tensor passthrough.
        """
        mb0 = {"input_ids": torch.zeros(1, 4), "labels": torch.zeros(1, 4), "tag": "a"}
        mb1 = {"input_ids": torch.ones(1, 4), "labels": torch.ones(1, 4), "tag": "b"}
        merged = BaseTrainer._pp_concat_micro_batches([mb0, mb1])
        self.assertEqual(tuple(merged["input_ids"].shape), (2, 4))
        self.assertTrue(torch.equal(merged["input_ids"][0], torch.zeros(4)))
        self.assertTrue(torch.equal(merged["input_ids"][1], torch.ones(4)))
        self.assertEqual(
            merged["tag"], "a",
            f"expected non-tensor passthrough from the first micro-batch, got {merged['tag']}",
        )

    def test_variable_length_micro_batches_are_rejected(self):
        """
        Feature: differing-length micro-batches are rejected, not silently merged.
        Description: the pipeline runs one fused sum-CE backward over the whole
            grad-accum group, which only matches the trainer's token_weighted
            single-card gradient when every micro-batch shares the sequence
            length. A len-3 + len-5 group therefore raises a clear error
            (directing the user to pad to a fixed ``max_seq_len``) rather than
            mis-normalizing.
        Expectation: ``NotImplementedError`` mentioning ``uniform``.
        """
        mb0 = {"input_ids": torch.ones(1, 3, dtype=torch.long), "labels": torch.ones(1, 3, dtype=torch.long)}
        mb1 = {"input_ids": torch.full((1, 5), 7, dtype=torch.long),
               "labels": torch.full((1, 5), 7, dtype=torch.long)}
        with self.assertRaises(NotImplementedError) as ctx:
            BaseTrainer._pp_concat_micro_batches([mb0, mb1])
        self.assertIn("uniform", str(ctx.exception))


class TestShardMicroBatchesForCp(unittest.TestCase):
    """``BaseTrainer._shard_micro_batches_for_cp`` context-parallel input split.

    The key property is the cross-rank boundary next-token target: the shift is
    applied to the **full** sequence before slicing, so the last token of an
    interior CP rank keeps the first token of the next rank's slice as its
    target (instead of being dropped to ``-100``).
    """

    @staticmethod
    def _make_trainer(cp_size: int, cp_rank: int) -> BaseTrainer:
        """Build a trainer stub carrying just the CP mesh fields."""
        from types import SimpleNamespace  # pylint: disable=C0415
        from unittest.mock import MagicMock  # pylint: disable=C0415
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.parallel_dims = SimpleNamespace(cp=cp_size)
        cp_mesh = MagicMock()
        cp_mesh.get_local_rank.return_value = cp_rank
        mesh = MagicMock()
        mesh.__getitem__.side_effect = lambda key: cp_mesh
        trainer.mesh = mesh
        return trainer

    def test_cp1_is_noop(self):
        """``cp == 1`` returns the micro-batches unchanged (identity)."""
        trainer = self._make_trainer(1, 0)
        micro_batches = [{"input_ids": torch.arange(8).view(1, 8)}]
        result = trainer._shard_micro_batches_for_cp(micro_batches)
        assert result is micro_batches, "cp=1 must be a no-op returning the same list"

    def test_cp2_slices_and_preserves_boundary_target(self):
        """
        Feature: each CP rank gets its sequence slice + a pre-shifted target
            slice that preserves the cross-rank boundary next-token target.
        Description: with ``seq=8, cp=2`` the full shift of ``[0..7]`` is
            ``[1..7, -100]``; rank 0 takes input ``[0,1,2,3]`` / target
            ``[1,2,3,4]`` (token 3's target 4 lives in rank 1's slice) and
            rank 1 takes input ``[4,5,6,7]`` / target ``[5,6,7,-100]``.
        Expectation: exact slices, ``_hp_labels_are_shifted=True`` and this
            rank's global ``position_ids`` slice on each rank.
        """
        ids = torch.arange(8).view(1, 8)
        rank0 = self._make_trainer(2, 0)._shard_micro_batches_for_cp(
            [{"input_ids": ids, "labels": ids.clone()}],
        )[0]
        assert torch.equal(rank0["input_ids"], torch.tensor([[0, 1, 2, 3]])), rank0["input_ids"]
        assert torch.equal(rank0["labels"], torch.tensor([[1, 2, 3, 4]])), rank0["labels"]
        assert rank0["_hp_labels_are_shifted"] is True
        assert torch.equal(rank0["position_ids"], torch.tensor([[0, 1, 2, 3]])), rank0["position_ids"]

        rank1 = self._make_trainer(2, 1)._shard_micro_batches_for_cp(
            [{"input_ids": ids, "labels": ids.clone()}],
        )[0]
        assert torch.equal(rank1["input_ids"], torch.tensor([[4, 5, 6, 7]])), rank1["input_ids"]
        assert torch.equal(rank1["labels"], torch.tensor([[5, 6, 7, -100]])), rank1["labels"]
        assert torch.equal(rank1["position_ids"], torch.tensor([[4, 5, 6, 7]])), rank1["position_ids"]

    def test_rejects_indivisible_sequence(self):
        """A sequence length not divisible by ``cp`` raises ``ValueError``."""
        trainer = self._make_trainer(2, 0)
        with self.assertRaises(ValueError):
            trainer._shard_micro_batches_for_cp([{"input_ids": torch.arange(5).view(1, 5)}])


if __name__ == "__main__":
    unittest.main()
