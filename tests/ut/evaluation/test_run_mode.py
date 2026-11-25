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
"""Unit tests for run-mode validation and assembly planning."""

import unittest
from types import SimpleNamespace

from hyper_parallel.trainer.config.tasks import EvaluationConfig
from hyper_parallel.trainer.config.trainer import TrainerConfig
from hyper_parallel.trainer.runtime.run_mode import (
    BuildStep,
    get_build_steps,
    validate_run_capabilities,
    validate_run_config,
)


def _config(**overrides):
    """Build a minimal TrainerConfig-like object for validation checks."""
    defaults = {
        "run_mode": "eval",
        "optimizer": object(),
        "dataset": object(),
        "dataloader": object(),
        "compile": SimpleNamespace(enabled=False),
        "activation_checkpoint": None,
        "hyper_offload": SimpleNamespace(enable=False),
        "high_availability": SimpleNamespace(worker_recovery=SimpleNamespace(enable=False)),
        "training": SimpleNamespace(eval_iters=0),
        "evaluation": EvaluationConfig(metrics=[object()], max_samples=16),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestTrainerConfigCompatibility(unittest.TestCase):
    """Standalone options preserve the positional API used by training callers."""

    def test_legacy_model_optimizer_scheduler_positions(self):
        """An existing three-argument constructor must still select training."""
        model, scheduler = object(), object()
        optimizer = SimpleNamespace(fp32_main_params=False)
        config = TrainerConfig(model, optimizer, scheduler)
        self.assertIs(config.model, model)
        self.assertIs(config.optimizer, optimizer)
        self.assertIs(config.lr_scheduler, scheduler)
        self.assertEqual(config.run_mode, "train")


class TestBuildSteps(unittest.TestCase):
    """Assembly plans declare exactly what each run mode executes."""

    def test_training_plan_keeps_every_step(self):
        """Training still builds the optimizer, callbacks, and iterations."""
        steps = get_build_steps("train")
        for required in (BuildStep.OPTIMIZER, BuildStep.CALLBACKS, BuildStep.TRAIN_ITERS):
            self.assertIn(required, steps)

    def test_eval_skips_training_components(self):
        """Standalone evaluation builds no optimizer or callback manager."""
        steps = get_build_steps("eval")
        for skipped in (
                BuildStep.OPTIMIZER, BuildStep.LR_SCHEDULER, BuildStep.CALLBACKS, BuildStep.GRAPH_COMPILER):
            self.assertNotIn(skipped, steps)

    def test_eval_restores_weights_but_has_no_training_loop(self):
        """Eval restores a checkpoint and never resolves training iterations."""
        steps = get_build_steps("eval")
        self.assertIn(BuildStep.MODEL_RESTORE, steps)
        self.assertNotIn(BuildStep.TRAIN_ITERS, steps)

    def test_unknown_mode_is_rejected(self):
        """An unknown mode fails instead of silently planning training."""
        with self.assertRaises(ValueError):
            get_build_steps("inference")


class TestValidateRunConfig(unittest.TestCase):
    """Pure configuration checks run before distributed initialization."""

    def test_train_requires_an_optimizer(self):
        """A training run without an optimizer is rejected."""
        with self.assertRaises(ValueError):
            validate_run_config(_config(run_mode="train", optimizer=None))

    def test_train_accepts_a_complete_configuration(self):
        """The default training shape passes unchanged."""
        validate_run_config(_config(run_mode="train"))

    def test_eval_requires_metrics(self):
        """An evaluation without any metric cannot report anything."""
        with self.assertRaises(ValueError):
            validate_run_config(
                _config(evaluation=EvaluationConfig(metrics=[], max_samples=8))
            )

    def test_eval_rejects_generation_mode(self):
        """Generation-based evaluation has not been delivered."""
        with self.assertRaises(NotImplementedError):
            validate_run_config(
                _config(evaluation=EvaluationConfig(mode="generation", metrics=[object()]))
            )

    def test_eval_requires_a_dataset_and_dataloader(self):
        """Loss evaluation needs a batch source."""
        with self.assertRaises(ValueError):
            validate_run_config(_config(dataset=None))
        with self.assertRaises(ValueError):
            validate_run_config(_config(dataloader=None))

    def test_eval_rejects_compile_and_recompute(self):
        """Eager, forward-only evaluation rejects training-only acceleration."""
        with self.assertRaises(ValueError):
            validate_run_config(_config(compile=SimpleNamespace(enabled=True)))
        with self.assertRaises(ValueError):
            validate_run_config(
                _config(activation_checkpoint=SimpleNamespace(mode="full"))
            )
        with self.assertRaisesRegex(ValueError, "hyper_offload"):
            validate_run_config(_config(hyper_offload=SimpleNamespace(enable=True)))

    def test_eval_rejects_training_recovery(self):
        """Evaluation has no optimizer or training progress for worker recovery."""
        availability = SimpleNamespace(worker_recovery=SimpleNamespace(enable=True))
        with self.assertRaisesRegex(ValueError, "worker_recovery"):
            validate_run_config(_config(high_availability=availability))

    def test_eval_accepts_missing_or_disabled_activation_checkpoint(self):
        """The new optional checkpoint configuration is valid when disabled."""
        for checkpoint in (None, SimpleNamespace(mode="off")):
            with self.subTest(checkpoint=checkpoint):
                validate_run_config(_config(activation_checkpoint=checkpoint))

    def test_eval_accepts_the_whole_finite_split(self):
        """``max_samples=None`` means the whole selected split and must pass.

        Only an unbounded source needs an explicit cap, and that is decided from
        the resolved Dataset rather than from the training configuration.
        """
        validate_run_config(
            _config(evaluation=EvaluationConfig(metrics=[object()], max_samples=None))
        )


class TestValidateRunCapabilities(unittest.TestCase):
    """Topology checks that need the resolved mesh."""

    def _mesh(self, **sizes):
        """Build a mesh stub with the requested parallel widths."""
        defaults = {"pp_size": 1, "tp_size": 1, "cp_size": 1, "ep_size": 1}
        defaults.update(sizes)
        return SimpleNamespace(**defaults)

    def test_dense_dp_is_accepted(self):
        """P0 supports dense data parallelism."""
        validate_run_capabilities(_config(), self._mesh())

    def test_any_model_parallel_dimension_is_rejected(self):
        """TP/CP/EP/PP change the statistics domain or need new schedules."""
        for dimension in ("pp_size", "tp_size", "cp_size", "ep_size"):
            with self.assertRaises(NotImplementedError, msg=dimension):
                validate_run_capabilities(_config(), self._mesh(**{dimension: 2}))

    def test_training_is_not_capability_checked(self):
        """The capability gate applies to standalone evaluation only."""
        validate_run_capabilities(_config(run_mode="train"), self._mesh(pp_size=8))


if __name__ == "__main__":
    unittest.main()
