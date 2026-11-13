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

from hyper_parallel.trainer.config.evaluation import EvaluationConfig
from hyper_parallel.trainer.runtime.run_mode import (
    BuildStep,
    build_steps_for,
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
        "activation_checkpoint": SimpleNamespace(mode="off"),
        "activation_swap": "none",
        "training": SimpleNamespace(eval_iters=0),
        "evaluation": EvaluationConfig(metrics=[object()], max_samples=16),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildSteps(unittest.TestCase):
    """Assembly plans declare exactly what each run mode executes."""

    def test_training_plan_keeps_every_step(self):
        """Training still builds the optimizer, callbacks, and iterations."""
        steps = build_steps_for("train")
        for required in (BuildStep.OPTIMIZER, BuildStep.CALLBACKS, BuildStep.TRAIN_ITERS):
            self.assertIn(required, steps)

    def test_standalone_modes_skip_training_components(self):
        """Neither standalone mode builds an optimizer or callback manager."""
        for mode in ("eval", "predict"):
            steps = build_steps_for(mode)
            for skipped in (
                BuildStep.OPTIMIZER,
                BuildStep.LR_SCHEDULER,
                BuildStep.CALLBACKS,
                BuildStep.GRAPH_COMPILER,
            ):
                self.assertNotIn(skipped, steps, msg=f"{mode}:{skipped}")

    def test_predict_skips_the_dataset_pipeline(self):
        """Predict reads prompt files, so no dataset or dataloader is built."""
        steps = build_steps_for("predict")
        for skipped in (BuildStep.DATASET, BuildStep.COLLATE, BuildStep.DATALOADER):
            self.assertNotIn(skipped, steps)

    def test_eval_restores_weights_but_has_no_training_loop(self):
        """Eval restores a checkpoint and never resolves training iterations."""
        steps = build_steps_for("eval")
        self.assertIn(BuildStep.MODEL_RESTORE, steps)
        self.assertNotIn(BuildStep.TRAIN_ITERS, steps)

    def test_unknown_mode_is_rejected(self):
        """An unknown mode fails instead of silently planning training."""
        with self.assertRaises(ValueError):
            build_steps_for("inference")


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
        with self.assertRaises(ValueError):
            validate_run_config(_config(activation_swap="attention"))

    def test_eval_requires_a_finite_data_range(self):
        """Without an explicit count, a bounded training-derived range is needed."""
        with self.assertRaises(ValueError):
            validate_run_config(
                _config(evaluation=EvaluationConfig(metrics=[object()], max_samples=None))
            )

    def test_predict_is_declared_but_not_implemented(self):
        """The predict shell fails with a concrete capability message."""
        with self.assertRaises(NotImplementedError):
            validate_run_config(_config(run_mode="predict"))


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
