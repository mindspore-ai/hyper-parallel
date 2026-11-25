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
"""Prediction initialization plans and early rejection of training services."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.trainer.config.trainer import TrainerConfig
from hyper_parallel.trainer.runtime.run_mode import (
    BuildStep,
    get_build_steps,
    validate_run_capabilities,
    validate_run_config,
)
from hyper_parallel.trainer.text_trainer import TextTrainer
from tests.ut.predict.config_helpers import prediction_config


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
    """Prediction initializes weights and assets while preserving the training plan."""

    def test_training_plan_keeps_training_components(self):
        """Training still builds its optimizer, callbacks, and iteration plan."""
        steps = get_build_steps("train")
        for required in (BuildStep.OPTIMIZER, BuildStep.CALLBACKS, BuildStep.TRAIN_ITERS):
            self.assertIn(required, steps)

    def test_prediction_skips_training_and_dataset_components(self):
        """File-based prediction has no training loop, loss, or dataset pipeline."""
        steps = get_build_steps("predict")
        for skipped in (
                BuildStep.OPTIMIZER, BuildStep.LR_SCHEDULER, BuildStep.TRAINING_CONTEXT,
                BuildStep.GRAPH_COMPILER, BuildStep.CALLBACKS, BuildStep.TRAIN_ITERS,
                BuildStep.LOSS, BuildStep.DATASET, BuildStep.COLLATE, BuildStep.DATALOADER):
            self.assertNotIn(skipped, steps)
        self.assertIn(BuildStep.MODEL_RESTORE, steps)
        self.assertIn(BuildStep.MODEL_ASSETS, steps)

    def test_other_modes_are_rejected(self):
        """Only training and prediction are exposed by this feature."""
        for mode in ("inference", "eval"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                get_build_steps(mode)


class TestStandaloneValidation(unittest.TestCase):
    """Unsupported configuration fails before constructing distributed services."""

    def test_training_still_requires_an_optimizer(self):
        """Prediction's optional optimizer does not weaken the training contract."""
        with self.assertRaisesRegex(ValueError, "requires an 'optimizer'"):
            validate_run_config(prediction_config(run_mode="train"))
        validate_run_config(prediction_config(run_mode="train", optimizer=object()))

    def test_training_topologies_remain_unrestricted(self):
        """Prediction capability limits must not reject a model-parallel training run."""
        validate_run_capabilities(
            prediction_config(run_mode="train", optimizer=object()),
            SimpleNamespace(tp_size=2, cp_size=2, ep_size=2, pp_size=2),
        )

    def test_worker_recovery_is_rejected_before_session_initialization(self):
        """Prediction cannot start training recovery services before validation."""
        config = prediction_config(
            high_availability=SimpleNamespace(worker_recovery=SimpleNamespace(enable=True)),
        )
        with patch("hyper_parallel.trainer.runtime.worker_recovery.RecoverySession") as session:
            with self.assertRaisesRegex(ValueError, "training worker recovery"):
                TextTrainer(config)
        session.assert_not_called()
