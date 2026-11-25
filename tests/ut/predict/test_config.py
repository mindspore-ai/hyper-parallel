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
"""Prediction configuration and task-host integration contracts."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from hyper_parallel.infer import GenerationOptions
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.config.tasks import PredictionConfig
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.runtime.prediction import run_prediction
from hyper_parallel.trainer.runtime.run_mode import validate_run_config, validate_run_capabilities
from tests.ut.predict.config_helpers import prediction_config


class TestPredictionConfig(unittest.TestCase):
    """Serializable generation fields are accepted; runtime injection through YAML is rejected."""

    @staticmethod
    def raw() -> dict:
        """Return the minimal serializable prediction task configuration."""
        return {"run_mode": "predict",
                "model": {"_target_": "tests.ut.predict.config_helpers.build_model_stub"},
                "prediction": {"input": "input.txt", "output_file": "output.jsonl"}}

    def test_generation_resolves_and_round_trips(self):
        raw = self.raw()
        raw["prediction"]["generation"] = {"eos_token_id": [2, 3], "max_new_tokens": 4}
        config = resolve_config(raw)
        self.assertIsNone(config.optimizer)
        self.assertIsNone(config.dataset)
        self.assertIsInstance(config.prediction.generation, GenerationOptions)
        validate_run_config(config)
        self.assertEqual(config.prediction.to_dict()["generation"]["eos_token_id"], [2, 3])

    def test_runtime_and_multi_sample_fields_are_not_yaml_options(self):
        for field in ("model_adapter", "device", "rng", "logits_group", "num_return_sequences", "n"):
            raw = self.raw()
            raw["prediction"]["generation"] = {field: 2}
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_config(raw)

    def test_invalid_prediction_ranges(self):
        for values in ({"batch_size": 0}, {"batch_size": True}, {"max_batch_tokens": -1},
                       {"overwrite": "yes"}, {"generation": {"max_new_tokens": -1}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PredictionConfig(**values)

    def test_training_accelerations_and_missing_paths_are_rejected(self):
        config = prediction_config(run_mode="predict", optimizer=None)
        for change in ({"compile": SimpleNamespace(enabled=True)}, {"hyper_offload": SimpleNamespace(enable=True)},
                       {"activation_checkpoint": SimpleNamespace(mode="full")},
                       {"prediction": PredictionConfig(input="i.txt")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_run_config(prediction_config(**{**vars(config), **change}))

    def test_unsupported_topologies_fail_before_model_execution(self):
        config = prediction_config(run_mode="predict")
        for dimension in ("tp_size", "cp_size", "ep_size", "pp_size", "dp_shard_size", "edp_shard_size"):
            with self.subTest(dimension=dimension), self.assertRaises(NotImplementedError):
                validate_run_capabilities(config, SimpleNamespace(**{dimension: 2}))
        validate_run_capabilities(config, SimpleNamespace(dp_size=8, dp_shard_size=1))

    def test_model_generation_capability_is_explicit(self):
        config = prediction_config(run_mode="predict")
        trainer = SimpleNamespace(config=config, mesh=SimpleNamespace(), model=object())
        adapter = ModelAdapterSpec("CustomModel", "custom")
        self.assertFalse(adapter.supports_generation)
        with patch("hyper_parallel.trainer.runtime.prediction.resolve_model_adapter", return_value=adapter):
            with self.assertRaisesRegex(NotImplementedError, "supports_generation"):
                run_prediction(trainer)

    def test_predict_dispatches_without_training(self):
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = SimpleNamespace(config=prediction_config(run_mode="predict"))
        summary = Mock()
        with (patch.object(trainer, "predict", return_value=summary) as prediction,
              patch.object(trainer, "train") as train):
            trainer.run()
        prediction.assert_called_once_with()
        train.assert_not_called()
