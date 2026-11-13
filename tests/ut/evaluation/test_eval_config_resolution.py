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
"""YAML resolution of the run-mode, evaluation, and optional-optimizer sections."""

import unittest

from hyper_parallel.evaluation import LossMetric, PerplexityMetric
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, resolve_config
from hyper_parallel.trainer.config.target import Target

_MODEL = {"_target_": f"{__name__}._build_model"}


def _build_model(name: str = "stub") -> str:
    """Stand-in model target with an inspectable signature."""
    return name


def _raw(**overrides):
    """Return a minimal YAML root mapping for an evaluation run."""
    raw = {
        "model": dict(_MODEL),
        "run_mode": "eval",
        "evaluation": {
            "batch_size": 2,
            "max_samples": 8,
            "metrics": [
                {"_target_": "hyper_parallel.evaluation.metrics.LossMetric"},
                {
                    "_target_": "hyper_parallel.evaluation.metrics.PerplexityMetric",
                    "statistic": "foundation",
                },
            ],
        },
    }
    raw.update(overrides)
    return raw


class TestEvalConfigResolution(unittest.TestCase):
    """Standalone sections resolve without an optimizer and build real metrics."""

    def test_eval_config_resolves_without_an_optimizer(self):
        """A standalone task needs no optimizer section."""
        config = resolve_config(_raw())
        self.assertEqual(config.run_mode, "eval")
        self.assertIsNone(config.optimizer)
        self.assertEqual(config.evaluation.batch_size, 2)
        self.assertEqual(config.evaluation.max_samples, 8)

    def test_metrics_resolve_to_unbuilt_targets_and_build_in_order(self):
        """``evaluation.metrics`` is a list of deferred targets."""
        config = resolve_config(_raw())
        self.assertTrue(all(isinstance(metric, Target) for metric in config.evaluation.metrics))
        built = [metric.build() for metric in config.evaluation.metrics]
        self.assertIsInstance(built[0], LossMetric)
        self.assertIsInstance(built[1], PerplexityMetric)

    def test_default_run_mode_is_train_and_requires_an_optimizer(self):
        """Old training YAML without run_mode keeps meaning training."""
        raw = _raw()
        del raw["run_mode"]
        with self.assertRaisesRegex(ValueError, "requires an 'optimizer'"):
            resolve_config(raw)

    def test_invalid_run_mode_is_rejected(self):
        """An unknown mode fails at resolution rather than at assembly."""
        with self.assertRaises(ConfigResolutionError):
            resolve_config(_raw(run_mode="inference"))

    def test_non_positive_ranges_are_rejected(self):
        """Zero is not a way to say 'unbounded'; null is."""
        for field, value in (("batch_size", 0), ("max_samples", 0), ("max_batches", 0)):
            raw = _raw()
            raw["evaluation"][field] = value
            # Field validation raises ValueError, as optimizer.swap does.
            with self.assertRaisesRegex(ValueError, f"evaluation.{field}", msg=field):
                resolve_config(raw)

    def test_metrics_must_be_a_list_of_targets(self):
        """A mapping where a list is expected is a clear error."""
        raw = _raw()
        raw["evaluation"]["metrics"] = {"_target_": "hyper_parallel.evaluation.metrics.LossMetric"}
        with self.assertRaises(ConfigResolutionError):
            resolve_config(raw)


if __name__ == "__main__":
    unittest.main()
