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
"""Tests for target-selected YAML resolution and typed CLI overrides."""

import dataclasses
import importlib.util
import sys
import tempfile
import textwrap
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from hyper_models.components.loss import CausalLMLoss
from hyper_models.components.optim import AdamW, LRScheduler
from hyper_models.config.manager import parse_training_args
from hyper_models.config.resolver import (
    ConfigResolutionError,
    _annotation_assignable,
    coerce_value,
    resolve_component,
    resolve_root,
)
from hyper_models.trainer.config import (
    AcceleratorConfig,
    DebugConfig,
    GradientCheckpointingConfig,
    MixedPrecisionConfig,
    TrainerConfig,
    TrainingConfig,
)
from hyper_parallel.trainer.config import ModelConfig


def _required_model_factory(identifier: str) -> ModelConfig:
    return ModelConfig(name=identifier)


def _untyped_model_factory():
    return ModelConfig(name="untyped")


def _lying_model_factory() -> ModelConfig:
    return TrainingConfig()


class _RuntimeObject:
    constructions = 0

    def __init__(self) -> None:
        """Record whether resolver called this runtime constructor."""

        type(self).constructions += 1


class _MyWarmup(LRScheduler):
    @dataclass
    class Config(LRScheduler.Config):
        warmup_steps: int = 10

    def __init__(self, config: "_MyWarmup.Config") -> None:
        """Store the custom scheduler configuration."""
        self.config = config


@dataclass
class _WithInitVar:
    seed: dataclasses.InitVar[int]
    derived: int = 0

    def __post_init__(self, seed: int) -> None:
        """Derive a stored field from the InitVar seed."""
        self.derived = seed * 2


class TestTrainingConfigResolution(unittest.TestCase):
    """A complete YAML resolves every declared top-level component type."""

    def setUp(self) -> None:
        """Own temporary YAML files for each test."""
        self._resources = ExitStack()
        self.addCleanup(self._resources.close)

    def _write_yaml(self, body: str) -> Path:
        directory = self._resources.enter_context(tempfile.TemporaryDirectory())
        path = Path(directory) / "config.yaml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_complete_yaml_and_cli_overrides(self):
        path = self._write_yaml(
            """
            model:
              _target_: hyper_parallel.trainer.config.ModelConfig
              name: qwen3_5
              weights_path: /models/Qwen3.5-0.8B-Base
            optimizer:
              _target_: hyper_models.components.optim.AdamW.Config
              lr: 0.0002
              weight_decay: 0.1
              betas: [0.8, 0.95]
              foreach: null
            lr_scheduler:
              _target_: hyper_models.components.optim.CosineWithWarmup.Config
              warmup_ratio: 0.05
              min_lr: 0.00001
            loss:
              _target_: hyper_models.components.loss.CausalLMLoss.Config
              ignore_index: -100
            training:
              _target_: hyper_models.trainer.config.TrainingConfig
              max_steps: 100
              global_batch_size: 8
              loss_aggregation: rank_average
            accelerator:
              _target_: hyper_models.trainer.config.AcceleratorConfig
              tp_size: 2
              dp_shard_size: 4
            mixed_precision:
              _target_: hyper_models.trainer.config.MixedPrecisionConfig
              enabled: true
            gradient_checkpointing:
              _target_: hyper_models.trainer.config.GradientCheckpointingConfig
              activation_checkpoint: full
            debug:
              _target_: hyper_models.trainer.config.DebugConfig
              check_nan_inf: true
            """
        )

        config = parse_training_args(
            [
                str(path),
                "--model.weights_path=/models/Qwen3.5-2B-Base",
                "--training.max_steps=200",
                "--accelerator.tp_size=4",
                "--optimizer.lr=0.0003",
                "--optimizer.foreach=null",
            ],
        )

        self.assertIsInstance(config, TrainerConfig)
        self.assertEqual(config.model.name, "qwen3_5")
        self.assertEqual(config.model.weights_path, "/models/Qwen3.5-2B-Base")
        self.assertIsInstance(config.optimizer, AdamW.Config)
        self.assertEqual(config.optimizer.betas, (0.8, 0.95))
        self.assertAlmostEqual(config.optimizer.lr, 0.0003)
        self.assertIsNone(config.optimizer.foreach)
        self.assertIsInstance(config.loss, CausalLMLoss.Config)
        self.assertEqual(config.training.max_steps, 200)
        self.assertEqual(config.training.loss_aggregation, "rank_average")
        self.assertIsInstance(config.accelerator, AcceleratorConfig)
        self.assertEqual(config.accelerator.tp_size, 4)
        self.assertEqual(config.accelerator.dp_shard_size, 4)
        self.assertIsInstance(
            config.gradient_checkpointing,
            GradientCheckpointingConfig,
        )
        self.assertIsInstance(config.mixed_precision, MixedPrecisionConfig)
        self.assertTrue(config.mixed_precision.enabled)
        self.assertIsInstance(config.debug, DebugConfig)
        self.assertTrue(config.debug.check_nan_inf)

    def test_typed_fields_reject_invalid_values(self):
        model = {
            "_target_": f"{__name__}._required_model_factory",
            "identifier": "test",
        }
        cases = (
            (
                "training",
                {
                    "_target_": "hyper_models.trainer.config.TrainingConfig",
                    "init_device": "metal",
                },
                r"\$\.training\.init_device.*expected one of",
            ),
            (
                "training",
                {
                    "_target_": "hyper_models.trainer.config.TrainingConfig",
                    "loss_aggregation": "tokens",
                },
                r"\$\.training\.loss_aggregation.*expected one of",
            ),
            (
                "accelerator",
                {
                    "_target_": "hyper_models.trainer.config.AcceleratorConfig",
                    "tp_size": "two",
                },
                r"\$\.accelerator\.tp_size.*expected int",
            ),
            (
                "gradient_checkpointing",
                {
                    "_target_": "hyper_models.trainer.config.GradientCheckpointingConfig",
                    "activation_checkpoint": "fulll",
                },
                r"\$\.gradient_checkpointing\.activation_checkpoint.*expected one of",
            ),
        )
        for field, component, error in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ConfigResolutionError, error):
                    resolve_root({"model": model, field: component})

    def test_unquoted_off_is_preserved_for_activation_checkpointing(self):
        path = self._write_yaml(
            """
            model:
              _target_: hyper_parallel.trainer.config.ModelConfig
              name: qwen3_5
            gradient_checkpointing:
              _target_: hyper_models.trainer.config.GradientCheckpointingConfig
              activation_checkpoint: off
            """
        )

        config = parse_training_args([str(path)])

        self.assertEqual(config.gradient_checkpointing.activation_checkpoint, "off")

    def test_parse_training_args_reads_config_path_and_overrides(self):
        path = self._write_yaml(
            """
            model:
              _target_: hyper_parallel.trainer.config.ModelConfig
              name: qwen3_5
            training:
              _target_: hyper_models.trainer.config.TrainingConfig
              max_steps: 100
            """
        )

        config = parse_training_args(
            [
                str(path),
                "--training.max_steps=200",
                "--model.weights_path=/models/Qwen3.5-0.8B-Base",
            ]
        )

        self.assertEqual(config.model.name, "qwen3_5")
        self.assertEqual(
            config.model.weights_path,
            "/models/Qwen3.5-0.8B-Base",
        )
        self.assertEqual(config.training.max_steps, 200)

    def test_cli_override_requires_equals_form(self):
        path = self._write_yaml(
            """
            model:
              _target_: hyper_parallel.trainer.config.ModelConfig
              name: qwen3_5
            """
        )

        with self.assertRaisesRegex(ConfigResolutionError, r"--field=value"):
            parse_training_args([str(path), "--model.name", "llama"])

    def test_root_has_no_target(self):
        config = resolve_root(
            {
                "model": {
                    "_target_": f"{__name__}._required_model_factory",
                    "identifier": "test",
                }
            }
        )
        self.assertEqual(config.model.name, "test")

    def test_root_target_is_rejected_as_unknown_field(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"unknown.*_target_"):
            resolve_root(
                {
                    "_target_": "hyper_models.trainer.config.TrainerConfig",
                    "model": {
                        "_target_": f"{__name__}._required_model_factory",
                        "identifier": "test",
                    },
                }
            )

    def test_unowned_compile_group_is_rejected(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"unknown.*compile"):
            resolve_root(
                {
                    "model": {
                        "_target_": f"{__name__}._required_model_factory",
                        "identifier": "test",
                    },
                    "compile": {
                        "_target_": "hyper_models.trainer.config.TrainingConfig"
                    },
                }
            )

    def test_missing_required_root_field_reports_root_path(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"\$: missing.*model"):
            resolve_root({})

    def test_component_without_target_reports_field_path(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"\$\.training.*_target_"):
            resolve_root(
                {
                    "model": {
                        "_target_": f"{__name__}._required_model_factory",
                        "identifier": "test",
                    },
                    "training": {"max_steps": 10},
                }
            )

    def test_target_import_and_callability_errors_include_field_path(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"\$\.model\._target_"):
            resolve_root({"model": {"_target_": "does.not.exist"}})
        with self.assertRaisesRegex(ConfigResolutionError, r"not callable"):
            resolve_root(
                {"model": {"_target_": "hyper_models.trainer.config.__doc__"}}
            )

    def test_unknown_missing_and_wrong_typed_arguments_fail_before_trainer(self):
        model = {
            "_target_": f"{__name__}._required_model_factory",
            "identifier": "test",
        }
        with self.assertRaisesRegex(ConfigResolutionError, r"max_stepz"):
            resolve_root(
                {
                    "model": model,
                    "training": {
                        "_target_": "hyper_models.trainer.config.TrainingConfig",
                        "max_stepz": 10,
                    },
                }
            )
        with self.assertRaisesRegex(ConfigResolutionError, r"missing.*identifier"):
            resolve_root(
                {"model": {"_target_": f"{__name__}._required_model_factory"}}
            )
        with self.assertRaisesRegex(ConfigResolutionError, r"training\.max_steps.*int"):
            resolve_root(
                {
                    "model": model,
                    "training": {
                        "_target_": "hyper_models.trainer.config.TrainingConfig",
                        "max_steps": "ten",
                    },
                }
            )

    def test_factory_return_contract_is_checked(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"return annotation"):
            resolve_root({"model": {"_target_": f"{__name__}._untyped_model_factory"}})
        with self.assertRaisesRegex(ConfigResolutionError, r"expected ModelConfig"):
            resolve_root({"model": {"_target_": f"{__name__}._lying_model_factory"}})

    def test_runtime_class_is_rejected_before_construction(self):
        _RuntimeObject.constructions = 0
        with self.assertRaisesRegex(ConfigResolutionError, r"expected TrainingConfig"):
            resolve_root(
                {
                    "model": {
                        "_target_": f"{__name__}._required_model_factory",
                        "identifier": "test",
                    },
                    "training": {"_target_": f"{__name__}._RuntimeObject"},
                }
            )
        self.assertEqual(_RuntimeObject.constructions, 0)

    def test_cli_unknown_type_and_unselected_component_errors(self):
        path = self._write_yaml(
            f"""
            model:
              _target_: {__name__}._required_model_factory
              identifier: test
            training:
              _target_: hyper_models.trainer.config.TrainingConfig
            """
        )

        with self.assertRaisesRegex(ConfigResolutionError, r"max_stepz.*max_steps"):
            parse_training_args([str(path), "--training.max_stepz=1"])
        with self.assertRaisesRegex(ConfigResolutionError, r"max_steps.*int"):
            parse_training_args([str(path), "--training.max_steps=not-an-int"])
        with self.assertRaisesRegex(ConfigResolutionError, r"optimizer.*not selected"):
            parse_training_args([str(path), "--optimizer.lr=0.1"])

    def test_coerce_value_handles_supported_container_and_union_types(self):
        self.assertEqual(
            coerce_value([0.8, 0.95], tuple[float, float], path="test.betas"),
            (0.8, 0.95),
        )
        self.assertEqual(
            coerce_value([1, 2], list[int], path="test.layers"),
            [1, 2],
        )
        self.assertIsNone(coerce_value(None, Optional[bool], path="test.foreach"))
        with self.assertRaisesRegex(ConfigResolutionError, r"expected bool"):
            coerce_value("yes", bool, path="test.enabled")

    def test_component_category_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"target returns CausalLMLoss\.Config, expected .*Optimizer\.Config",
        ):
            resolve_root(
                {
                    "model": {
                        "_target_": f"{__name__}._required_model_factory",
                        "identifier": "test",
                    },
                    "optimizer": {
                        "_target_": "hyper_models.components.loss.CausalLMLoss.Config"
                    },
                }
            )

    def test_new_scheduler_implementation_does_not_change_trainer_config(self):
        config = resolve_root(
            {
                "model": {
                    "_target_": f"{__name__}._required_model_factory",
                    "identifier": "test",
                },
                "lr_scheduler": {
                    "_target_": f"{__name__}._MyWarmup.Config",
                    "warmup_steps": 25,
                },
            }
        )

        self.assertIsInstance(config.lr_scheduler, _MyWarmup.Config)
        scheduler = config.lr_scheduler.build()
        self.assertIsInstance(scheduler, _MyWarmup)
        self.assertEqual(scheduler.config.warmup_steps, 25)

    def test_component_config_keeps_build_contract(self):
        # 设计文档 03 §9.3 新契约：build(model, ...) -> list[torch.optim.Optimizer]
        import torch

        optimizer_config = AdamW.Config(lr=0.1)
        optimizers = optimizer_config.build(torch.nn.Linear(2, 2))
        self.assertIsInstance(optimizers, list)
        self.assertIsInstance(optimizers[0], torch.optim.AdamW)
        self.assertEqual(optimizers[0].param_groups[0]["lr"], 0.1)


class TestCoercionEdgeCases(unittest.TestCase):
    """Regression tests for scalar, Literal, union, and InitVar coercion."""

    def test_cli_scientific_notation_float_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    model:
                      _target_: hyper_parallel.trainer.config.ModelConfig
                      name: qwen3_5
                    optimizer:
                      _target_: hyper_models.components.optim.AdamW.Config
                    """
                ),
                encoding="utf-8",
            )
            config = parse_training_args([str(path), "--optimizer.lr=1e-4"])

        self.assertEqual(config.optimizer.lr, 1e-4)
        self.assertIsInstance(config.optimizer.lr, float)

    def test_literal_yaml11_bool_words(self):
        self.assertEqual(
            coerce_value(False, Literal["off", "none", "full"], path="test.ac"),
            "off",
        )
        self.assertEqual(
            coerce_value(True, Literal["on", "off"], path="test.mode"),
            "on",
        )
        with self.assertRaisesRegex(ConfigResolutionError, r"expected one of"):
            coerce_value(False, Literal["none", "full"], path="test.ac")

    def test_annotation_assignable_accepts_identical_unions(self):
        self.assertTrue(
            _annotation_assignable(Union[int, str], Union[int, str])
        )
        self.assertTrue(
            _annotation_assignable(int, Union[int, str])
        )
        self.assertFalse(
            _annotation_assignable(Union[int, str], int)
        )

    def test_initvar_field_is_accepted(self):
        result = resolve_component(
            {"_target_": f"{__name__}._WithInitVar", "seed": 21},
            expected_type=object,
            path="$.component",
        )
        self.assertEqual(result.derived, 42)


class TestPlainClassTarget(unittest.TestCase):
    """Plain (non-dataclass) classes resolve constructor hints from __init__."""

    def test_future_annotations_module(self):
        with tempfile.TemporaryDirectory() as directory:
            module_path = Path(directory) / "pep563_target.py"
            module_path.write_text(
                textwrap.dedent(
                    '''
                    from __future__ import annotations


                    class PlainTarget:
                        def __init__(self, name: str, count: int = 1) -> None:
                            self.name = name
                            self.count = count
                    '''
                ),
                encoding="utf-8",
            )
            spec = importlib.util.spec_from_file_location("pep563_target", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["pep563_target"] = module
            self.addCleanup(sys.modules.pop, "pep563_target")
            spec.loader.exec_module(module)

            result = resolve_component(
                {"_target_": "pep563_target.PlainTarget", "name": "x", "count": 2},
                expected_type=object,
                path="$.component",
            )

        self.assertEqual((result.name, result.count), ("x", 2))


if __name__ == "__main__":
    unittest.main()
