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
"""Tests for deferred target resolution and typed CLI overrides."""

import json
import tempfile
import textwrap
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal, Optional

from hyper_models._transformers import HyperAutoModelForCausalLM
from hyper_models.components.data import (
    DataLoader,
    DummyDataset,
    IdentityDataTransform,
    MakeMicroBatchCollator,
)
from hyper_models.components.optim import AdamW, cosine_with_warmup
from hyper_models.config.manager import parse_training_args
from hyper_models.config.resolver import (
    ConfigResolutionError,
    coerce_value,
    resolve_component,
    resolve_root,
)
from hyper_models.trainer.config import Target, TrainerConfig, TrainingConfig


OPTIMIZER_TARGET = "hyper_models.components.optim.optimizer.optimizer.AdamW"
MODEL_TARGET = (
    "hyper_models._transformers.HyperAutoModelForCausalLM.from_pretrained"
)
SCHEDULER_TARGET = (
    "hyper_models.components.optim.lr_scheduler.lr_scheduler.cosine_with_warmup"
)


def _model_target(*, name: str = "model"):
    return {"kind": "model", "name": name}


def _tokenizer_target(*, name: str = "tokenizer"):
    return {"kind": "tokenizer", "name": name}


def _function_target(*, value: int = 1):
    return value


def _runtime_target(*, runtime: object, value: int = 1):
    return runtime, value


def _strict_target(*, count: int = 1):
    return count


def _variadic_target(*, count: int = 1, **kwargs):
    return count, kwargs


class _RuntimeClass:
    constructions = 0

    def __init__(self, *, value: int = 1) -> None:
        type(self).constructions += 1
        self.value = value


def _root(**changes: object) -> dict[str, object]:
    root = {
        "model": {"_target_": f"{__name__}._model_target"},
        "tokenizer": {"_target_": f"{__name__}._tokenizer_target"},
        "optimizer": {"_target_": OPTIMIZER_TARGET},
        "lr_scheduler": {"_target_": SCHEDULER_TARGET},
    }
    root.update(changes)
    return root


class TestTargetResolution(unittest.TestCase):
    """Runtime components resolve to ``Target`` without being constructed."""

    def test_class_and_function_targets_are_deferred(self):
        config = resolve_root(_root())

        self.assertIsInstance(config, TrainerConfig)
        self.assertIsInstance(config.optimizer, Target)
        self.assertIs(config.optimizer._target_, AdamW)
        self.assertIsInstance(config.lr_scheduler, Target)
        self.assertIs(config.lr_scheduler._target_, cosine_with_warmup)

    def test_resolver_does_not_invoke_class_target(self):
        _RuntimeClass.constructions = 0

        target = resolve_component(
            {"_target_": f"{__name__}._RuntimeClass", "value": 3},
            expected_type=Target[Any],
            path="$.component",
        )

        self.assertEqual(_RuntimeClass.constructions, 0)
        result = target.build()
        self.assertEqual(_RuntimeClass.constructions, 1)
        self.assertEqual(result.value, 3)

    def test_function_without_return_annotation_is_allowed(self):
        target = resolve_component(
            {"_target_": f"{__name__}._function_target", "value": 4},
            expected_type=Target[Any],
            path="$.component",
        )

        self.assertEqual(target.build(), 4)

    def test_generic_parameter_is_not_a_runtime_result_check(self):
        config = resolve_root(_root())

        self.assertEqual(
            config.model.build(),
            {"kind": "model", "name": "model"},
        )

    def test_bound_model_classmethod_is_resolved_directly(self):
        target = resolve_component(
            {
                "_target_": MODEL_TARGET,
                "pretrained_model_name_or_path": "./model",
                "force_hf": True,
            },
            expected_type=Target[Any],
            path="$.model",
        )

        self.assertIs(
            target._target_.__func__,
            HyperAutoModelForCausalLM.from_pretrained.__func__,
        )
        self.assertEqual(target.pretrained_model_name_or_path, "./model")
        self.assertTrue(target.force_hf)

    def test_data_class_targets_are_deferred_and_serializable(self):
        config = resolve_root(
            _root(
                dataset={
                    "_target_": "hyper_models.components.data.datasets.DummyDataset",
                    "num_samples": 16,
                    "seq_len": 8,
                    "vocab_size": 32,
                },
                data_transform={
                    "_target_": (
                        "hyper_models.components.data.identity_transform."
                        "IdentityDataTransform"
                    ),
                },
                collate_fn={
                    "_target_": (
                        "hyper_models.components.data.data_collator."
                        "MakeMicroBatchCollator"
                    ),
                },
                dataloader={
                    "_target_": "hyper_models.components.data.dataloader.DataLoader",
                },
            )
        )

        self.assertIs(config.data_transform._target_, IdentityDataTransform)
        self.assertIs(config.dataset._target_, DummyDataset)
        self.assertIs(config.collate_fn._target_, MakeMicroBatchCollator)
        self.assertIs(config.dataloader._target_, DataLoader)
        serialized = config.to_dict()
        json.dumps(serialized)
        self.assertEqual(
            serialized["collate_fn"]["internal_data_collator"],
            "torch.utils.data._utils.collate.default_collate",
        )

    def test_annotated_arguments_are_coerced_and_defaults_are_exposed(self):
        config = resolve_root(
            _root(
                optimizer={
                    "_target_": OPTIMIZER_TARGET,
                    "betas": [0.8, 0.95],
                }
            )
        )

        self.assertEqual(config.optimizer.betas, (0.8, 0.95))
        self.assertEqual(config.optimizer.lr, 1e-4)
        self.assertEqual(config.optimizer.eps, 1e-8)
        self.assertIsNone(config.optimizer.foreach)
        with self.assertRaises(AttributeError):
            _ = config.optimizer.model

    def test_gradient_clipping_is_not_an_optimizer_argument(self):
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"unexpected keyword argument 'max_grad_norm'",
        ):
            resolve_root(
                _root(
                    optimizer={
                        "_target_": OPTIMIZER_TARGET,
                        "max_grad_norm": 1.0,
                    }
                )
            )

    def test_unknown_argument_is_rejected_without_var_kwargs(self):
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"unexpected keyword argument 'extra'",
        ):
            resolve_component(
                {
                    "_target_": f"{__name__}._strict_target",
                    "extra": "kept only by variadic targets",
                },
                expected_type=Target[Any],
                path="$.component",
            )

    def test_extra_argument_is_preserved_with_var_kwargs(self):
        target = resolve_component(
            {
                "_target_": f"{__name__}._variadic_target",
                "extra": {"nested": True},
            },
            expected_type=Target[Any],
            path="$.component",
        )

        self.assertEqual(
            target.build(runtime="injected"),
            (1, {"extra": {"nested": True}, "runtime": "injected"}),
        )

    def test_runtime_argument_is_optional_during_resolution_but_required_at_build(self):
        target = resolve_component(
            {"_target_": f"{__name__}._runtime_target", "value": 2},
            expected_type=Target[Any],
            path="$.component",
        )

        with self.assertRaisesRegex(
            TypeError,
            r"missing .*required keyword-only argument: 'runtime'",
        ):
            target.build()
        sentinel = object()
        self.assertEqual(target.build(runtime=sentinel), (sentinel, 2))

    def test_runtime_arguments_override_configured_values(self):
        target = resolve_component(
            {"_target_": f"{__name__}._function_target", "value": 2},
            expected_type=Target[Any],
            path="$.component",
        )

        self.assertEqual(target.build(value=7), 7)

    def test_unused_runtime_argument_is_ignored(self):
        target = resolve_component(
            {"_target_": f"{__name__}._strict_target", "count": 2},
            expected_type=Target[Any],
            path="$.component",
        )

        self.assertEqual(target.build(device_mesh=object()), 2)

    def test_to_dict_preserves_original_target_path(self):
        config = resolve_root(
            _root(
                optimizer={
                    "_target_": OPTIMIZER_TARGET,
                    "betas": [0.8, 0.95],
                }
            )
        )

        serialized = config.optimizer.to_dict()
        self.assertEqual(serialized["_target_"], OPTIMIZER_TARGET)
        self.assertEqual(serialized["betas"], [0.8, 0.95])


class TestPureDataclassResolution(unittest.TestCase):
    """Pure parameter sections are ordinary dataclass mappings."""

    def test_dataclass_section_does_not_require_target(self):
        training = resolve_component(
            {"max_steps": 20, "max_grad_norm": 0.5},
            expected_type=TrainingConfig,
            path="$.training",
        )

        self.assertIsInstance(training, TrainingConfig)
        self.assertEqual(training.max_steps, 20)
        self.assertEqual(training.max_grad_norm, 0.5)

    def test_dataclass_section_rejects_target_key(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"unknown.*_target_"):
            resolve_component(
                {
                    "_target_": "hyper_models.trainer.config.TrainingConfig",
                    "max_steps": 20,
                },
                expected_type=TrainingConfig,
                path="$.training",
            )

    def test_root_uses_dataclass_defaults(self):
        config = resolve_root(
            _root(training={"max_steps": 8, "max_grad_norm": 0.25})
        )

        self.assertEqual(config.training.max_steps, 8)
        self.assertEqual(config.training.max_grad_norm, 0.25)
        self.assertEqual(config.accelerator.tp_size, 1)

    def test_training_max_steps_defaults_to_dataset_derived(self):
        config = resolve_root(_root())

        self.assertIsNone(config.training.max_steps)


class TestTypedOverrides(unittest.TestCase):
    """Dotted CLI overrides update selected target kwargs and dataclass fields."""

    def setUp(self) -> None:
        self._resources = ExitStack()
        self.addCleanup(self._resources.close)

    def _write_yaml(self) -> Path:
        directory = self._resources.enter_context(tempfile.TemporaryDirectory())
        path = Path(directory) / "config.yaml"
        path.write_text(
            textwrap.dedent(
                f"""
                model:
                  _target_: {__name__}._model_target
                tokenizer:
                  _target_: {__name__}._tokenizer_target
                optimizer:
                  _target_: {OPTIMIZER_TARGET}
                  lr: 0.0002
                lr_scheduler:
                  _target_: {SCHEDULER_TARGET}
                training:
                  max_steps: 10
                  max_grad_norm: 1.0
                """
            ),
            encoding="utf-8",
        )
        return path

    def test_target_and_dataclass_overrides_are_typed(self):
        config = parse_training_args(
            [
                str(self._write_yaml()),
                "--optimizer.lr=1e-4",
                "--optimizer.betas=[0.8, 0.95]",
                "--training.max_grad_norm=0.5",
            ]
        )

        self.assertEqual(config.optimizer.lr, 1e-4)
        self.assertIsInstance(config.optimizer.lr, float)
        self.assertEqual(config.optimizer.betas, (0.8, 0.95))
        self.assertEqual(config.training.max_grad_norm, 0.5)

    def test_target_selection_cannot_be_changed_by_override(self):
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"changing _target_ through an override is not supported",
        ):
            parse_training_args(
                [
                    str(self._write_yaml()),
                    f"--optimizer._target_={SCHEDULER_TARGET}",
                ]
            )

    def test_unknown_target_argument_has_close_match(self):
        with self.assertRaisesRegex(
            ConfigResolutionError,
            r"unknown target argument 'lrr'.*did you mean 'lr'",
        ):
            parse_training_args(
                [str(self._write_yaml()), "--optimizer.lrr=0.1"]
            )

    def test_override_requires_equals_form(self):
        with self.assertRaisesRegex(ConfigResolutionError, r"--field=value"):
            parse_training_args(
                [str(self._write_yaml()), "--optimizer.lr", "0.1"]
            )


class TestCoercionEdgeCases(unittest.TestCase):
    """Keep scalar, tuple, Optional, and YAML 1.1 Literal coercion stable."""

    def test_supported_container_and_optional_types(self):
        self.assertEqual(
            coerce_value([0.8, 0.95], tuple[float, float], path="test.betas"),
            (0.8, 0.95),
        )
        self.assertEqual(
            coerce_value([1, 2], list[int], path="test.layers"),
            [1, 2],
        )
        self.assertIsNone(coerce_value(None, Optional[bool], path="test.foreach"))

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


if __name__ == "__main__":
    unittest.main()
