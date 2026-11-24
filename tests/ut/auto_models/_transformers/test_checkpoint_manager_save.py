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
"""Unit tests for Hugging Face checkpoint export sources."""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from transformers import PretrainedConfig, PreTrainedModel

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager
from hyper_parallel.core.fully_shard.api import _extend_module_with_hsdp_interface


class ExportableModel(PreTrainedModel):
    """Small CPU model recording HF's metadata mutation without device setup."""

    config_class = PretrainedConfig

    def __init__(self, model_type: str) -> None:
        """Initialize a single CPU parameter and an isolated export record."""
        config = PretrainedConfig(architectures=["ExportableModel"])
        config.model_type = model_type
        super().__init__(config)
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.exports = []
        self.fail_export = False

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        """Emulate HF overwriting architectures before serializing configuration."""
        self.config.architectures = [type(self).__name__.removeprefix("FSDP")]
        self.exports.append((save_directory, self.config, kwargs))
        if self.fail_export:
            raise OSError("export failed")


class TestCheckpointManagerSave(unittest.TestCase):
    """Verify DCP-backed export does not read live training parameters."""

    def test_hsdp_export_uses_original_class_without_mutating_training_model(self) -> None:
        """Feature: HF checkpoint export.

        Description: Save a wrapped model.
        Expectation: only the export view is unwrapped.
        """
        model = ExportableModel("qwen3_5_text")
        _extend_module_with_hsdp_interface(model)
        training_class, training_config, weight = type(model), model.config, model.weight

        CheckpointManager(model).save_pretrained("/unused")

        self.assertEqual(model.exports[0][1].architectures, ["ExportableModel"])
        self.assertIs(type(model), training_class)
        self.assertIs(model.config, training_config)
        self.assertIs(model.weight, weight)
        self.assertEqual(model.config.architectures, ["ExportableModel"])
        self.assertIsNot(model.exports[0][1], model.config)

    def test_export_failure_preserves_live_class_config_and_conversions(self) -> None:
        """Feature: HF checkpoint export.

        Description: HF writing fails.
        Expectation: no training metadata changes.
        """
        model = ExportableModel("qwen3_5_text")
        _extend_module_with_hsdp_interface(model)
        model.fail_export = True
        model._weight_conversions = [object()]
        original_class, original_config = type(model), model.config
        conversions = model._weight_conversions

        with self.assertRaisesRegex(OSError, "export failed"):
            CheckpointManager(model).save_pretrained("/unused")

        self.assertIs(type(model), original_class)
        self.assertIs(model.config, original_config)
        self.assertIs(model._weight_conversions, conversions)
        self.assertEqual(model.config.architectures, ["ExportableModel"])

    def test_qwen_text_keeps_native_namespace_and_fused_experts(self) -> None:
        """Feature: HF checkpoint export.

        Description: Export dense/MoE text.
        Expectation: do not undo source VLM prefix removal.
        """
        for model_type in ("qwen3_5_text", "qwen3_5_moe_text"):
            with self.subTest(model_type=model_type):
                model = ExportableModel(model_type)
                model._weight_conversions = [object()]
                CheckpointManager(model).save_pretrained("/unused")
                self.assertFalse(model.exports[0][2]["save_original_format"])
                torch.testing.assert_close(model.exports[0][2]["state_dict"]["weight"], model.weight)

    def test_qwen_text_still_reverses_replacement_layout(self) -> None:
        """Feature: HF checkpoint export.

        Description: Grouped experts plus prefix history.
        Expectation: reverse only replacement layout.
        """
        model = ExportableModel("qwen3_5_moe_text")
        replacements, base = [object()], [object()]
        model._hp_used_replacement_weight_conversions = replacements
        model._hp_used_base_weight_conversions = base
        original_mapping = model._weight_conversions = base + replacements

        def reverse_layout(export_model: ExportableModel, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Check that export reverses only the replacement transforms."""
            self.assertIs(export_model._weight_conversions, replacements)
            return {"weight": state["weight"] + 1}

        with patch("hyper_parallel.models.qwen3_5.adapter.checkpoint.revert_weight_conversion",
                   side_effect=reverse_layout) as reverse:
            CheckpointManager(model).save_pretrained("/unused")

        self.assertEqual(reverse.call_count, 1)
        self.assertIs(model._weight_conversions, original_mapping)
        self.assertFalse(model.exports[0][2]["save_original_format"])
        torch.testing.assert_close(model.exports[0][2]["state_dict"]["weight"], model.weight + 1)

    def test_explicit_original_format_policy_is_preserved(self) -> None:
        """Feature: HF checkpoint export.

        Description: Explicit original-format export for text, VLM, and other models.
        Expectation: honor the caller's conversion policy.
        """
        for model_type in ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_moe", "llama"):
            with self.subTest(model_type=model_type):
                model = ExportableModel(model_type)
                with patch.object(CheckpointManager, "_gather_full_state_dict",
                                  return_value={"model.language_model.weight": model.weight}):
                    CheckpointManager(model).save_pretrained("/unused", save_original_format=True)
                self.assertTrue(model.exports[0][2]["save_original_format"])
                self.assertIn("model.language_model.weight", model.exports[0][2]["state_dict"])

    def test_unregistered_model_preserves_explicit_export_policy(self) -> None:
        """Feature: HF checkpoint export.

        Description: No adapter exists.
        Expectation: honor the caller's format flag.
        """
        model = ExportableModel("unregistered_export_test")
        CheckpointManager(model).save_pretrained("/unused", save_original_format=False)
        self.assertFalse(model.exports[0][2]["save_original_format"])

    def test_qwen_conversion_failure_preserves_metadata(self) -> None:
        """Feature: HF checkpoint export.

        Description: Adapter conversion fails.
        Expectation: live metadata stays intact.
        """
        model = ExportableModel("qwen3_5_text")
        original_mapping = model._weight_conversions = [object()]
        model._hp_used_replacement_weight_conversions = [object()]
        with patch("hyper_parallel.models.qwen3_5.adapter.checkpoint.revert_weight_conversion",
                   side_effect=ValueError("conversion failed")):
            with self.assertRaisesRegex(ValueError, "conversion failed"):
                CheckpointManager(model).save_pretrained("/unused", save_original_format=False)
        self.assertIs(model._weight_conversions, original_mapping)
        self.assertEqual(model.exports, [])

    def test_live_export_disables_autograd_for_parameter_gather(self) -> None:
        """Feature: Live HF export without autograd.

        Description: Export a live DTensor while the caller enables gradients.
        Expectation: Gathering disables gradients and restores caller grad mode.
        """
        parameter = torch.ones(3, requires_grad=True)
        distributed = MagicMock(spec=DTensor)

        def _full_tensor():
            """Verify the export boundary before emulating collective output."""
            self.assertFalse(torch.is_grad_enabled())
            return parameter.clone()

        distributed.full_tensor.side_effect = _full_tensor
        model = MagicMock()
        model.state_dict.return_value = {"weight": distributed}
        with torch.enable_grad():
            state = CheckpointManager(model)._gather_full_state_dict(keep_state_dict=True)
            self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(state["weight"].requires_grad)
        self.assertIsNone(state["weight"].grad_fn)
        model.state_dict.assert_called_once_with(keep_vars=True)


if __name__ == "__main__":
    unittest.main()
