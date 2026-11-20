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
"""CPU checks for model-driven VeRL weight conversion without rollout transport."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers.core_model_loading import Concatenate, MergeModulelist

from hyper_parallel.components.checkpoint.conversion_ops import InterleaveQKV
from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.integration.verl.engine import weight_export


def _model() -> nn.Module:
    model = nn.Module()
    model.config = SimpleNamespace(model_type="arbitrary_family")
    model._hp_model_adapter_spec = None
    model._weight_conversions = []
    return model


def _export_weights(weights, model):
    """Exercise per-tensor export with CPU weights and no accelerator transport."""
    with patch.object(weight_export, "get_model_state_dict", return_value=dict(weights)), \
            patch.object(weight_export, "hsdp_sync_stream"), \
            patch.object(weight_export, "get_device_name", return_value="cpu"), \
            patch.object(weight_export, "get_device_id", return_value=0):
        yield from weight_export.get_per_tensor_param(model)


class TestWeightConversion(unittest.TestCase):
    """Reuse registered rules for any model family without mutating live metadata."""

    def test_identity_parameter_without_family_allowlist(self) -> None:
        """An ordinary parameter needs no model-specific adapter or packed buffer."""
        model, tensor = _model(), torch.arange(6).reshape(2, 3).float()
        name, actual = next(_export_weights(iter((("ordinary.weight", tensor),)), model))
        self.assertEqual(name, "ordinary.weight", f"output name={name}, expected=ordinary.weight")
        torch.testing.assert_close(actual, tensor)

    def test_registered_qkv_layout_is_reversed(self) -> None:
        """A generic interleave rule restores distinct query/key/value projections."""
        model = _model()
        query = torch.arange(24).reshape(8, 3).float()
        key = torch.arange(12).reshape(4, 3).float() + 100
        value = key + 100
        fused = torch.cat((query[:4], key[:2], value[:2], query[4:], key[2:], value[2:]))
        replacements = [WeightConverter(
            source_patterns=["query", "key", "value"], target_patterns="packed",
            operations=[InterleaveQKV(2, 2, 2, 2, source_is_fused=False)],
        )]
        model._hp_used_replacement_weight_conversions = replacements
        original_mapping, original_config = model._weight_conversions, model.config
        outputs = dict(_export_weights(iter((("packed", fused),)), model))
        self.assertEqual(set(outputs), {"query", "key", "value"}, f"unexpected projections: {set(outputs)}")
        for name, expected in (("query", query), ("key", key), ("value", value)):
            torch.testing.assert_close(outputs[name], expected)
        self.assertIs(model._weight_conversions, original_mapping)
        self.assertIs(model._hp_used_replacement_weight_conversions, replacements)
        self.assertIs(model.config, original_config)

    def test_expert_transpose_and_base_rules_are_composed(self) -> None:
        """Replacement reversal precedes the existing batched-expert checkpoint mapping."""
        model = _model()
        gate = torch.arange(24).reshape(2, 4, 3).float()
        up = gate + 100
        checkpoint_layout = torch.cat((gate, up), dim=1)
        training_layout = checkpoint_layout.transpose(1, 2).contiguous()
        model._hp_used_replacement_weight_conversions = [WeightConverter(
            source_patterns="experts.gate_up_proj", target_patterns="experts.gate_up_proj",
            operations=[Transpose(1, 2)],
        )]
        model._hp_used_base_weight_conversions = [WeightConverter(
            source_patterns=["experts.*.gate_proj.weight", "experts.*.up_proj.weight"],
            target_patterns="experts.gate_up_proj", operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        )]
        outputs = dict(_export_weights(iter((("experts.gate_up_proj", training_layout),)), model))
        self.assertEqual(len(outputs), 4, f"expert output count={len(outputs)}, expected=4")
        for expert in range(2):
            torch.testing.assert_close(outputs[f"experts.{expert}.gate_proj.weight"], gate[expert])
            torch.testing.assert_close(outputs[f"experts.{expert}.up_proj.weight"], up[expert])

    def test_model_export_provider_receives_original_format_request(self) -> None:
        """Export hooks receive the source-format request without mutating live config."""
        model, tensor = _model(), torch.ones(2)

        def export(view, weights, save_original_format):
            self.assertIs(save_original_format, True)
            self.assertIs(view, model)
            return {"exported.weight": weights["weight"]}, True

        model._hp_model_adapter_spec = SimpleNamespace(checkpoint_export=export)
        with patch(
            "hyper_parallel.integration.verl.engine.weight_export.revert_weight_conversion",
            side_effect=lambda view, weights: weights,
        ) as revert:
            outputs = dict(_export_weights(iter((("weight", tensor),)), model))
        revert.assert_not_called()
        torch.testing.assert_close(outputs["exported.weight"], tensor)
        self.assertEqual(model.config.model_type, "arbitrary_family")

    def test_conversion_failure_preserves_live_metadata(self) -> None:
        """A failed reverse conversion must not overwrite the model's mapping."""
        model = _model()
        model._hp_used_replacement_weight_conversions = [object()]
        mapping, config = model._weight_conversions, model.config
        with patch("hyper_parallel.integration.verl.engine.weight_export.revert_weight_conversion",
                   side_effect=ValueError("invalid conversion")):
            with self.assertRaisesRegex(ValueError, "invalid conversion"):
                list(_export_weights(iter((("weight", torch.ones(2)),)), model))
        self.assertIs(model._weight_conversions, mapping)
        self.assertIs(model.config, config)
