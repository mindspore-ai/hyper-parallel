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
"""Keep V4.1 native vision and router names independent of upstream V4 rules."""
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers.core_model_loading import WeightRenaming

from hyper_parallel.components.checkpoint.huggingface_checkpointer import HuggingFaceCheckpointer
from hyper_parallel.models.deepseek_v41.adapter.conversion.checkpoint_mapping import (
    build_deepseek_v41_checkpoint_mapping,
)


def _native_model():
    """Build exactly the ten native targets affected by generalized V4 rules."""
    model = torch.nn.Module()
    model.base_model_prefix = "model"
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.model.layers[0]
    layer.mlp = torch.nn.Module()
    layer.mlp.gate = torch.nn.Module()
    layer.mlp.gate.register_parameter("bias", torch.nn.Parameter(torch.zeros(4)))
    vision = model.model.vision = torch.nn.Module()
    vision.blocks = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for block in vision.blocks:
        block.attn = torch.nn.Module()
        block.attn.wo = torch.nn.Linear(4, 4)
        block.attn.wqkv = torch.nn.Linear(4, 12)
    vision.norm = torch.nn.RMSNorm(4)
    return model


class TestCheckpointMappingCompatibility(unittest.TestCase):
    """Changes to the V4 family mapping cannot rename V4.1-owned parameters."""

    def test_ten_native_parameters_load_despite_generalized_v4_rules(self):
        """Feature: V4.1 conversion compatibility across Transformers versions.

        Description: Supply broad upstream V4 renames and load native vision/router tensors.
        Expectation: All ten targets load exactly through both legacy and DCP.
        """
        upstream = [
            WeightRenaming(r"\.attn\.", ".self_attn."),
            WeightRenaming(r"\.norm\.", ".kv_norm."),
            WeightRenaming(r"\.gate\.bias$", ".gate.e_score_correction_bias"),
        ]
        template = _native_model()
        source = {}
        for name, value in template.named_parameters():
            source_name = name.removeprefix("model.").replace("layers.0.mlp.", "layers.0.ffn.")
            source[source_name] = torch.arange(value.numel()).reshape(value.shape).float() + 1
        self.assertEqual(len(source), 10)
        with tempfile.TemporaryDirectory() as directory:
            save_file(source, str(Path(directory) / "model.safetensors"))
            for loader in ("legacy", "dcp"):
                with self.subTest(loader=loader):
                    # Transformers can replace its lazy parent without reattaching cached child modules.
                    with patch.object(
                        import_module("transformers.conversion_mapping"),
                        "get_checkpoint_conversion_mapping",
                        return_value=upstream
                    ) as family_rules:
                        mapping = build_deepseek_v41_checkpoint_mapping()
                    family_rules.assert_not_called()
                    model = _native_model()
                    report = HuggingFaceCheckpointer(loader=loader, weights_mapping=mapping).load(
                        directory, {"model": model}, strict_model=True)["load_report"]
                    self.assertEqual(report.missing_keys, ())
                    self.assertEqual(report.unexpected_keys, ())
                    for name, value in model.named_parameters():
                        source_name = name.removeprefix("model.").replace("layers.0.mlp.", "layers.0.ffn.")
                        torch.testing.assert_close(value, source[source_name], rtol=0, atol=0)
