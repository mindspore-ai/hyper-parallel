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
"""Regression coverage for native V4.1 router names in both HF loaders."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn
from transformers.utils.hub import get_checkpoint_shard_files

from hyper_parallel.components.checkpoint.huggingface_checkpointer import HuggingFaceCheckpointer
from hyper_parallel.models.deepseek_v41.adapter.conversion.checkpoint_mapping import (
    build_deepseek_v41_checkpoint_mapping,
)

from hyper_parallel.models.deepseek_v41.adapter.validation import cropped_model


class _RouterModel(nn.Module):
    """Minimal router model for checkpoint mapping regression tests."""

    base_model_prefix = "model"

    def __init__(self) -> None:
        """Create router parameters and aliases exercised by the HF loaders."""
        super().__init__()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(4, 2)
        layer.mlp.gate.register_parameter("bias_vl", nn.Parameter(torch.zeros(2)))
        layer.input_layernorm = nn.RMSNorm(4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])


class TestCheckpointMapping(unittest.TestCase):
    """Load nonzero official router biases without applying the V4-only rename."""

    def test_native_router_biases_and_shared_norm_mapping(self) -> None:
        """Legacy and DCP both load native names with strict completeness."""
        source = {
            "layers.0.ffn.gate.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "layers.0.ffn.gate.bias": torch.tensor([1.25, -3.5]),
            "layers.0.ffn.gate.bias_vl": torch.tensor([7.25, 9.5]),
            "layers.0.attn_norm.weight": torch.tensor([2.0, 3.0, 4.0, 5.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(source, Path(directory) / "model.safetensors")
            for loader in ("legacy", "dcp"):
                with self.subTest(loader=loader):
                    model = _RouterModel()
                    report = HuggingFaceCheckpointer(
                        loader=loader, weights_mapping=build_deepseek_v41_checkpoint_mapping(),
                    ).load(directory, {"model": model}, strict_model=True)["load_report"]
                    self.assertEqual(report.missing_keys, (), f"missing={report.missing_keys}, expected=()")
                    self.assertEqual(report.unexpected_keys, (), f"unexpected={report.unexpected_keys}, expected=()")
                    gate = model.model.layers[0].mlp.gate
                    torch.testing.assert_close(gate.weight, source["layers.0.ffn.gate.weight"], rtol=0, atol=0)
                    torch.testing.assert_close(gate.bias, source["layers.0.ffn.gate.bias"], rtol=0, atol=0)
                    torch.testing.assert_close(gate.bias_vl, source["layers.0.ffn.gate.bias_vl"], rtol=0, atol=0)
                    torch.testing.assert_close(model.model.layers[0].input_layernorm.weight,
                                               source["layers.0.attn_norm.weight"], rtol=0, atol=0)


    def test_runtime_crop_index_is_readable_by_transformers(self) -> None:
        """Filtered runtime indexes retain metadata required by the initial HF load."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v41", "text_config": {}}), encoding="utf-8")
            save_file({"norm.weight": torch.ones(4, dtype=torch.bfloat16),
                       "layers.0.ffn.gate.weight": torch.ones(2, 4)}, source / "weights.safetensors")
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "norm.weight": "weights.safetensors", "layers.0.ffn.gate.weight": "weights.safetensors",
                "layers.39.ffn.gate.weight": "unused-missing.safetensors",
            }}), encoding="utf-8")
            observed_views = []

            def load(view: str, **kwargs: object) -> str:
                """Inspect the selected index while its temporary view exists."""
                del kwargs
                observed_views.append(Path(view))
                files, metadata = get_checkpoint_shard_files(
                    view, str(Path(view) / "model.safetensors.index.json"), local_files_only=True)
                self.assertEqual(metadata["total_size"], 40,
                                 f"total_size={metadata['total_size']}, expected=40")
                self.assertEqual(len(files), 1, f"files={files}, expected one selected shard")
                self.assertEqual(Path(files[0]).resolve(), source / "weights.safetensors",
                                 f"file={files[0]}, expected source weights")
                self.assertNotIn("layers.39.ffn.gate.weight", metadata["weight_map"],
                                 f"unused key found in {metadata['weight_map']}")
                return "loaded"

            config = unittest.mock.MagicMock()
            config.v41_vision_enabled = True
            config.v41_vision_num_hidden_layers = 2
            config.num_hidden_layers = 1
            with patch.object(cropped_model, "_validate_pretrained_dimensions"), patch.object(
                    cropped_model.HyperAutoModelForCausalLM, "from_pretrained", side_effect=load):
                result = cropped_model._load_validation_hf_weights(config, str(source))
            self.assertEqual(result, "loaded", f"result={result}, expected=loaded")
            self.assertFalse(observed_views[0].exists(), f"temporary view remains: {observed_views[0]}")
