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
"""HF checkpoint and delayed-initialization contracts for routed experts."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from safetensors.torch import save_file
import torch
from torch import nn
from transformers.conversion_mapping import get_checkpoint_conversion_mapping

from hyper_parallel.components.checkpoint.huggingface_checkpointer import HuggingFaceCheckpointer
from hyper_parallel.components.checkpoint.weight_verifier import verify_hf_weights
from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import (
    GroupedExpertsConfig, GroupedExpertsModule,
)
from hyper_parallel.models.deepseek_v41.adapter.conversion.checkpoint_mapping import (
    build_deepseek_v41_checkpoint_mapping,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)
from tests.common.mark_utils import arg_mark


class _PackedExperts(nn.Module):
    """Packed checkpoint source with the same routed forward signature as HF experts."""

    def __init__(self) -> None:
        """Declare canonical projections without initializing meta storage."""
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(12, 10, 6))
        self.down_proj = nn.Parameter(torch.empty(12, 6, 5))

    def forward(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the replacement call contract; checkpoint tests do not run expert math."""
        del topk_ids, topk_weights
        return hidden_states


def _model() -> nn.Module:
    """Build a packed source with enough experts to distinguish numeric sorting."""
    model = nn.Module()
    model.config = SimpleNamespace(num_hidden_layers=8)
    model.base_model_prefix = ""
    model.mlp = nn.Module()
    model.mlp.experts = _PackedExperts()
    return model


def _checkpoint_weights(format_name: str) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Create distinct gate/up/down values in a supported source checkpoint layout."""
    gate_up = torch.arange(720, dtype=torch.float32).reshape(12, 10, 6)
    down = torch.arange(360, dtype=torch.float32).reshape(12, 6, 5).neg()
    if format_name == "canonical":
        return {"mlp.experts.gate_up_proj": gate_up, "mlp.experts.down_proj": down}, gate_up, down
    if format_name == "qwen3_vl_moe":
        return {
            "mlp.experts.gate_up_proj": gate_up.transpose(-2, -1).contiguous(),
            "mlp.experts.down_proj": down.transpose(-2, -1).contiguous(),
        }, gate_up, down
    names = ("w1", "w3", "w2") if format_name == "deepseek_v41" else (
        "gate_proj", "up_proj", "down_proj",
    )
    tensors = {}
    for expert in reversed(range(12)):
        for name, value in zip(names, (gate_up[expert, :5], gate_up[expert, 5:], down[expert])):
            tensors[f"mlp.experts.{expert}.{name}.weight"] = value.contiguous()
    return tensors, gate_up, down


class TestRoutedGroupedExpertsCheckpoint(unittest.TestCase):
    """Use production HF mappings and both loaders without accelerator dependencies."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_replacement_owns_post_materialization_reset_marker(self) -> None:
        """Feature: Builder-owned delayed initialization.
        Description: Construct meta experts directly and through module replacement.
        Expectation: Only the replacement pipeline marks adopted expert weights for reset.
        """
        for ep_size in (1, 2):
            with self.subTest(ep_size=ep_size), torch.device("meta"):
                model = _model()
                direct = GroupedExperts(module=model.mlp.experts, context={"ep_size": ep_size, "num_layers": 8})
                self.assertFalse(hasattr(direct, "_hp_reset_after_materialization"))
                mapping = []
                plan = compile_module_replacements(model, [ModuleReplacementSpec(
                    match=("mlp.experts",), factory=GroupedExperts,
                )])
                apply_module_replacements(model, plan, weights_mapping=mapping, context={"ep_size": ep_size})
                self.assertTrue(getattr(model.mlp.experts, "_hp_reset_after_materialization", False))
                self.assertEqual(len(mapping), 2)
                self.assertTrue(all(transform.scope_prefix == "mlp.experts" for transform in mapping))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_projection_transforms_preserve_canonical_and_normalize_transposed_weights(self) -> None:
        """Feature: Packed expert checkpoint orientation.
        Description: Apply declared converters to canonical and input-major projections.
        Expectation: Canonical storage is preserved; transposed data restores exact linear weights.
        """
        module = GroupedExpertsModule(config=GroupedExpertsConfig(12, 6, 5, num_layers=1))
        for transform in module.make_transforms():
            name = transform.target_patterns[0]
            expected = getattr(module, name).detach().clone()
            for transposed in (False, True):
                value = expected.transpose(-2, -1).contiguous() if transposed else expected
                with self.subTest(parameter=name, transposed=transposed):
                    actual = transform.operations[0].convert(
                        {name: [value]}, transform.source_patterns, transform.target_patterns,
                        model=module, full_layer_name=name,
                    )[name]
                    self.assertTrue(torch.equal(actual, expected), f"expected={expected}, actual={actual}")
                    if not transposed:
                        self.assertIs(actual, value)
                    reverse = transform.operations[0].reverse_op.convert(
                        {name: [actual]}, transform.target_patterns, transform.source_patterns,
                        model=module, full_layer_name=name,
                    )[name]
                    self.assertTrue(torch.equal(reverse, expected), f"expected={expected}, actual={reverse}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
        card_mark="onecard", essential_mark="essential",
    )
    def test_common_hf_expert_formats_load_and_supported_formats_reverse_exactly(self) -> None:
        """Feature: Two-stage HF model and expert replacement conversion.
        Description: Load packed, individual-expert and legacy VL checkpoints through both HF loaders.
        Expectation: All formats load exactly; supported inverse mappings also recover the HF checkpoint.
        """
        formats = ("canonical", "qwen3_moe", "deepseek_v32", "qwen3_5_moe_text", "kimi_k25",
                   "deepseek_v41", "qwen3_vl_moe")
        for format_name in formats:
            tensors, gate_up, down = _checkpoint_weights(format_name)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                save_file(tensors, root / "model.safetensors")
                for loader in ("legacy", "dcp"):
                    for ep_size in (1, 2):
                        with self.subTest(format=format_name, loader=loader, ep_size=ep_size):
                            if format_name == "canonical":
                                mapping = []
                            elif format_name == "deepseek_v41":
                                mapping = build_deepseek_v41_checkpoint_mapping()
                            else:
                                mapping = get_checkpoint_conversion_mapping(format_name)
                            with torch.device("meta"):
                                model = _model()
                            plan = compile_module_replacements(model, [ModuleReplacementSpec(
                                match=("mlp.experts",), factory=GroupedExperts,
                            )])
                            apply_module_replacements(
                                model, plan, weights_mapping=mapping, context={"ep_size": ep_size},
                            )
                            model.to_empty(device="cpu")
                            report = HuggingFaceCheckpointer(loader=loader, weights_mapping=mapping).load(
                                directory, {"model": model}, strict_model=True,
                            )["load_report"]
                            self.assertEqual(report.missing_keys, ())
                            self.assertEqual(report.unexpected_keys, ())
                            self.assertTrue(torch.equal(model.mlp.experts.gate_up_proj, gate_up))
                            self.assertTrue(torch.equal(model.mlp.experts.down_proj, down))
                            # HF's legacy VL transpose keeps check_dims on reversal, so it cannot
                            # recover input-major checkpoint shapes even without module replacement.
                            if format_name != "qwen3_vl_moe":
                                result = verify_hf_weights(model, directory, str(root / "verification"))
                                self.assertTrue(result["exact_reference_match"])
                                self.assertEqual(result["incomplete_tensors"], 0)


if __name__ == "__main__":
    unittest.main()
