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
"""Check reverse conversion, live local storage, CPU tables and strict coverage."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn
from transformers.core_model_loading import Concatenate, MergeModulelist

from hyper_parallel.components.checkpoint import weight_verifier as verifier
from hyper_parallel.components.checkpoint.conversion_ops import ConcatenateWithSections
from hyper_parallel.components.checkpoint.huggingface_checkpointer import HuggingFaceCheckpointer
from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter, WeightRenaming
from hyper_parallel.models.deepseek_v41.adapter.conversion.checkpoint_mapping import (
    build_deepseek_v41_checkpoint_mapping,
)
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import DeepseekV41HostState
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.external_state import CheckpointTensorShard


def _model(weight: torch.Tensor) -> nn.Module:
    """Construct a finalized local parameter with an explicit identity mapping."""
    model = nn.Module()
    model.config = SimpleNamespace()
    model.weight = nn.Parameter(weight.clone())
    model._weight_conversions = []
    return model


class TestWeightVerifier(unittest.TestCase):
    """Fail on corruption and missing coverage without changing the model."""

    def test_checks_real_loader_output_using_recorded_inverse(self) -> None:
        """Feature: Loader integration.

        Description: Load a transposed tensor using the real HF checkpointer.
        Expectation: The checker reads finalized storage and uses recorded export metadata.
        """
        source = torch.arange(6).float().reshape(2, 3)
        model = _model(torch.zeros((3, 2)))
        mapping = [WeightConverter("weight", "weight", operations=[Transpose()])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": source}, root / "model.safetensors")
            HuggingFaceCheckpointer(loader="dcp", weights_mapping=mapping).load(
                directory, {"model": model}, strict_model=True,
            )
            result = verifier.verify_hf_weights(model, directory, str(root / "report"))
            self.assertTrue(result["exact_reference_match"])
            self.assertEqual(result["model_tensors"], 1)

    def test_inverse_resolves_base_prefix_and_chained_names_after_split(self) -> None:
        """Feature: Prefixed and chained checkpoint mappings.

        Description: Load relative HF names through layered renames and a fusion.
        Expectation: Undo fusion first, then renames in reverse order, including anchored root names.
        """
        source = {
            "embed.weight": torch.arange(6).float().reshape(2, 3),
            "layers.0.ffn.w1.weight": torch.arange(6, 12).float().reshape(2, 3),
            "layers.0.ffn.w3.weight": torch.arange(12, 18).float().reshape(2, 3),
        }
        model = nn.Module()
        model.config = SimpleNamespace()
        model.base_model_prefix = "model"
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(2, 3)
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.fused = nn.Parameter(torch.zeros((4, 3)))
        model.model.layers = nn.ModuleList([layer])
        mapping = [
            WeightRenaming(r"^embed\.weight$", "embed_tokens.weight"),
            WeightRenaming(r"^layers\.(\d+)\.ffn\.", r"layers.\1.mlp."),
            WeightRenaming(r"^layers\.(\d+)\.mlp\.w1\.", r"layers.\1.mlp.gate_proj."),
            WeightRenaming(r"^layers\.(\d+)\.mlp\.w3\.", r"layers.\1.mlp.up_proj."),
            WeightConverter(
                ["mlp.gate_proj.weight", "mlp.up_proj.weight"], "mlp.fused",
                operations=[ConcatenateWithSections((2, 2), dim=0)],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(source, root / "model.safetensors")
            HuggingFaceCheckpointer(loader="dcp", weights_mapping=mapping).load(
                directory, {"model": model}, strict_model=True,
            )
            result = verifier.verify_hf_weights(model, directory, str(root / "report"), chunk_elements=2)
            self.assertTrue(result["exact_reference_match"])
            self.assertEqual(result["hf_tensors"], 3)
            self.assertEqual(result["incomplete_tensors"], 0)

    def test_reverse_split_and_transpose_round_trip(self) -> None:
        """Feature: Inverse layouts; Description: Split a transposed fused weight; Expectation: Exact source match."""
        source = {
            "left": torch.arange(6).float().reshape(2, 3),
            "right": torch.arange(6, 12).float().reshape(2, 3),
        }
        model = _model(torch.cat(list(source.values())).t().contiguous())
        model._hp_used_base_weight_conversions = [WeightConverter(
            source_patterns=["left", "right"], target_patterns="weight",
            operations=[ConcatenateWithSections((2, 2), dim=0)],
        )]
        model._hp_used_replacement_weight_conversions = [WeightConverter(
            source_patterns="weight", target_patterns="weight", operations=[Transpose()],
        )]
        before = model.weight.detach().clone()
        mapping = model._weight_conversions
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(source, root / "model.safetensors")
            with patch.object(verifier.DTensor, "full_tensor", side_effect=AssertionError("Unexpected full gather")):
                result = verifier.verify_hf_weights(model, directory, str(root / "report"), chunk_elements=2)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["exact_reference_match"])
            self.assertEqual(result["hf_tensors"], 2)
            self.assertEqual(result["hf_elements"], 12)
            with (root / "report/tensors.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["name"] for row in rows}, {"left", "right"})
            self.assertTrue(all(row["result"] == "EXACT" for row in rows))
        self.assertIs(model._weight_conversions, mapping)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)

    def test_reverse_merged_experts_and_native_layout(self) -> None:
        """Feature: MoE checkpoint layouts.

        Description: Undo expert stacking, gate/up concatenation and native transposition.
        Expectation: Compare all original per-expert HF matrices without a full gather.
        """
        source = {
            f"experts.{expert}.{name}.weight": torch.arange(6).float().reshape(2, 3) + expert * 20 + branch * 10
            for expert in range(2) for branch, name in enumerate(("w1", "w3"))
        }
        fused = torch.stack([
            torch.cat([source[f"experts.{expert}.w1.weight"], source[f"experts.{expert}.w3.weight"]])
            for expert in range(2)
        ])
        model = _model(fused.transpose(-2, -1).contiguous())
        model._hp_used_base_weight_conversions = [WeightConverter(
            ["experts.*.w1.weight", "experts.*.w3.weight"], "weight",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        )]
        model._hp_used_replacement_weight_conversions = [WeightConverter(
            "weight", "weight", operations=[Transpose(dim0=-2, dim1=-1)],
        )]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(source, root / "model.safetensors")
            result = verifier.verify_hf_weights(model, directory, str(root / "report"), chunk_elements=2)
            self.assertTrue(result["exact_reference_match"])
            self.assertEqual(result["hf_tensors"], 4)
            self.assertEqual(result["hf_elements"], 24)

    def test_corruption_reports_location_and_error(self) -> None:
        """Feature: Corruption detection.

        Description: Change one loaded value.
        Expectation: Failure with the exact source location and error.
        """
        source = torch.arange(12).float().reshape(4, 3)
        model = _model(source)
        with torch.no_grad():
            model.weight[2, 1] += 0.5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": source}, root / "model.safetensors")
            with self.assertLogs(verifier.logger, level="INFO") as captured:
                with self.assertRaisesRegex(RuntimeError, "HF weight check failed"):
                    verifier.verify_hf_weights(model, directory, str(root / "report"), chunk_elements=3)
            self.assertIn("MISMATCH", "\n".join(captured.output))
            report = json.loads((root / "report/rank_00000.json").read_text(encoding="utf-8"))
            record = report["tensors"][0]
            self.assertEqual(record["load_dtype_mismatches"], 1)
            self.assertEqual(record["max_abs_error"], 0.5)
            self.assertEqual(record["first_difference"]["hf_index"], [2, 1])

    def test_load_dtype_rounding_is_explicit(self) -> None:
        """Feature: Cast diagnostics; Description: Store FP32 in BF16; Expectation: Label rounding separately."""
        source = torch.tensor([[1.0001, 2.0001]])
        model = _model(source.bfloat16())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": source}, root / "model.safetensors")
            result = verifier.verify_hf_weights(model, directory, str(root / "report"))
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["exact_reference_match"])
            self.assertEqual(result["dtype_cast_tensors"], 1)
            self.assertEqual(result["exact_tensors"], 0)

    def test_exact_values_do_not_allocate_error_tensors(self) -> None:
        """Feature: Large-table checking.

        Description: Compare equal BF16 reference and FP32 stored values.
        Expectation: Report zero error without computing double-precision differences.
        """
        expected = torch.arange(12).bfloat16().reshape(4, 3)
        with patch.object(torch.Tensor, "double", side_effect=AssertionError("Unnecessary error tensor")):
            result = verifier._difference(expected.float(), expected, torch.float32)
        self.assertEqual(result["elements"], 12)
        self.assertEqual(result["exact_reference_mismatches"], 0)
        self.assertEqual(result["max_abs_error"], 0)

    def test_coverage_bookkeeping_does_not_grow_with_read_tiles(self) -> None:
        """Feature: Bounded coverage bookkeeping.

        Description: Compare one source region using many small read tiles.
        Expectation: Retain one coverage box for the model region, not one per tile.
        """
        source = torch.arange(120).float().reshape(12, 10)
        model = _model(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": source}, root / "model.safetensors")
            with patch.object(verifier, "box_union_volume", wraps=verifier.box_union_volume) as volume:
                result = verifier._check_local(model, verifier._Reference(directory), 2)
            volume.assert_called_once_with([((0, 0), (12, 10))])
            self.assertEqual(result["uncovered_model_shards"], [])
            self.assertEqual(result["tensors"][0]["elements"], 120)

    def test_fp8_reference_is_decoded_before_comparison(self) -> None:
        """Feature: Quantized source; Description: Compare against FP8 with scales; Expectation: BF16 exact match."""
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
        model = _model((weight.float() * 2).bfloat16())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": weight, "weight_scale_inv": torch.tensor([[2.0]])}, root / "model.safetensors")
            (root / "config.json").write_text(json.dumps({
                "quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 2]},
            }), encoding="utf-8")
            result = verifier.verify_hf_weights(model, directory, str(root / "report"), chunk_elements=2)
            self.assertTrue(result["exact_reference_match"])
            self.assertEqual(result["hf_tensors"], 1)

    def test_cpu_host_table_and_padding_are_both_checked(self) -> None:
        """Feature: CPU external storage.

        Description: Check logical rows and padding.
        Expectation: Detect corrupt padding independently of real weights.
        """
        source = torch.arange(10).float().reshape(5, 2)
        model = nn.Module()
        model.config = SimpleNamespace()
        model._weight_conversions = []
        model.embed = HostEngramTable(
            source_weight=nn.Parameter(torch.empty((8, 2), device="meta")), logical_rows=5,
        )
        model.embed.bind_planned_shard(ep_rank=0, ep_size=1)
        torch.utils.swap_tensors(model.embed.weight, nn.Parameter(torch.cat([source, torch.zeros((3, 2))])))
        model._hp_external_state = DeepseekV41HostState(model, {"embed": model.embed}, None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"embed.weight": source.bfloat16()}, root / "model.safetensors")
            result = verifier.verify_hf_weights(model, directory, str(root / "report"))
            self.assertTrue(result["exact_reference_match"])
            self.assertEqual(result["padding_tensors"], 1)
            self.assertEqual(result["hf_elements"], 10)
            with torch.no_grad():
                model.embed.weight[7, 1] = 1
            with self.assertRaisesRegex(RuntimeError, "HF weight check failed"):
                verifier.verify_hf_weights(model, directory, str(root / "bad_padding"))
            report = json.loads((root / "bad_padding/rank_00000.json").read_text(encoding="utf-8"))
            padding = next(row for row in report["tensors"] if row.get("kind") == "padding")
            self.assertEqual(padding["load_dtype_mismatches"], 1)
            self.assertEqual(padding["first_difference"]["local_index"], [7, 1])

    def test_sharded_and_empty_rank_coverage(self) -> None:
        """Feature: Distributed coverage.

        Description: Compare three simulated ranks.
        Expectation: Check the full union, including an empty rank.
        """
        source = torch.arange(12).float().reshape(4, 3)
        model = _model(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": source}, root / "model.safetensors")
            reference = verifier._Reference(directory)
            reports = []
            for rank, (start, stop) in enumerate(((0, 2), (2, 4), (4, 4))):
                shard = CheckpointTensorShard(source[start:stop], (4, 3), (start, 0))
                with patch.object(verifier, "_model_shards", return_value={"weight": shard}):
                    report = verifier._check_local(model, reference, 2)
                report.update(rank=rank, load_dtype_mismatches=0, exact_reference_mismatches=0)
                reports.append(report)
            self.assertEqual(verifier._summarize(reports, reference.shapes)["status"], "PASS")
            self.assertEqual(verifier._summarize(reports[:1], reference.shapes)["status"], "FAIL")

    def test_unchecked_source_is_not_reported_as_aligned(self) -> None:
        """Feature: Missing coverage.

        Description: Include an unused disk tensor.
        Expectation: Fail and label the missing tensor INCOMPLETE.
        """
        model = _model(torch.ones(2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": torch.ones(2), "missing": torch.ones(3)}, root / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "HF weight check failed"):
                verifier.verify_hf_weights(model, directory, str(root / "report"))
            summary = json.loads((root / "report/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["incomplete_tensors"], 1)
            self.assertEqual(summary["uncovered_hf_tensors"]["missing"], {"checked": 0, "expected": 3})

    def test_nonfinite_values_fail_with_readable_json(self) -> None:
        """Feature: Nonfinite values; Description: Equal infinities; Expectation: Fail without invalid JSON numbers."""
        model = _model(torch.tensor([float("inf")]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": torch.tensor([float("inf")])}, root / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "HF weight check failed"):
                verifier.verify_hf_weights(model, directory, str(root / "report"))
            text = (root / "report/rank_00000.json").read_text(encoding="utf-8")
            self.assertNotIn("Infinity", text)
            self.assertEqual(json.loads(text)["nonfinite_elements"], 1)

    def test_unmapped_model_parameter_is_not_skipped(self) -> None:
        """Feature: Complete model coverage.

        Description: Add a parameter with no corresponding source weight.
        Expectation: Fail and name the unchecked parameter.
        """
        model = _model(torch.ones(2))
        model.unloaded = nn.Parameter(torch.ones(3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"weight": torch.ones(2)}, root / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "HF weight check failed"):
                verifier.verify_hf_weights(model, directory, str(root / "report"))
            report = json.loads((root / "report/rank_00000.json").read_text(encoding="utf-8"))
            self.assertIn("unloaded", report["unmatched_recovered_keys"])

    def test_external_state_without_shard_provider_fails(self) -> None:
        """Feature: External-state protocol.

        Description: Expose an external parameter without its ownership geometry.
        Expectation: Reject incomplete checks instead of silently omitting CPU storage.
        """
        model = _model(torch.ones(2))
        model._hp_external_state = SimpleNamespace(parameters={"weight": model.weight})
        with self.assertRaisesRegex(ValueError, "checkpoint_tensor_shards"):
            verifier._model_shards(model)

    def test_deepseek_indexer_loading_and_inverse_names(self) -> None:
        """Feature: DeepSeek indexer inverse mapping.

        Description: Load distinct attention and indexer weights at reported source layers.
        Expectation: Both HF loaders preserve the paths and all weights compare exactly.
        """
        source_layers = (2, 8, 14, 20, 24, 28, 32, 36)
        for loader in ("dcp", "legacy"):
            with self.subTest(loader=loader), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = nn.Module()
                model.config = SimpleNamespace()
                model.base_model_prefix = "model"
                model.model = nn.Module()
                model.model.layers = nn.ModuleList([nn.Module() for _ in range(37)])
                source = {}
                for layer_id in source_layers:
                    attention = nn.Module()
                    attention.q_b_proj = nn.Linear(3, 2, bias=False)
                    attention.indexer = nn.Module()
                    for name in ("q_b_proj", "weights_proj", "wk"):
                        attention.indexer.add_module(name, nn.Linear(3, 2, bias=False))
                    attention.indexer.k_norm = nn.RMSNorm(2)
                    model.model.layers[layer_id].self_attn = attention
                    for position, (name, parameter) in enumerate(attention.named_parameters()):
                        hf_name = name.replace("q_b_proj", "wq_b")
                        source[f"layers.{layer_id}.attn.{hf_name}"] = (
                            torch.arange(parameter.numel()).reshape(parameter.shape).float()
                            + layer_id * 10 + position)
                save_file(source, root / "model.safetensors")
                report = HuggingFaceCheckpointer(
                    loader=loader, weights_mapping=build_deepseek_v41_checkpoint_mapping(),
                ).load(directory, {"model": model}, strict_model=True)["load_report"]
                self.assertEqual(report.missing_keys, ())
                self.assertEqual(report.unexpected_keys, ())
                result = verifier.verify_hf_weights(model, directory, str(root / "report"), chunk_elements=3)
                self.assertTrue(result["exact_reference_match"])
                self.assertEqual(result["incomplete_tensors"], 0)
                self.assertEqual(result["exact_tensors"], len(source))
