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
"""Native quantized HF reads agree with independently decoded BF16 tensors."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn
from transformers.core_model_loading import Concatenate, MergeModulelist

from hyper_parallel.components.checkpoint.huggingface_checkpointer import HuggingFaceCheckpointer
from hyper_parallel.components.checkpoint.huggingface_load_planner import _load_in_process
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter
from hyper_parallel.components.checkpoint.remap_planner import RemapBlock, RemapLoadPlanner
from hyper_parallel.components.checkpoint.quantized_hf_storage import (
    QuantizedHuggingFaceStorageReader, build_quantized_hf_reader,
)
from hyper_parallel.models._transformers.auto_model import HyperAutoModelForImageTextToText
from hyper_parallel.models.deepseek_v41.adapter.engram.host_source import materialize_owner_rows
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.external_state import ExternalLoadContext
from tests.common.mark_utils import arg_mark


_QUANTIZATION = {"quant_method": "fp8", "weight_block_size": [32, 32],
                 "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
_EXPERT = "layers.0.ffn.experts.0.w1.weight"


def _fixture(root: Path) -> dict[str, torch.Tensor]:
    """Write split weight/scale files and compute reference values without reader code."""
    packed = torch.arange(35 * 32).remainder(256).to(torch.uint8).reshape(35, 32)
    fp8 = torch.arange(35 * 67).remainder(17).sub(8).float().reshape(35, 67).to(torch.float8_e4m3fn)
    engram = torch.arange(35 * 64).remainder(17).sub(8).float().reshape(35, 64).to(torch.float8_e4m3fn)
    weights = {_EXPERT: packed.view(torch.int8), "layers.0.attn.wq_a.weight": fp8,
               "layers.1.engram.embed.weight": engram, "norm.weight": torch.arange(3).float()}
    scales = {_EXPERT[:-6] + "scale": torch.full((35, 2), 2.0).to(torch.float8_e8m0fnu),
              "layers.0.attn.wq_a.scale": torch.tensor([[0.5, 2, 4], [8, 0.25, 1]]).to(torch.float8_e8m0fnu),
              "layers.1.engram.embed.scale": torch.full((35, 2), 0.25).to(torch.float8_e8m0fnu)}
    save_file(weights, root / "weights.safetensors")
    save_file(scales, root / "scales.safetensors")
    weight_map = {name: "weights.safetensors" for name in weights}
    weight_map.update({name: "scales.safetensors" for name in scales})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"quantization_config": _QUANTIZATION}), encoding="utf-8")
    # E2M1 is sign * (1 + mantissa / 2) * 2**(exponent - 1), except exponent zero.
    codes = torch.empty((35, 64), dtype=torch.int64)
    codes[:, 0::2] = (packed & 15).long()
    codes[:, 1::2] = (packed >> 4).long()
    exponent, mantissa = (codes & 7) >> 1, codes & 1
    magnitude = torch.where(exponent == 0, mantissa.float() / 2,
                            (1 + mantissa.float() / 2) * torch.exp2(exponent.float() - 1))
    decoded = torch.where(codes >= 8, -magnitude, magnitude)
    expanded = torch.tensor([[0.5, 2, 4], [8, 0.25, 1]]).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return {_EXPERT: (decoded * 2).bfloat16(),
            "layers.0.attn.wq_a.weight": (fp8.float() * expanded[:35, :67]).bfloat16(),
            "layers.1.engram.embed.weight": (engram.float() * 0.25).bfloat16(),
            "norm.weight": weights["norm.weight"]}


class TestQuantizedReader(unittest.TestCase):
    """Exercise real safetensors, DCP plans, cross-file scales and invalid metadata."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fp4_blocks_must_align_packed_byte_pairs(self):
        """Feature: Packed FP4 read alignment.

        Description: Declare one scale per logical column, crossing packed byte pairs.
        Expectation: Reject this unsupported layout before generating incorrect reads.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            scale_name = _EXPERT[:-6] + "scale"
            save_file({scale_name: torch.ones((35, 64))}, root / "column_scales.safetensors")
            index = {"weight_map": {_EXPERT: "weights.safetensors", scale_name: "column_scales.safetensors"}}
            (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "packed byte pairs"):
                QuantizedHuggingFaceStorageReader(root).load_metadata()

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unaligned_regions_and_bounded_chunks(self):
        """Feature: Online HF checkpoint dequantization.

        Description: Odd FP4 columns and partial FP8 blocks exactly match full offline decoding.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = _fixture(root)
            reader = QuantizedHuggingFaceStorageReader(root)
            metadata = reader.load_metadata()
            self.assertEqual(set(metadata.state_dict_metadata), set(references))
            for name, expected in references.items():
                if name == "norm.weight":
                    continue
                self.assertEqual(tuple(metadata.state_dict_metadata[name].size), tuple(expected.shape))
                for offsets, lengths in (((0, 0), tuple(expected.shape)), ((1, 3), (33, 47)),
                                         ((31, 31), (4, 2)), ((35, 1), (0, 2))):
                    with self.subTest(name=name, offsets=offsets), \
                            patch("hyper_parallel.components.checkpoint.quantized_hf_storage."
                                  "_SCRATCH_BYTES", 1024):
                        actual = reader.read_region(name, offsets, lengths)
                        row, column = offsets
                        rows, columns = lengths
                        reference = expected[row:row + rows, column:column + columns].contiguous()
                        self.assertTrue(torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_quantization_builds_float_model_without_preload(self):
        """Feature: Online HF checkpoint dequantization.

        Description: Stored native quantization selects config-first meta construction, not quantized modules.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        config = SimpleNamespace(model_type="synthetic_model", architectures=[],
                                 quantization_config=_QUANTIZATION)
        model = nn.Linear(2, 2)
        target = "hyper_parallel.models._transformers.auto_model"
        with patch.dict(os.environ, {"HYPER_PARALLEL_HF_LOADER": "dcp"}), \
                patch(target + "._init_model", return_value=(False, model)) as initialize, \
                patch(target + ".apply_model_infrastructure", return_value=model) as infrastructure, \
                patch(target + "._current_device", return_value=torch.device("cpu")):
            HyperAutoModelForImageTextToText._build_model(
                "/weights", hf_config=config, mesh=None, sharding_planner=None, fsdp2_manager=None,
                backend=None, is_hf_model=False, peft_config=None, torch_dtype=torch.bfloat16,
                attn_implementation="eager", validate_placement=False, load_base_model=True)
        self.assertIsNone(initialize.call_args.args[1])
        self.assertTrue(infrastructure.call_args.kwargs["is_meta_device"])
        self.assertTrue(infrastructure.call_args.kwargs["load_base_model"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dcp_remap_transpose_reads_logical_fp4_region(self):
        """Feature: Online HF checkpoint dequantization.

        Description: A transposed shard consumes unpacked coordinates through DCP read/apply.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = _fixture(root)[_EXPERT]
            reader = QuantizedHuggingFaceStorageReader(root)
            target = torch.zeros((47, 33), dtype=torch.float32)
            block = RemapBlock(offsets=(0, 0), lengths=(47, 33), source=_EXPERT,
                               base=(1, 3), coeff=((0, 1), (1, 0)))
            planner = RemapLoadPlanner({"fused": [block]})
            _load_in_process({"fused": target}, reader, planner)
            self.assertTrue(torch.equal(target, expected[1:34, 3:50].T.float()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_hf_checkpointer_decodes_before_expert_merge(self):
        """Feature: Online HF checkpoint dequantization.

        Description: The normal DCP HF entry selects the adapter reader and merges decoded experts.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = _fixture(root)[_EXPERT]
            model = nn.Module()
            model.config = SimpleNamespace(model_type="deepseek_v41", architectures=[])
            model.register_parameter("fused", nn.Parameter(torch.zeros((1, 70, 64))))
            # Reuse one source under two patterns by adding a second native expert projection.
            from_file = torch.arange(35 * 32).remainder(256).to(torch.uint8).reshape(35, 32).view(torch.int8)
            second = _EXPERT.replace("w1", "w3")
            save_file({_EXPERT: from_file, second: from_file.clone()}, root / "experts.safetensors")
            scale = torch.full((35, 2), 2.0).to(torch.float8_e8m0fnu)
            save_file({_EXPERT[:-6] + "scale": scale, second[:-6] + "scale": scale.clone()},
                      root / "scales.safetensors")
            index = {name: "experts.safetensors" for name in (_EXPERT, second)}
            index.update({name[:-6] + "scale": "scales.safetensors" for name in (_EXPERT, second)})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}), encoding="utf-8")
            mapping = [WeightConverter(
                source_patterns=["layers.0.ffn.experts.*.w1.weight", "layers.0.ffn.experts.*.w3.weight"],
                target_patterns="fused", operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
            )]
            state = {"model": model}
            HuggingFaceCheckpointer(weights_mapping=mapping, loader="dcp").load(str(root), state)
            self.assertEqual(state["load_report"].missing_keys, ())
            self.assertTrue(torch.equal(model.fused, torch.cat((expected, expected), dim=0).unsqueeze(0).float()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_missing_scale_and_unsupported_formats_fail(self):
        """Feature: Online HF checkpoint dequantization.

        Description: Malformed native checkpoints fail while planning, before weights are copied.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            path = root / "model.safetensors.index.json"
            index = json.loads(path.read_text(encoding="utf-8"))
            del index["weight_map"][_EXPERT[:-6] + "scale"]
            path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing scale"):
                QuantizedHuggingFaceStorageReader(root).load_metadata()
            (root / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "other"}}),
                                             encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported HF quantization"):
                build_quantized_hf_reader(root)
            (root / "config.json").write_text("{}", encoding="utf-8")
            self.assertIsNone(build_quantized_hf_reader(root))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bf16_checkpoint_keeps_ordinary_reader(self):
        """Feature: Online HF checkpoint dequantization.

        Description: An unquantized HF directory loads unchanged, without selecting a decoder.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = torch.arange(35).reshape(5, 7).bfloat16()
            save_file({"weight": expected}, root / "model.safetensors")
            (root / "config.json").write_text("{}", encoding="utf-8")
            model = nn.Linear(7, 5, bias=False, dtype=torch.bfloat16)
            model.config = SimpleNamespace(model_type="deepseek_v41", architectures=[])
            self.assertIsNone(build_quantized_hf_reader(root))
            HuggingFaceCheckpointer(weights_mapping=[], loader="dcp").load(str(root), {"model": model})
            self.assertTrue(torch.equal(model.weight, expected))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_scale_values_and_region_bounds_fail(self):
        """Feature: Online HF checkpoint dequantization.

        Description: Reserved UE8M0 values and out-of-bounds logical rectangles are rejected.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            reader = QuantizedHuggingFaceStorageReader(root)
            with self.assertRaisesRegex(ValueError, "Invalid logical"):
                reader.read_region(_EXPERT, (0, 63), (1, 2))
            scale_name = _EXPERT[:-6] + "scale"
            save_file({scale_name: torch.full((35, 2), 255, dtype=torch.uint8).view(torch.float8_e8m0fnu)},
                      root / "badscale.safetensors")
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"] = {_EXPERT: "weights.safetensors", scale_name: "badscale.safetensors"}
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reserved NaN"):
                QuantizedHuggingFaceStorageReader(root).read_region(_EXPERT, (0, 0), (1, 1))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_host_engram_decodes_to_fp32_owner_rows(self):
        """Feature: Online HF checkpoint dequantization.

        Description: The separate Host materialization path uses the same native decoding.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = _fixture(root)["layers.1.engram.embed.weight"]
            model = nn.Module()
            model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            model.layers[1].engram = nn.Module()
            table = HostEngramTable(source_weight=nn.Parameter(torch.empty((35, 64), device="meta")),
                                    logical_rows=35)
            table.bind_planned_shard(ep_rank=0, ep_size=1)
            model.layers[1].engram.embed = table
            sources = materialize_owner_rows(model, {"layers.1.engram.embed": table},
                                              ExternalLoadContext(True, str(root), []))
            self.assertEqual(sources, frozenset({"layers.1.engram.embed.weight"}))
            self.assertEqual(table.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(table.weight, expected.float()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_synthetic_fp8_block128_uses_the_same_reader(self):
        """Feature: Online HF checkpoint dequantization.

        Description: Synthetic FP8/FP32 scales reuse the reader used by FP4/UE8M0 storage.
        Expectation: Exact reference values load successfully, or invalid inputs fail explicitly.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "model.layers.0.mlp.gate.weight"
            quant = {"quant_method": "fp8", "weight_block_size": [128, 128]}
            weights = torch.arange(137 * 259).remainder(17).sub(8).float().reshape(137, 259)
            scales = torch.tensor([[0.5, 2, 4], [8, 0.25, 1]])
            save_file({name: weights.to(torch.float8_e4m3fn), name[:-6] + "weight_scale_inv": scales},
                      root / "model.safetensors")
            (root / "config.json").write_text(json.dumps({"quantization_config": quant}), encoding="utf-8")
            model = nn.Module()
            model.config = SimpleNamespace(model_type="synthetic_model", architectures=[])
            current = model
            for part in name.split(".")[:-1]:
                current.add_module(part, nn.Module())
                current = current.get_submodule(part)
            current.register_parameter("weight", nn.Parameter(torch.zeros_like(weights)))
            expected = (weights * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:137, :259]).bfloat16()
            report_state = {"model": model}
            HuggingFaceCheckpointer(weights_mapping=[], loader="dcp").load(str(root), report_state)
            self.assertEqual(report_state["load_report"].missing_keys, ())
            self.assertEqual(report_state["load_report"].unexpected_keys, ())
            self.assertTrue(torch.equal(current.weight, expected.float()))
            reader = build_quantized_hf_reader(root)
            self.assertIsInstance(reader, QuantizedHuggingFaceStorageReader)
            region = reader.read_region(name, (119, 123), (18, 136))
            self.assertTrue(torch.equal(region, expected[119:137, 123:259]))
