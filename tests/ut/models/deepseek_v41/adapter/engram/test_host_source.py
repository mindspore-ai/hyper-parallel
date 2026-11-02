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
"""Load unpadded HF Engram tables into padded EP owner shards on CPU."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from hyper_parallel.components.checkpoint.quantized_hf_storage import QuantizedHuggingFaceStorageReader
from hyper_parallel.models.deepseek_v41.adapter.engram.host_source import materialize_owner_rows
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.external_state import ExternalLoadContext


def _owner_table(rank: int) -> HostEngramTable:
    """Create one of four owner shards for five logical rows padded to sixteen."""
    table = HostEngramTable(
        source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")), logical_rows=5,
    )
    table.weight = torch.nn.Parameter(torch.empty((4, 2), device="meta"))
    table.bind_planned_shard(ep_rank=rank, ep_size=4, ep_group=object())
    return table


class TestHostSourcePadding(unittest.TestCase):
    """Preserve real rows, zero padding, and reject incompatible source shapes."""

    def _check_owner_rows(self, quantized: bool, source_rows: int = 5) -> None:
        """Exercise real HF storage reads without creating a process group."""
        full = torch.arange(source_rows * 2, dtype=torch.float32).reshape(source_rows, 2)
        tensors = {"embed.weight": full.bfloat16()}
        if quantized:
            tensors = {
                "embed.weight": full.to(torch.float8_e4m3fn),
                "embed.weight_scale_inv": torch.ones(((source_rows + 1) // 2, 1)),
            }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(tensors, root / "model.safetensors")
            if quantized:
                (root / "config.json").write_text(json.dumps({
                    "quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 2]},
                }), encoding="utf-8")
            for rank in range(4):
                with self.subTest(quantized=quantized, source_rows=source_rows, rank=rank):
                    model = torch.nn.Module()
                    model.embed = _owner_table(rank)
                    parameter = model.embed.weight
                    with patch.object(
                        QuantizedHuggingFaceStorageReader, "read_region",
                        autospec=True,
                        side_effect=QuantizedHuggingFaceStorageReader.read_region,
                    ) as read:
                        consumed = materialize_owner_rows(
                            model, {"embed": model.embed}, ExternalLoadContext(True, directory, None),
                        )
                        if quantized:
                            count = max(0, min(4, 5 - rank * 4))
                            if count:
                                self.assertEqual(read.call_args.args[2:], ((rank * 4, 0), (count, 2)))
                            else:
                                read.assert_not_called()
                    expected = torch.zeros((4, 2))
                    count = max(0, min(4, 5 - rank * 4))
                    expected[:count] = full[rank * 4:rank * 4 + count]
                    self.assertIs(model.embed.weight, parameter)
                    self.assertEqual(model.embed.weight.dtype, torch.float32)
                    self.assertEqual(model.embed.weight.device.type, "cpu")
                    self.assertEqual(consumed, frozenset({"embed.weight"}))
                    torch.testing.assert_close(model.embed.weight, expected, rtol=0, atol=0)

    def test_bf16_source_preserves_rows_and_zeros_tail(self) -> None:
        """Feature: BF16 HF loading; Description: Load every owner; Expectation: Zero padded rows."""
        self._check_owner_rows(quantized=False)

    def test_quantized_source_reads_only_logical_overlap(self) -> None:
        """Feature: FP8 HF loading; Description: Read partial/empty owners; Expectation: No out-of-bounds reads."""
        self._check_owner_rows(quantized=True)

    def test_already_padded_source_ignores_padding_values(self) -> None:
        """Feature: Padded HF source; Description: Load nonzero source padding; Expectation: Zero runtime padding."""
        self._check_owner_rows(quantized=False, source_rows=16)

    def test_incompatible_source_shape_is_rejected(self) -> None:
        """Feature: Shape validation; Description: Supply wrong rows/width; Expectation: Reject before loading."""
        for shape in ((4, 2), (6, 2), (5, 3)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as directory:
                save_file({"embed.weight": torch.zeros(shape)}, Path(directory) / "model.safetensors")
                model = torch.nn.Module()
                model.embed = _owner_table(0)
                with self.assertRaisesRegex(ValueError, "Host Engram source shape mismatch"):
                    materialize_owner_rows(
                        model, {"embed": model.embed}, ExternalLoadContext(True, directory, None),
                    )
