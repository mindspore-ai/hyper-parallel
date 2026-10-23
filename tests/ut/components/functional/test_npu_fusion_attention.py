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
"""CPU regression for compressed causal-mask size and alignment."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class TestCausalMask(unittest.TestCase):
    """Keep the long-sequence and unequal-length regression in one test."""

    def test_compressed_mask_size_and_alignment(self) -> None:
        """Compress equal-length causal masks; preserve unequal lengths and windows."""
        source = Path(__file__).resolve().parents[4] / "hyper_parallel/components/functional/npu_fusion_attention.py"
        spec = importlib.util.spec_from_file_location("_causal_mask_test", source)
        attention = importlib.util.module_from_spec(spec)
        # Isolate the NPU import without changing another test's cached production module.
        with patch.dict(sys.modules, {spec.name: attention, "torch_npu": types.ModuleType("torch_npu")}):
            spec.loader.exec_module(attention)

        for q_len, kv_len in (
            (8, 8), (4, 8), (8, 4), (2048, 2048), (2049, 2049),
            (1024, 4097), (4097, 1024), (1, 65536), (65536, 65536)
        ):
            for window in (None, 2):
                if q_len == 65536 and window is not None:
                    continue
                with self.subTest(q_len=q_len, kv_len=kv_len, window=window):
                    query = torch.empty(1, 1, q_len, 1)
                    key = torch.empty(1, 1, kv_len, 1)
                    context = attention._prepare_fusion_attention_context(
                        SimpleNamespace(), query, key, key, None, {"sliding_window": window}
                    )
                    explicit = q_len != kv_len or window is not None or q_len * kv_len <= 2048 * 2048
                    mask_shape = (q_len, kv_len) if explicit else (2048, 2048)
                    self.assertEqual(tuple(context.attention_mask.shape), mask_shape)
                    expected_mode = 0 if explicit else 2
                    self.assertEqual(context.sparse_mode, expected_mode)
                    if q_len == 65536:
                        continue
                    query_positions = torch.arange(q_len)[:, None]
                    key_positions = torch.arange(kv_len)[None, :]
                    expected_positions = query_positions + max(kv_len - q_len, 0)
                    expected = key_positions <= expected_positions
                    if window is not None:
                        expected &= key_positions > expected_positions - window
                    if explicit:
                        actual = torch.logical_not(context.attention_mask)
                        self.assertEqual(context.pre_tokens, window if window is not None else 1048576)
                        self.assertEqual(context.next_tokens, 0)
                    else:
                        actual = key_positions <= query_positions
                    torch.testing.assert_close(actual, expected)
