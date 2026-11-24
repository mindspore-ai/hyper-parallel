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
"""Public attention masks and declarative Qwen3-VL-MoE family boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward


def test_text_cp_keeps_global_2d_padding_mask() -> None:
    """Retain global padding metadata and restore variable-length kernel output."""
    query = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    key = torch.empty(2, 1, 4, 4)
    padding_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    module = SimpleNamespace(training=True, attention_dropout=0.0, scaling=0.5)
    kernel = Mock(side_effect=lambda *args, **kwargs: (args[0],))
    with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
        output, _ = npu_fusion_attention_forward(module, query, key, query, padding_mask)
    expected = query.transpose(1, 2) * padding_mask[:, :, None, None]
    torch.testing.assert_close(output, expected)
    assert kernel.call_args.args[4] == "TND", f"Expected TND: {kernel.call_args}"
    assert kernel.call_args.kwargs["actual_seq_qlen"] == (3, 5), f"Expected Q ends (3, 5): {kernel.call_args}"
    assert kernel.call_args.kwargs["actual_seq_kvlen"] == (3, 5), f"Expected KV ends (3, 5): {kernel.call_args}"
    assert kernel.call_args.kwargs["sparse_mode"] == 3, f"Expected causal mode 3: {kernel.call_args}"
    with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
        direct, _ = npu_fusion_attention_forward(
            module, query, key, query, padding_mask, dropout=0.1, scaling=0.25,
        )
    torch.testing.assert_close(direct, expected)
    assert kernel.call_args.kwargs["keep_prob"] == 0.9, f"Expected keep_prob=0.9: {kernel.call_args}"
    assert kernel.call_args.kwargs["scale"] == 0.25, f"Expected scale=0.25: {kernel.call_args}"
    for mask in (
        torch.tensor([[False, True, True, True]]),
        torch.tensor([[True, False, True, False]]),
        torch.zeros(1, 4, dtype=torch.bool),
    ):
        kernel.reset_mock()
        with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
            output, _ = npu_fusion_attention_forward(module, query, key, query, mask)
        torch.testing.assert_close(output, query.transpose(1, 2) * mask[:, :, None, None])
        if mask.any():
            assert kernel.call_args.args[4] == "TND", f"Expected unpadding for {mask}"
        else:
            kernel.assert_not_called()


def test_text_cp_window_uses_compressed_band_mask() -> None:
    """Unpad contiguous padding and pass the causal window to the NPU band mode."""
    query = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    module = SimpleNamespace(training=True, attention_dropout=0.0, scaling=0.5)
    kernel = Mock(side_effect=lambda *args, **kwargs: (args[0],))
    for mask in (
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]]),
        torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]]),
        torch.zeros(2, 4, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long),
        None,
    ):
        for options in ({"sliding_window": 2}, {"pre_tokens": 1}):
            kernel.reset_mock()
            with patch.dict("sys.modules", {"torch_npu": SimpleNamespace(npu_fusion_attention=kernel)}):
                output, _ = npu_fusion_attention_forward(module, query, query, query, mask, **options)
            expected = query.transpose(1, 2)
            if mask is not None:
                expected = expected * mask[:, :, None, None]
            torch.testing.assert_close(output, expected)
            if mask is not None and not mask.any():
                kernel.assert_not_called()
                continue
            assert kernel.call_args.kwargs["sparse_mode"] == 4, f"Expected band mode: {kernel.call_args}"
            assert kernel.call_args.kwargs["pre_tockens"] == 1, f"Expected one preceding token: {kernel.call_args}"
            assert kernel.call_args.kwargs["next_tockens"] == 0, f"Expected no future tokens: {kernel.call_args}"
            actual_mask = kernel.call_args.kwargs["atten_mask"]
            assert actual_mask.shape == (2048, 2048), f"Expected compressed mask: {actual_mask.shape}"
            if mask is not None and not mask.all():
                assert kernel.call_args.args[4] == "TND", f"Expected TND: {kernel.call_args}"
                ends = tuple(mask.sum(dim=-1)[mask.any(dim=-1)].cumsum(0).tolist())
                assert kernel.call_args.kwargs["actual_seq_qlen"] == ends, f"Expected sequence ends {ends}"
