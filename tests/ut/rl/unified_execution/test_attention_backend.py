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
"""Worker KV cache, graph replay and PCP sampling probabilities."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

import rl
from rl.unified_execution.modules import flash_attention_v4 as shared_attention
from rl.unified_execution.vllm.pcp_log_probs import compute_pcp_token_log_probs


@dataclass
class _Metadata:
    """Minimal native metadata fields consumed by the adapter."""

    causal: bool = True
    block_tables: torch.Tensor | None = None


class _NativeBuilder:
    """Retain the native builder boundary without importing an NPU runtime."""

    metadata_cls = _Metadata

    def __init__(self, spec: object, unused_layers: object, config: object, unused_device: object) -> None:
        """Keep the real builder initialization interface for the adapter."""
        del unused_layers, unused_device
        self.kv_cache_spec = spec
        self.model_config = config.model_config

    def build(self, unused_prefix: int, common: SimpleNamespace, unused_fast: bool = False) -> _Metadata:
        """Return metadata using the adapter's declared metadata class."""
        del unused_prefix, unused_fast
        return self.metadata_cls(causal=common.causal, block_tables=common.block_table_tensor)


class _NativeImpl:
    """Expose native initialization fields; device cache writing remains in ST."""

    def __init__(self, unused_heads: int, unused_dim: int, scale: float, *_args: object, **_kwargs: object) -> None:
        """Initialize the native contract fields used by the real adapter."""
        del unused_heads, unused_dim
        self.scale = scale
        self.use_bnsd_kv_cache = False
        self.key_cache = None
        self.value_cache = None
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
            model_config=SimpleNamespace(runner_type="generate", max_model_len=4096, enforce_eager=True),
            speculative_config=None,
        )


def _load_backend() -> ModuleType:
    """Load the real adapter with isolated native dependencies, preserving imports."""
    ascend = ModuleType("vllm_ascend.attention.attention_v1")
    ascend.AscendAttentionBackend = type("NativeBackend", (), {})
    ascend.AscendAttentionBackendImpl = _NativeImpl
    ascend.AscendAttentionMetadataBuilder = _NativeBuilder
    ascend.AscendMetadata = _Metadata
    base = ModuleType("vllm.v1.attention.backend")
    base.AttentionType = SimpleNamespace(DECODER="decoder")
    base.AttentionCGSupport = SimpleNamespace(UNIFORM_SINGLE_TOKEN_DECODE="decode_only")
    config = ModuleType("vllm.config")
    config.CUDAGraphMode = SimpleNamespace(FULL_DECODE_ONLY="decode_graph", FULL_AND_PIECEWISE="mixed_piecewise")
    fence = ModuleType("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence")
    fence.attention_transfer_window = nullcontext
    path = Path(rl.__file__).parent / "unified_execution/vllm/attention/backend_v4.py"
    spec = importlib.util.spec_from_file_location("_rl_v4_backend_ut", path)
    module = importlib.util.module_from_spec(spec)
    modules = {entry.__name__: entry for entry in (ascend, base, config, fence, module)}
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TestUnifiedAttentionBackend(unittest.TestCase):
    """Check metadata ownership and shared-provider calls without a device."""

    def setUp(self) -> None:
        """Use actual shared validation and replace only the native V4 operator."""
        self.backend = _load_backend()
        self.provider = SimpleNamespace(
            flash_attn_varlen_func=Mock(side_effect=lambda query, *_a, **_k: query + 1),
            get_scheduler_metadata=Mock(side_effect=lambda **_k: torch.zeros(32, dtype=torch.uint8)),
        )
        provider_patch = patch.object(shared_attention, "_provider", return_value=self.provider)
        device_patch = patch.object(shared_attention, "_validate_device")
        provider_patch.start()
        device_patch.start()
        self.addCleanup(provider_patch.stop)
        self.addCleanup(device_patch.stop)

    def _impl(self, **changes: object) -> object:
        """Create a supported decoder implementation unless a test changes it."""
        args = {
            "num_heads": 4, "head_size": 8, "scale": 0.25, "num_kv_heads": 2,
            "alibi_slopes": None, "sliding_window": None, "kv_cache_dtype": "auto",
            "logits_soft_cap": None, "attn_type": "decoder", "kv_sharing_target_layer_name": None,
        }
        args.update(changes)
        impl = self.backend.UnifiedAscendAttentionImpl(**args)
        impl.key_cache = torch.randn(8, 128, 2, 8, dtype=torch.bfloat16)
        impl.value_cache = torch.randn_like(impl.key_cache)
        return impl

    def _metadata(self, offsets: list[int], lengths: list[int]) -> object:
        """Build actual metadata from buffers representing the native runner."""
        common = SimpleNamespace(
            causal=True, num_reqs=len(lengths),
            query_start_loc=torch.tensor(offsets, dtype=torch.int32),
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            block_table_tensor=torch.tensor([[3, 1], [5, 2]], dtype=torch.int32)[:len(lengths)],
            num_actual_tokens=offsets[-1], num_input_tokens=offsets[-1],
        )
        metadata = self._builder().build(0, common)
        return metadata, common

    def _builder(self) -> object:
        """Create the real adapter with a minimal native configuration boundary."""
        config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096, get_num_attention_heads=lambda _parallel: 4),
            parallel_config=None,
        )
        return self.backend.UnifiedAttentionMetadataBuilder(
            SimpleNamespace(num_kv_heads=2, head_size=8), ["layer"], config, torch.device("cpu"),
        )

    def test_scheduler_refresh_retains_captured_address(self) -> None:
        """Every builder call refreshes scheduler bytes without replacing graph storage."""
        _, common = self._metadata([0, 1, 2], [113, 127])
        self.provider.get_scheduler_metadata.side_effect = [
            torch.zeros(32, dtype=torch.uint8), torch.ones(32, dtype=torch.uint8),
        ]
        builder = self._builder()
        first = builder.build(0, common)
        address = first.v4_scheduler_metadata.data_ptr()
        common.seq_lens.copy_(torch.tensor([641, 514], dtype=torch.int32))
        second = builder.build(0, common)
        self.assertEqual(second.v4_scheduler_metadata.data_ptr(), address)
        self.assertIs(first.v4_scheduler_metadata, second.v4_scheduler_metadata)
        torch.testing.assert_close(first.v4_scheduler_metadata, torch.ones(32, dtype=torch.uint8))

    def test_prefix_chunk_decode_share_paged_cache_and_output_buffer(self) -> None:
        """Mixed prefill/decode retains prefix lengths, physical pages and scale."""
        impl = self._impl()
        metadata, _ = self._metadata([0, 2, 3], [130, 17])
        query = torch.randn(3, 4, 8, dtype=torch.bfloat16)
        output = torch.empty_like(query)
        key_before = impl.key_cache.clone()
        result = impl.forward_impl(query, None, None, (), metadata, output)
        self.assertIs(result, output)
        torch.testing.assert_close(result, query + 1)
        torch.testing.assert_close(impl.key_cache, key_before)
        call = self.provider.flash_attn_varlen_func.call_args
        self.assertIs(call.args[1], impl.key_cache)
        self.assertIs(call.args[2], impl.value_cache)
        self.assertIs(call.kwargs["cu_seqlens_q"], metadata.v4_query_start_loc)
        self.assertIs(call.kwargs["seqused_k"], metadata.v4_seq_lens)
        torch.testing.assert_close(call.kwargs["page_table"], metadata.block_tables)
        self.assertEqual(call.kwargs["max_seqlen_q"], 3)
        self.assertEqual(call.kwargs["max_seqlen_k"], 4096)
        self.assertEqual(call.kwargs["softmax_scale"], 0.25)
        self.assertTrue(call.kwargs["deterministic"])
        self.assertEqual(call.kwargs["num_splits"], 1)
        self.assertIs(call.kwargs["scheduler_metadata"], metadata.v4_scheduler_metadata)

    def test_empty_pcp_rank_returns_zero_without_calling_v4(self) -> None:
        """Native forward still gathers KV; an empty local query has no attention work."""
        impl = self._impl()
        impl.pcp_enabled = True
        metadata, _ = self._metadata([0], [])
        query = torch.randn(4, 4, 8, dtype=torch.bfloat16)
        output = torch.full_like(query, float("nan"))
        result = impl.forward_impl(query, None, None, (), metadata, output)
        self.assertIs(result, output)
        self.assertEqual(int(result.count_nonzero()), 0)
        self.provider.flash_attn_varlen_func.assert_not_called()

    def test_pcp_segments_keep_independent_causal_prefixes(self) -> None:
        """Two query segments can reuse one request's pages with distinct KV ends."""
        metadata, common = self._metadata([0, 3, 5], [35, 99])
        common.block_table_tensor[1].copy_(common.block_table_tensor[0])
        impl = self._impl()
        impl.pcp_enabled = True
        query = torch.randn(5, 4, 8, dtype=torch.bfloat16)
        impl.forward_impl(query, None, None, (), metadata, torch.empty_like(query))
        arguments = self.provider.flash_attn_varlen_func.call_args.kwargs
        torch.testing.assert_close(arguments["cu_seqlens_q"], torch.tensor([0, 3, 5], dtype=torch.int32))
        torch.testing.assert_close(arguments["seqused_k"], torch.tensor([35, 99], dtype=torch.int32))
        self.assertTrue(torch.equal(arguments["page_table"][0], arguments["page_table"][1]))


class TestPCPLogProbs(unittest.TestCase):
    """Check the numerical boundary and the narrowly scoped native binding."""

    def test_scores_native_ids_without_mutating_logits_or_selection(self) -> None:
        """Selected and top-k IDs retain order, duplicates and the existing FP32 semantics."""
        generator = torch.Generator().manual_seed(51)
        for dtype in (torch.bfloat16, torch.float32):
            logits = torch.randn(3, 131, generator=generator).to(dtype)[:, ::2]
            ids = torch.tensor([[4, 0, 4], [65, 3, 1], [2, 17, 6]], dtype=torch.int32)
            before_logits, before_ids = logits.clone(), ids.clone()
            actual = compute_pcp_token_log_probs(logits, ids)
            expected = torch.stack([row.float().log_softmax(-1)[selected.long()]
                                    for row, selected in zip(logits, ids)])
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
            self.assertTrue(torch.equal(logits, before_logits))
            self.assertTrue(torch.equal(ids, before_ids))
