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
"""CPU contracts for native cache ownership and graph-resident V4 metadata."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import rl
from rl.unified_execution.ops.attention import flash_attention_npu_v4 as shared_attention
from rl.unified_execution.runtimes.vllm.paged_attention import install_paged_attention


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
    path = Path(rl.__file__).parent / "unified_execution/runtimes/vllm/attention_backend.py"
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

    def test_metadata_tracks_runner_in_place_updates(self) -> None:
        """Replay must read new lengths through the original persistent storage."""
        metadata, common = self._metadata([0, 2, 3], [130, 17])
        self.assertEqual(metadata.v4_query_start_loc.data_ptr(), common.query_start_loc.data_ptr())
        self.assertEqual(metadata.v4_seq_lens.data_ptr(), common.seq_lens.data_ptr())
        common.query_start_loc.copy_(torch.tensor([0, 1, 3], dtype=torch.int32))
        common.seq_lens.add_(4)
        torch.testing.assert_close(metadata.v4_query_start_loc, torch.tensor([0, 1, 3], dtype=torch.int32))
        torch.testing.assert_close(metadata.v4_seq_lens, torch.tensor([134, 21], dtype=torch.int32))

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

    def test_hnd_cache_uses_a_view_without_copying_history(self) -> None:
        """V4 sees logical page/token/head axes while retaining BNBD storage."""
        impl = self._impl()
        impl.use_bnsd_kv_cache = True
        impl.key_cache = torch.randn(8, 2, 128, 8, dtype=torch.bfloat16)
        impl.value_cache = torch.randn_like(impl.key_cache)
        metadata, _ = self._metadata([0, 1], [129])
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        impl.forward_impl(query, None, None, (), metadata, torch.empty_like(query))
        key, value = self.provider.flash_attn_varlen_func.call_args.args[1:]
        self.assertEqual(key.shape, (8, 128, 2, 8))
        self.assertEqual(key.data_ptr(), impl.key_cache.data_ptr())
        self.assertEqual(value.data_ptr(), impl.value_cache.data_ptr())
        self.assertEqual(key.stride(), (2048, 8, 1024, 1))

    def test_missing_graph_metadata_fails(self) -> None:
        """Missing device offsets must not use stale host lists as a fallback."""
        impl = self._impl()
        metadata, _ = self._metadata([0, 1], [1])
        metadata.v4_query_start_loc = None
        query = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "device request boundaries"):
            impl.forward_impl(query, None, None, (), metadata, torch.empty_like(query))
        self.provider.flash_attn_varlen_func.assert_not_called()

    def test_graph_modes_preserve_decode_only_attention_capability(self) -> None:
        """Piecewise prefill leaves attention outside the captured pieces."""
        native_init = _NativeImpl.__init__
        for mode in ("decode_graph", "mixed_piecewise"):
            def initialize(instance: object, *args: object, mode=mode, **kwargs: object) -> None:
                """Supply each permitted runner graph configuration."""
                native_init(instance, *args, **kwargs)
                instance.vllm_config.model_config.enforce_eager = False
                instance.vllm_config.compilation_config = SimpleNamespace(cudagraph_mode=mode)

            with self.subTest(mode=mode), patch.object(_NativeImpl, "__init__", initialize):
                self._impl()
                self.assertEqual(
                    self.backend.UnifiedAttentionMetadataBuilder.get_cudagraph_support(None, None), "decode_only",
                )

    def test_install_selects_v4_per_layer_and_leaves_native_default(self) -> None:
        """Explicit class selection must bypass Ascend's global backend selector."""
        config = SimpleNamespace(model_type="qwen3", hidden_size=32, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=8)
        attention = SimpleNamespace(config=config, scaling=0.25)
        model = SimpleNamespace(config=config, layers=[SimpleNamespace(self_attn=attention)])
        native_module_name = "vllm.model_executor.layers.attention"
        native_module = ModuleType(native_module_name)
        attention_factory = Mock()
        native_module.Attention = attention_factory
        modules = {
            native_module_name: native_module,
            "rl.unified_execution.runtimes.vllm.attention_backend": self.backend,
        }
        for name in ("native", "flash_attention_npu_v4"):
            runtime = SimpleNamespace(
                cache_config=object(),
                additional_config={},
            )
            with self.subTest(backend=name), patch.dict(sys.modules, modules), patch(
                "rl.unified_execution.runtimes.vllm.paged_attention.payload_from_vllm_config",
                return_value={"attention_backend": name},
            ), patch(
                "rl.unified_execution.runtimes.vllm.paged_attention.validate_attention_execution_path",
                return_value=(model, [attention]),
            ):
                install_paged_attention(model, runtime)
            options = attention_factory.call_args.kwargs
            if name == "native":
                self.assertNotIn("attn_backend", options)
            else:
                self.assertIs(options["attn_backend"], self.backend.UnifiedAscendAttentionBackend)
            self.assertEqual(options["num_heads"], 4)
            self.assertEqual(options["num_kv_heads"], 2)
            self.assertEqual(options["scale"], 0.25)
