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

"""Native deepseek_sparse_attention forward and gradient precision contracts."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.ops import ops, flash_mla_sparse_prefill
from hyper_parallel.components.ops.impl.deepseek_sparse_attention import (
    DSA_SPARSE_PREFILL, DSA_SPARSE_DECODE, _load_flash_prefill, _load_flash_decode, register_deepseek_sparse_attention,
)
from hyper_parallel.components.ops.registry import OpRegistry
from hyper_parallel.components.modules.attention.flash_mla import FlashMLASparseDecode, FlashMLASparsePrefill
from tests.common.op_precision_flash_mla import flash_mla_cases, _prefill_reference, _decode_reference
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestDeepseekSparseAttentionPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("deepseek_sparse_attention", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)


class TestFlashMLA(unittest.TestCase):
    """CPU ABI and scheduler-lifetime checks; real FlashMLA remains a device acceptance task."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        self.metadata = Mock(side_effect=lambda: (object(), None))
        self.calls = []

        def prefill(q: Any, kv: Any, indices: Any, scale: Any, *, d_v: Any, attn_sink: Any, topk_length: Any) -> Any:
            """Simulate the upstream sparse prefill ABI using the independent reference."""
            self.assertEqual(d_v, 512)
            self.calls.append("prefill")
            return _prefill_reference(q, kv, indices, scale, attention_sink=attn_sink, topk_length=topk_length)

        def decode(
            q: Any, cache: Any, block_table: Any, cache_seqlens: Any,
            d_v: Any, metadata: Any, **options: Any,
        ) -> Any:
            """Simulate the upstream sparse decode ABI using the independent reference."""
            self.assertIsNone(block_table)
            self.assertIsNone(cache_seqlens)
            self.assertEqual(d_v, 512)
            self.assertIsNotNone(metadata)
            self.calls.append("decode")
            translate = {"attn_sink": "attention_sink", "extra_k_cache": "extra_key_cache",
                         "extra_indices_in_kvcache": "extra_indices"}
            return _decode_reference(q, cache, options.pop("indices"),
                                     **{translate.get(k, k): v for k, v in options.items()})

        kernels = SimpleNamespace(flash_mla_sparse_fwd=prefill, flash_mla_with_kvcache=decode,
                                   get_mla_metadata=self.metadata)
        dependency = patch("hyper_parallel.components.ops.impl.deepseek_sparse_attention.require_package",
                           return_value=kernels)
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_deepseek_sparse_attention(registry)
        for spec, loader in ((DSA_SPARSE_PREFILL, _load_flash_prefill), (DSA_SPARSE_DECODE, _load_flash_decode)):
            registry.register_impl(spec, name="flash_mla", devices=("cpu",), load=loader)
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_prefill_and_quantized_decode(self) -> None:
        for case in flash_mla_cases(torch.bfloat16, hardware=True):
            with self.subTest(variant=case.variant, name=case.name):
                check_case(case)
                check_case(case, backend="flash_mla")
        self.assertIn("prefill", self.calls)
        self.assertIn("decode", self.calls)

    def test_forward_only_guard_precedes_native_calls(self) -> None:
        ops.set_backends({"deepseek_sparse_attention": "flash_mla"})
        query = torch.zeros(3, 64, 512, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "no backward"):
            flash_mla_sparse_prefill(query, torch.zeros(8, 1, 512), torch.zeros(3, 1, 4, dtype=torch.int32), 1.0)
        self.assertEqual(self.calls, [])

    def test_scheduler_reuse_and_length_mutation(self) -> None:
        ops.set_backends({"deepseek_sparse_attention": "flash_mla"})
        case = flash_mla_cases(torch.bfloat16)[2]
        query, cache, indices = case.args
        module = FlashMLASparseDecode()
        module(query, cache, indices)
        module(query, cache, indices)
        self.assertEqual(self.metadata.call_count, 1)
        lengths = torch.tensor([2], dtype=torch.int32)
        module(query, cache, indices, topk_length=lengths)
        lengths.fill_(1)
        module(query, cache, indices, topk_length=lengths)
        self.assertEqual(self.metadata.call_count, 3)
        module.enable_batch_invariant = True
        module(query, cache, indices)
        self.assertEqual(self.metadata.call_count, 4)
        module.to("cpu")
        module(query, cache, indices)
        self.assertEqual(self.metadata.call_count, 5)
        self.assertEqual(dict(module.state_dict()), {})

    def test_modules_native_backend_does_not_load_optional_package(self) -> None:
        ops.set_backends({"deepseek_sparse_attention": "native"})
        cases = flash_mla_cases(torch.bfloat16)
        with patch("hyper_parallel.components.ops.impl.deepseek_sparse_attention.require_package",
                   side_effect=AssertionError("native path imported optional backend")):
            FlashMLASparsePrefill(cases[0].args[3])(*cases[0].args[:3])
            FlashMLASparseDecode()(*cases[2].args)
