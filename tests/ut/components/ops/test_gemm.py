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
"""Native formulas and simulated DeepGEMM ABI; device acceptance is separate."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops import ops, gemm, m_grouped_gemm, mqa_logits
from hyper_parallel.components.ops.impl.gemm import GEMM, HEAD_EINSUM, load_gemm, load_head_einsum, register_gemm
from hyper_parallel.components.ops.impl.grouped_matmul import M_GROUPED_GEMM, K_GROUPED_GEMM, load_padded_grouped
from hyper_parallel.components.ops.impl.grouped_matmul import register_grouped_matmul
from hyper_parallel.components.ops.impl.mhc_pre import register_mhc_pre
from hyper_parallel.components.ops.impl.mhc_pre import MHC_PRENORM, load_prenorm
from hyper_parallel.components.ops.impl.dsa_indexer import register_dsa_indexer
from hyper_parallel.components.ops.impl.dsa_indexer import DSA_MQA, DSA_PAGED_MQA, load_mqa, load_paged_mqa
from hyper_parallel.components.ops.registry import OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_gemm import (
    gemm_cases,
    grouped_gemm_cases,
    prenorm_cases,
    mqa_cases,
    gemm_reference,
    grouped_reference,
    einsum_reference,
    prenorm_reference,
    _mqa_reference,
    _paged_oracle,
)


def fake_deep_gemm() -> Any:
    """Implement only documented entry points, returning None for output-buffer ABIs."""
    symbols = {"get_mk_alignment_for_contiguous_layout": lambda: 32}
    for layout in ("nt", "nn", "tn", "tt"):

        def dense(
            a: Any, b: Any, out: Any, *, c: Any=None, alpha: Any=None,
            recipe: Any=(1, 1, 32), order: Any=layout,
        ) -> Any:
            """Dense."""
            if c is not None and c.data_ptr() != out.data_ptr():
                if tuple(c.shape) != (1, out.shape[1]) or alpha is not None:
                    raise ValueError("Dense C must alias D for accumulation; separate bias cannot use alpha")
            out.copy_(
                gemm_reference(
                    a,
                    b,
                    layout=order,
                    c=c,
                    alpha=1.0 if alpha is None else alpha,
                    output_dtype=out.dtype,
                    recipe=recipe,
                )
            )

        symbols[f"bf16_gemm_{layout}"] = dense
        symbols[f"fp8_fp4_gemm_{layout}"] = dense
    for mode, layouts in (("m", ("nt", "nn")), ("k", ("tn", "nt"))):
        for layout in layouts:

            def grouped(
                a: Any,
                b: Any,
                out: Any,
                *args: Any,
                c: Any=None,
                use_psum_layout: Any=True,
                ensure_zero_padding: Any=True,
                recipe: Any=(1, 1, 32),
                kind: Any=mode,
                order: Any=layout,
            ) -> Any:
                """Grouped."""
                if not use_psum_layout or not ensure_zero_padding:
                    raise ValueError("Missing padded/zero-output contract")
                ends = args[-1]
                out.copy_(
                    grouped_reference(
                        a, b, ends, mode=kind, layout=order, alignment=32, c=c, output_dtype=out.dtype, recipe=recipe
                    )
                )

            if mode == "m" or layout == "tn":
                symbols[f"{mode}_grouped_bf16_gemm_{layout}_contiguous"] = grouped
            name = "fp8_fp4" if mode == "m" else "fp8"
            symbols[f"{mode}_grouped_{name}_gemm_{layout}_contiguous"] = grouped

    def einsum(expression: Any, a: Any, b: Any, out: Any, *, c: Any=None, recipe: Any=(1, 1, 32)) -> Any:
        """Einsum."""
        out.copy_(einsum_reference(a, b, expression, c=c, output_dtype=out.dtype, recipe=recipe))

    def prenorm(x: Any, w: Any, out: Any, squared: Any, *, num_splits: Any=None) -> Any:
        """Prenorm."""
        projection, expected = prenorm_reference(x, w, num_splits=num_splits)
        out.copy_(projection)
        squared.copy_(expected)

    def paged(
        q: Any,
        cache: Any,
        weights: Any,
        lengths: Any,
        table: Any,
        metadata: Any,
        max_length: Any,
        requests: Any,
        *,
        use_ascend_kv_layout: Any=False,
    ) -> Any:
        """Paged."""
        if metadata is None:
            raise ValueError("Missing scheduling metadata")
        values, mask = _paged_oracle(q, cache, weights, lengths, table, max_length, requests, use_ascend_kv_layout)
        return values.masked_fill(mask, 999.0)

    symbols.update(
        einsum=einsum,
        fp8_einsum=einsum,
        tf32_hc_prenorm_gemm=prenorm,
        fp8_fp4_mqa_logits=_mqa_reference,
        fp8_fp4_paged_mqa_logits=paged,
        get_paged_mqa_logits_metadata=lambda *args: torch.zeros(1, dtype=torch.int32),
    )
    return SimpleNamespace(**symbols)


class TestGemmPrecision(unittest.TestCase):
    """Check native formulas independently of optional backend installation."""

    def test_native_contracts(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn, torch.int8):
            for builder in (gemm_cases, grouped_gemm_cases, mqa_cases):
                for case in builder(dtype):
                    with self.subTest(dtype=dtype, name=case.name):
                        check_case(case)
        for dtype in (torch.float32, torch.bfloat16):
            for case in prenorm_cases(dtype):
                with self.subTest(dtype=dtype, name=case.name):
                    check_case(case)


class TestDeepGemmABI(unittest.TestCase):
    """Simulated ABI, gradient and input-immutability checks, not device validation."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        registry = OpRegistry()
        for register in (register_gemm, register_grouped_matmul, register_mhc_pre, register_dsa_indexer):
            register(registry)
        self.extension = fake_deep_gemm()
        for module in ("gemm", "grouped_matmul", "mhc_pre", "dsa_indexer"):
            dependency = patch(
                f"hyper_parallel.components.ops.impl.{module}.require_package", return_value=self.extension
            )
            dependency.start()
            self.addCleanup(dependency.stop)
        for spec, load in (
            (GEMM, load_gemm),
            (HEAD_EINSUM, load_head_einsum),
            (M_GROUPED_GEMM, lambda: load_padded_grouped("m")),
            (K_GROUPED_GEMM, lambda: load_padded_grouped("k")),
            (MHC_PRENORM, load_prenorm),
            (DSA_MQA, load_mqa),
            (DSA_PAGED_MQA, load_paged_mqa),
        ):
            registry.register_impl(spec, name="deep_gemm", devices=("cpu",), load=load)
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_dense_and_grouped_forward_backward(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.int8):
            for builder in (gemm_cases, grouped_gemm_cases, mqa_cases):
                for case in builder(dtype):
                    with self.subTest(dtype=dtype, name=case.name):
                        check_case(case, backend="deep_gemm")
        for case in prenorm_cases(torch.bfloat16):
            check_case(case, backend="deep_gemm")

    def test_fp32_grouped_output_rejects_training_before_launch(self):
        ops.set_backends({"grouped_matmul": "deep_gemm"})
        with self.assertRaisesRegex(RuntimeError, "BF16 output"):
            m_grouped_gemm(
                torch.ones(32, 64, dtype=torch.bfloat16, requires_grad=True),
                torch.ones(1, 64, 64, dtype=torch.bfloat16),
                torch.tensor([2], dtype=torch.int32),
                alignment=32,
                output_dtype=torch.float32,
            )

    def test_quantized_contract_rejects_gradient_loss(self):
        ops.set_backends({"gemm": "deep_gemm"})
        pair = (torch.ones(4, 64).to(torch.float8_e4m3fn), torch.ones(4, 2, requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            gemm(pair, pair)
        ops.set_backends({"dsa_indexer": "deep_gemm"})
        case = mqa_cases(torch.float8_e4m3fn)[0]
        args = list(case.args)
        args[2] = args[2].requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            mqa_logits(*args)

    def test_accumulation_and_bias_vjp(self):
        ops.set_backends({"gemm": "deep_gemm"})
        a = torch.ones(3, 64, dtype=torch.bfloat16, requires_grad=True)
        b = torch.ones(4, 64, dtype=torch.bfloat16, requires_grad=True)
        c = torch.ones(3, 4, requires_grad=True)
        result = gemm(a, b, c=c, alpha=0.5, output_dtype=torch.float32)
        result.sum().backward()
        torch.testing.assert_close(c.grad, torch.ones_like(c))
        torch.testing.assert_close(a.grad, torch.full_like(a, 2))
        torch.testing.assert_close(b.grad, torch.full_like(b, 1.5))
