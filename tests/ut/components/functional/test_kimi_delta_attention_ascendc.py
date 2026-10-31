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
"""CPU checks for the native KDA norm ABI and per-call packed activations."""

from __future__ import annotations

from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional import kimi_delta_attention_ascendc as backend


class _NativeOps:
    """Model native tensor contracts without loading a device runtime."""

    def __init__(self) -> None:
        """Record each call without retaining any external runtime state."""
        self.calls = []
        self.prepared = None

    def chunk_kda_fwd_prepare(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                              gate: torch.Tensor, beta: torch.Tensor, scale: float, *, backward_mode: str,
                              use_qk_l2norm_in_kernel: bool, use_beta_sigmoid_in_kernel: bool,
                              **kwargs: Any) -> list[torch.Tensor]:
        """Return distinct norm auxiliaries for each invocation."""
        del value, gate, scale, backward_mode, use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel, kwargs
        self.prepared = [torch.zeros_like(query) for _ in range(13)]
        self.prepared[0] = torch.zeros_like(query, dtype=torch.float32)
        self.prepared[8:13] = [query.detach() / 2, key.detach() / 2,
                               query.detach().float().mean(-1), key.detach().float().mean(-1),
                               torch.sigmoid(beta.float())]
        return self.prepared

    def chunk_fwd_h(self, key: torch.Tensor, weights: torch.Tensor, residual: torch.Tensor, *,
                    output_final_state: bool, cu_seqlens: Any = None, chunk_indices: Any = None,
                    **kwargs: Any) -> tuple:
        """Provide the state tensors expected by the wrapper."""
        del weights, residual, output_final_state, cu_seqlens, chunk_indices, kwargs
        return torch.zeros_like(key), torch.zeros_like(key), None

    def chunk_kda_fwd_finalize(self, query: torch.Tensor, aqk: torch.Tensor, v_new: torch.Tensor,
                               hidden: torch.Tensor, *, output_layout: str, cu_seqlens: Any = None,
                               chunk_indices: Any = None) -> torch.Tensor:
        """Return the public sequence-major output view."""
        del aqk, v_new, hidden, output_layout, cu_seqlens, chunk_indices
        return query.transpose(1, 2).contiguous()

    def chunk_kda_bwd(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor,
                      *args: Any, q_rstd: torch.Tensor, k_rstd: torch.Tensor, implementation: str,
                      disable_recompute: bool, layout: str, beta_eff: torch.Tensor | None,
                      use_beta_sigmoid_in_kernel: bool, **kwargs: Any) -> tuple:
        """Return raw Q/K gradients and effective-beta gradients."""
        del args
        self.calls.append({"q_rstd": q_rstd, "k_rstd": k_rstd, "implementation": implementation,
                           "disable_recompute": disable_recompute, "layout": layout, "beta_eff": beta_eff,
                           "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
                           **kwargs})
        return (q_rstd.unsqueeze(-1).expand_as(query), k_rstd.unsqueeze(-1).expand_as(key),
                torch.full_like(value, 3), torch.full_like(beta, 7),
                torch.full_like(kwargs["raw_g"], 5), None,
                torch.full_like(kwargs["A_log"], 6), torch.full_like(kwargs["dt_bias"], 8))


class TestAscendCKDA(unittest.TestCase):
    """Keep norm ownership, packed shape and lifecycle checks independent of NPU."""

    def setUp(self) -> None:
        """Keep runtime capability caches local to each test."""
        backend._load_ops.cache_clear()
        backend._use_native_beta_backward.cache_clear()

    def tearDown(self) -> None:
        """Discard capabilities learned from test doubles."""
        backend._load_ops.cache_clear()
        backend._use_native_beta_backward.cache_clear()

    @staticmethod
    def _inputs(marker, batch=1, logit_dtype=torch.bfloat16):
        tokens = [torch.full((batch, 3, 5, 128), marker, dtype=torch.bfloat16, requires_grad=True)
                  for _ in range(3)]
        gate = torch.full((batch, 5, 3, 128), marker, dtype=logit_dtype, requires_grad=True)
        beta = torch.full((batch, 5, 3), marker, dtype=logit_dtype, requires_grad=True)
        return (*tokens, gate, beta, torch.ones(3, requires_grad=True), torch.ones(3, 128, requires_grad=True))

    def test_native_norm_preserves_dense_and_packed_gradients(self):
        """Each rstd pair reaches native at the token rank and beta is differentiated once."""
        cases = [(packed, batch, dtype) for dtype in (torch.bfloat16, torch.float32)
                 for packed, batch in ((False, 1), (False, 2), (True, 1))]
        for packed, batch, logit_dtype in cases:
            metadata = backend._packed_metadata([0, 1, 5], 5) if packed else {}
            with self.subTest(packed=packed, batch=batch, logit_dtype=logit_dtype):
                ops = _NativeOps()
                inputs = self._inputs(2.0, batch, logit_dtype)
                sigmoid = Mock(side_effect=lambda raw, gradient: raw * gradient)
                with patch.object(backend, "_load_ops", return_value=ops), \
                     patch.object(backend, "_use_native_beta_backward", return_value=False), \
                     patch.object(backend, "_load_beta_backward", return_value=sigmoid):
                    output = backend._AscendCKDA.apply(*inputs, 0.1, -5.0, metadata)
                    output.sum().backward()
                call = ops.calls[0]
                expected_shape = (3, 5) if metadata else (batch, 3, 5)
                self.assertEqual(tuple(call["q_rstd"].shape), expected_shape)
                self.assertEqual(tuple(call["k_rstd"].shape), expected_shape)
                self.assertEqual(call["layout"], "NTD" if metadata else "BNSD")
                self.assertIsNone(call["beta_eff"])
                self.assertEqual(call["implementation"], "auto")
                self.assertTrue(call["disable_recompute"])
                self.assertFalse(call["use_beta_sigmoid_in_kernel"])
                sigmoid.assert_called_once()
                self.assertIs(sigmoid.call_args.args[0], inputs[4])
                self.assertEqual(call["raw_g"].dtype, logit_dtype)
                self.assertEqual(call["q_rstd"].dtype, torch.float32)
                for tensor, expected in zip(inputs, (2, 2, 3, 5, 14, 6, 8)):
                    torch.testing.assert_close(tensor.grad, torch.full_like(tensor, expected))

    def test_reverse_backward_owns_distinct_packed_norm_auxiliaries(self):
        """Two in-flight packed calls retain independent norm tensors and boundaries."""
        ops = _NativeOps()
        inputs = [self._inputs(marker) for marker in (2.0, 3.0)]
        metadata = [backend._packed_metadata(offsets, 5) for offsets in ([0, 1, 5], [0, 2, 5])]
        with patch.object(backend, "_load_ops", return_value=ops), \
             patch.object(backend, "_use_native_beta_backward", return_value=False), \
             patch.object(backend, "_load_beta_backward", return_value=lambda raw, grad: raw * grad):
            outputs = [backend._AscendCKDA.apply(*values, 0.1, -5.0, info)
                       for values, info in zip(inputs, metadata)]
            for output in reversed(outputs):
                output.sum().backward()
        for call, info, marker in zip(ops.calls, reversed(metadata), (3, 2)):
            self.assertEqual(call["cu_seqlens"], info["cu_seqlens"])
            torch.testing.assert_close(call["q_rstd"], torch.full((3, 5), marker, dtype=torch.float32))
        for values, marker in zip(inputs, (2, 3)):
            torch.testing.assert_close(values[0].grad, torch.full_like(values[0], marker))

    def test_prepare_releases_only_forward_workspaces_and_rejects_missing_aux(self):
        """Save mode must expose every required slot before state computation."""
        ops = _NativeOps()
        with patch.object(backend, "_load_ops", return_value=ops):
            _, prepared, _, _ = backend._forward_kda(self._inputs(2.0), 0.1, -5.0, {})
        self.assertIsNone(prepared[4])
        self.assertIsNone(prepared[7])
        for index in (3, 5, 6, 8, 9, 10, 11, 12):
            self.assertIs(prepared[index], ops.prepared[index])
        ops.chunk_kda_fwd_prepare = Mock(return_value=[None] * 13)
        ops.chunk_fwd_h = Mock()
        with patch.object(backend, "_load_ops", return_value=ops), \
             self.assertRaisesRegex(RuntimeError, "13 tensors"):
            backend._forward_kda(self._inputs(2.0), 0.1, -5.0, {})
        ops.chunk_fwd_h.assert_not_called()

    def test_capability_check_rejects_old_backward_before_execution(self):
        """Callable symbols alone cannot admit the old incompatible backward API."""
        ops = _NativeOps()
        with patch.object(backend.importlib, "import_module", return_value=ops):
            self.assertIs(backend._load_ops(), ops)
            backend._load_ops.cache_clear()
            ops.chunk_kda_bwd = lambda *args, **kwargs: None
            with self.assertRaisesRegex(RuntimeError, "chunk_kda_bwd lacks required parameters"):
                backend._load_ops()

    def test_fused_beta_uses_raw_logits_and_fp32_cache_without_external_chain(self):
        """The fused path returns raw-beta gradients and cannot apply sigmoid twice."""
        for offsets in (None, [0, 2, 5]):
            ops = _NativeOps()
            inputs = self._inputs(2.0)
            metadata = backend._packed_metadata(offsets, 5)
            with patch.object(backend, "_load_ops", return_value=ops), \
                 patch.object(backend, "_use_native_beta_backward", return_value=True), \
                 patch.object(backend, "_load_beta_backward") as external:
                output = backend._AscendCKDA.apply(*inputs, 0.1, -5.0, metadata)
                output.sum().backward()
            external.assert_not_called()
            call = ops.calls[0]
            self.assertTrue(call["use_beta_sigmoid_in_kernel"])
            self.assertEqual(call["beta_eff"].dtype, torch.float32)
            self.assertEqual(tuple(call["raw_g"].shape), (5, 3, 128) if offsets else (1, 5, 3, 128))
            torch.testing.assert_close(inputs[4].grad, torch.full_like(inputs[4], 7))
