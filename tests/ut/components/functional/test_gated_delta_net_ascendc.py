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
# ==========================================================================
"""CPU contracts for optional native GDN metadata and raw-input ownership."""

from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional import gated_delta_net_ascendc as backend
from hyper_parallel.components.modules import gated_delta_net as module


class _NativeOps:
    """Record paired local calls without importing an NPU runtime."""

    def __init__(self) -> None:
        """Create a fresh call log for each CPU scenario."""
        self.calls = []

    def chunk_gated_delta_rule_fwd(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                   gate: torch.Tensor, beta: torch.Tensor, **kwargs: Any) -> tuple:
        """Provide required saved slots with distinct values per invocation."""
        del key, beta
        self.calls.append(("forward", kwargs))
        return (value.clone(), None, gate.clone(), query.clone(), None, None, None, None, None, None)

    def chunk_gated_delta_rule_bwd(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                   gate: torch.Tensor, beta: torch.Tensor, matrix: torch.Tensor,
                                   grad: torch.Tensor, scale: float, **kwargs: Any) -> tuple:
        """Return distinct gradients, exposing any beta/gate slot reversal."""
        del matrix, grad, scale
        self.calls.append(("backward", kwargs))
        return (torch.ones_like(query), torch.full_like(key, 2), torch.full_like(value, 3),
                torch.full_like(beta, 5), torch.full_like(gate, 4), None, None, None)


class TestAscendCGDN(unittest.TestCase):
    """Cover the new contract without NPU hardware or distributed setup."""

    @staticmethod
    def _inputs(tokens=128, batch=1):
        """Build effective core inputs; norm and nonlinear gates belong to the caller."""
        values = [torch.ones((batch, tokens, 2, 128), dtype=torch.bfloat16, requires_grad=True)
                  for _ in range(3)]
        return (*values, torch.zeros((batch, tokens, 2), requires_grad=True),
                torch.ones((batch, tokens, 2), dtype=torch.bfloat16, requires_grad=True))

    def test_dense_and_packed_keep_effective_input_gradient_order(self):
        """Native receives un-fused gates/norm and returns five gradients in public order."""
        for batch, offsets in ((2, None), (1, [0, 1, 64, 128])):
            ops = _NativeOps()
            inputs = self._inputs(batch=batch)
            with self.subTest(batch=batch, packed=offsets is not None), \
                 patch.object(backend, "_load_ops", return_value=ops):
                output, final = backend.chunk_gated_delta_rule_ascendc(*inputs, cu_seqlens=offsets)
                self.assertIsNone(final)
                output.sum().backward()
            for tensor, expected in zip(inputs, (1, 2, 3, 4, 5)):
                torch.testing.assert_close(tensor.grad, torch.full_like(tensor, expected))
            for _, options in ops.calls:
                self.assertEqual(options["layout"], "TND" if offsets else "BSND")
                for flag in ("use_qk_l2norm_in_kernel", "use_beta_sigmoid_in_kernel", "use_gate_in_kernel"):
                    self.assertFalse(options[flag])

    def test_packed_offsets_are_frozen_for_two_reverse_backwards(self):
        """Mutating caller-owned metadata cannot corrupt an earlier saved invocation."""
        offsets = torch.tensor([0, 1, 64, 128])
        second = [0, 63, 65, 128]
        ops = _NativeOps()
        with patch.object(backend, "_load_ops", return_value=ops):
            first, _ = backend.chunk_gated_delta_rule_ascendc(*self._inputs(), cu_seqlens=offsets)
            later, _ = backend.chunk_gated_delta_rule_ascendc(*self._inputs(), cu_seqlens=second)
            offsets[1] = 2
            second[1] = 62
            later.sum().backward()
            first.sum().backward()
        backward_calls = [options for kind, options in ops.calls if kind == "backward"]
        self.assertEqual(backward_calls[0]["cu_seqlens"], (0, 63, 65, 128))
        self.assertEqual(backward_calls[1]["cu_seqlens"], (0, 1, 64, 128))
        self.assertEqual(backward_calls[1]["chunk_indices"], (0, 0, 1, 0, 2, 0))
        for invalid in ([0, 0, 128], [0, 129], torch.tensor([0., 128.])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                backend._packed_metadata(invalid, 128)

    def test_public_dispatch_normalizes_once_and_rejects_unsupported_state(self):
        """Public guards fail before native, and Q/K each receive one external norm."""
        inputs = self._inputs()
        native = Mock(return_value=(inputs[2], None))
        with patch.object(backend, "chunk_gated_delta_rule_ascendc", native), \
             patch.object(module, "_l2norm", wraps=module._l2norm) as norm:
            module.chunk_gated_delta_rule(*inputs, backend="ascendc", use_qk_l2norm_in_kernel=True)
            self.assertEqual(norm.call_count, 2)
            native.assert_called_once()
            for options, error in (({"initial_state": torch.zeros(1)}, NotImplementedError),
                                   ({"output_final_state": True}, NotImplementedError),
                                   ({"chunk_size": 32}, ValueError)):
                native.reset_mock()
                with self.subTest(options=options), self.assertRaises(error):
                    module.chunk_gated_delta_rule(*inputs, backend="ascendc", **options)
                native.assert_not_called()
