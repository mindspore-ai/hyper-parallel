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
"""Native-only boundary, finite-difference and public-entry integration contracts."""

from dataclasses import replace
import unittest

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.functional.mome import mome
from hyper_parallel.components.modules.attention.linear_attention_utils.sequence_layout import PackedSequenceInfo
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import FAMILY_BUILDERS, precision_cases


class TestNativeContracts(unittest.TestCase):
    """Protect shape errors, exact zeros, FP64 derivatives and document resets."""

    def setUp(self) -> None:
        """Isolate process-wide backend settings for each test."""
        self.addCleanup(operators.ops.set_backends, operators.ops.get_backends())
        operators.ops.set_backends(dict.fromkeys(FAMILY_BUILDERS, "native"))

    def test_fp64_gradcheck(self) -> None:
        """Check analytical derivatives independently with numerical finite differences."""
        generator = torch.Generator().manual_seed(61)

        def tensor(*shape: int) -> torch.Tensor:
            """Build small differentiable FP64 inputs for numerical gradient checks."""
            return (torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2).requires_grad_()

        cases = [
            ("rms_norm", operators.rms_norm, (tensor(2, 4), tensor(4))),
            ("swiglu", operators.swiglu, (tensor(2, 8),)),
            ("rope", operators.rope, (tensor(1, 2, 4), tensor(1, 2, 4), tensor(1, 2, 4))),
            ("mhc_post", operators.mhc_post, (tensor(1, 2, 3), tensor(1, 2, 6), tensor(1, 2, 2),
                                               tensor(1, 2, 2, 2), 2)),
            ("sinkhorn", operators.sinkhorn, (tensor(1, 2, 2), 3)),
            ("grouped_matmul", operators.grouped_matmul, (tensor(3, 2), tensor(2, 2, 3), torch.tensor([1, 3]))),
            ("grouped_swiglu", operators.grouped_swiglu,
             (tensor(3, 2), tensor(2, 3, 2), tensor(2, 2, 3), tensor(2, 3, 2), torch.tensor([1, 2]), tensor(3))),
            ("mome", mome, (tensor(1, 3, 2), torch.tensor([[True, False, True]]),
                                      tensor(2, 1, 2), tensor(2))),
        ]
        for family, operation, args in cases:
            with self.subTest(family=family):
                self.assertTrue(torch.autograd.gradcheck(operation, args, fast_mode=True))

    def test_swiglu_rejects_odd_packed_dimensions(self) -> None:
        """Invalid gate/up packing must fail before accidentally broadcasting halves."""
        with self.assertRaisesRegex(ValueError, "evenly packed"):
            operators.swiglu(torch.ones(2, 5))

    def test_clamp_saturates_gate_and_up_derivatives(self) -> None:
        """Gate has an upper clamp; up has both lower and upper clamps."""
        gate = torch.tensor([-3.0, 0.5, 3.0], requires_grad=True)
        up = torch.tensor([-3.0, 0.5, 3.0], requires_grad=True)
        operators.swiglu_clamped(gate, up, 1.0).sum().backward()
        self.assertEqual(gate.grad[-1].item(), 0.0)
        torch.testing.assert_close(up.grad[[0, 2]], torch.zeros(2), atol=0, rtol=0)
        self.assertNotEqual(gate.grad[0].item(), 0.0)

    def test_rope_identity_and_invalid_prefix(self) -> None:
        """An empty frequency prefix preserves values and gradients, including the tail."""
        query = torch.arange(24.0).reshape(1, 2, 3, 4).requires_grad_()
        empty = torch.empty(1, 1, 3, 0)
        output = operators.rope(query, empty, empty)
        torch.testing.assert_close(output, query, atol=0, rtol=0)
        output.sum().backward()
        torch.testing.assert_close(query.grad, torch.ones_like(query), atol=0, rtol=0)
        for width in (3, 6):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "even frequency"):
                operators.rope(query, torch.ones(1, 1, 3, width), torch.ones(1, 1, 3, width))

    def test_interleaved_rope_sequence_major_layout(self) -> None:
        """The two public layout conventions share the same partial-prefix contract."""
        case = precision_cases("rope", torch.float32)[1]
        query, key, cosine, sine = case.args
        check_case(replace(case, args=(query.transpose(1, 2), key.transpose(1, 2), cosine, sine),
                           kwargs={"unsqueeze_dim": 2}))

    def test_grouped_matmul_rejects_invalid_boundaries(self) -> None:
        """Offsets must be ordered, cover all tokens and contain one end per expert."""
        for ends in ([1], [2, 2], [4, 3]):
            with self.subTest(ends=ends), self.assertRaisesRegex(ValueError, "boundaries"):
                operators.grouped_matmul(torch.ones(3, 4), torch.ones(2, 4, 5), torch.tensor(ends))

    def test_masked_mome_has_zero_parameter_gradients(self) -> None:
        """An all-false update mask preserves the residual exactly and returns zero parameter gradients."""
        x = torch.ones(1, 4, 2, requires_grad=True)
        weight = torch.ones(2, 1, 3, requires_grad=True)
        bias = torch.ones(2, requires_grad=True)
        output = mome(x, torch.zeros(1, 4, dtype=torch.bool), weight, bias)
        torch.testing.assert_close(output, x, atol=0, rtol=0)
        output.sum().backward()
        for parameter in (weight, bias):
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), atol=0, rtol=0)
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)

    def test_sinkhorn_is_positive_and_doubly_stochastic(self) -> None:
        """Large, finite logits must produce finite nonnegative approximately normalized matrices."""
        logits = torch.tensor([[[40.0, -40.0], [-40.0, 40.0]]], requires_grad=True)
        output = operators.sinkhorn(logits)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output >= 0).all())
        for dim in (-1, -2):
            torch.testing.assert_close(output.sum(dim), torch.ones(1, 2), atol=3e-6, rtol=0)

    def test_router_top1_keeps_unnormalized_scores(self) -> None:
        """The top-1 contract scales the selected raw score rather than normalizing it to one."""
        case = precision_cases("moe_router", torch.float32)[0]
        args = (*case.args[:5], 1, *case.args[6:])
        check_case(replace(case, args=args))
        with self.assertRaisesRegex(ValueError, "scoring function"):
            operators.moe_router(*args[:6], "invalid", args[-1])

    def test_token_permutation_roundtrip_and_weight_gradients(self) -> None:
        """Repeated expert slots restore each original token and accumulate its route gradients."""
        states = torch.arange(12.0).reshape(3, 4).requires_grad_()
        indices = torch.tensor([[2, 0], [1, 2], [0, 1]])
        weights = torch.full((3, 2), 0.5, requires_grad=True)
        permuted, _, positions = operators.moe_token_permute(states, indices)
        restored = operators.moe_token_unpermute(permuted, positions, weights)
        torch.testing.assert_close(restored, states, atol=0, rtol=0)
        restored.sum().backward()
        torch.testing.assert_close(states.grad, torch.ones_like(states), atol=0, rtol=0)
        torch.testing.assert_close(weights.grad, states.detach().sum(-1, keepdim=True).expand_as(weights))

    def test_causal_convolution_packed_dilation(self) -> None:
        """Dilated convolution must not read previous documents at any tap."""
        for case in precision_cases("causal_conv1d", torch.float32):
            with self.subTest(case=case.name):
                check_case(replace(case, kwargs={"dilation": 2}))

    def test_delta_rules_across_chunk_boundary(self) -> None:
        """A 65-token sequence exercises carry and a one-token tail with an independent recurrence."""
        for family in ("gdn", "kda"):
            case = precision_cases(family, torch.float32)[0]
            args = tuple(tensor.repeat(1, 8, *([1] * (tensor.ndim - 2)))[:, :65] for tensor in case.args)
            for packed in (False, True):
                with self.subTest(family=family, packed=packed):
                    info = PackedSequenceInfo((0, 3, 65), 0, 65) if packed else None
                    check_case(replace(case, args=args, kwargs={**case.kwargs, "info": info}))

    def test_gdn_without_qk_normalization(self) -> None:
        """Disabling normalization preserves the public effective-gate contract."""
        case = precision_cases("gdn", torch.float32)[0]
        check_case(replace(case, kwargs={**case.kwargs, "normalize_qk": False}))

    def test_dsa_invalid_inputs_and_zero_loss(self) -> None:
        """Validate compressed geometry and ensure a disabled auxiliary loss cannot leak teacher gradients."""
        case = precision_cases("dsa_indexer", torch.float32)[0]
        for kwargs in ({"query_chunk_size": 0}, {"layout": operators.DsaQueryLayout(compress_ratio=0)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "positive"):
                operators.dsa_indexer(*case.args, **{**case.kwargs, **kwargs})
        case = precision_cases("dsa_indexer_kl", torch.float32)[0]
        loss = operators.dsa_indexer_kl(*case.args, **{**case.kwargs, "loss_coeff": 0.0})
        self.assertEqual(loss.item(), 0.0)
        gradients = torch.autograd.grad(loss, case.args[:5] + (case.args[6],), allow_unused=True)
        torch.testing.assert_close(gradients[0], torch.zeros_like(case.args[0]), atol=0, rtol=0)
        self.assertTrue(all(gradient is None for gradient in gradients[1:]))
