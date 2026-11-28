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
"""Regression coverage for foundation-op ownership and migrated autograd bridges."""

import ast
from dataclasses import replace
import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul as functional_grouped_matmul
from hyper_parallel.components.functional.mhc_post import mhc_post as functional_mhc_post
from hyper_parallel.components.functional.rms_norm import rms_norm as functional_rms_norm
from hyper_parallel.components.functional.rotary_embedding import apply_rotary_pos_emb_interleave
from hyper_parallel.components.functional.sinkhorn import sinkhorn as functional_sinkhorn
from hyper_parallel.components.ops import ops
from hyper_parallel.components.ops.impl import (
    aggregate_hidden, grouped_matmul, grouped_swiglu, mhc_post, moe_router, moe_token_permute, moe_token_unpermute,
    rms_norm, rope, sinkhorn, swiglu,
)
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestFoundationAdapters(unittest.TestCase):
    """Keep functional above ops and retain the migrated kernel boundary contracts."""

    def setUp(self) -> None:
        """Restore process configuration and reject legacy functional imports in loaders."""
        previous = ops.get_backends()
        self.addCleanup(ops.set_backends, previous)
        ops.set_backends({})
        blocker = patch.dict(sys.modules, {
            f"hyper_parallel.components.functional.{name}": None
            for name in ("mhc_post", "sinkhorn", "rms_norm", "rotary_embedding", "grouped_matmul", "aggregate_hidden")
        })
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_a_families_do_not_depend_on_functional(self) -> None:
        """Check eager and delayed imports across all A-owned families."""
        for module in (aggregate_hidden, rms_norm, swiglu, rope, mhc_post, sinkhorn, grouped_matmul,
                       grouped_swiglu, moe_router, moe_token_permute, moe_token_unpermute):
            with self.subTest(module=module.__name__):
                tree = ast.parse(inspect.getsource(module))
                imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
                imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                            for alias in node.names]
                self.assertFalse([name for name in imports if name and "components.functional" in name])

    def test_functional_entries_use_public_selection_and_preserve_gradients(self) -> None:
        """Legacy signatures share native precision cases and register their actual backend use."""
        def gmm(x, weight, ends):
            return functional_grouped_matmul(x, weight, group_list=ends)

        entries = {
            "mhc_post": functional_mhc_post, "sinkhorn": functional_sinkhorn,
            "rms_norm": functional_rms_norm, "grouped_matmul": gmm,
        }
        for family, operation in entries.items():
            for case in precision_cases(family, torch.float32):
                if case.variant != "standard":
                    continue
                with self.subTest(family=family, case=case.name):
                    check_case(replace(case, operation=operation))
        for case in precision_cases("rope", torch.float32):
            if case.variant == "interleaved":
                with self.subTest(family="rope", case=case.name):
                    check_case(replace(case, operation=apply_rotary_pos_emb_interleave))

    def test_rms_norm_loads_kernel_without_functional(self) -> None:
        """The tuple-returning NPU kernel keeps epsilon and FP32-affine conversion."""
        kernel = Mock(side_effect=lambda x, weight, *, epsilon: (x * weight + epsilon, None))
        extension = SimpleNamespace(npu_rms_norm=kernel)
        x = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.full((4,), 2.0, dtype=torch.float32, requires_grad=True)
        with patch.object(rms_norm, "require_package", return_value=extension):
            operation = rms_norm._load_npu_fp32_affine()  # pylint: disable=protected-access
            output = operation(x, weight, 0.25)
        torch.testing.assert_close(output, torch.full_like(x, 2.25))
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        torch.testing.assert_close(weight.grad, torch.full_like(weight, 2))
        self.assertEqual(kernel.call_args.args[0].dtype, torch.float32)
        self.assertEqual(kernel.call_args.kwargs, {"epsilon": 0.25})

    def test_mhc_post_omni_backward_preserves_argument_order(self) -> None:
        """Reshaping residuals and reordering coefficients retain all four input VJPs."""
        x = torch.ones(1, 2, 3, requires_grad=True)
        residual = torch.ones(1, 2, 6, requires_grad=True)
        post = torch.ones(1, 2, 2, requires_grad=True)
        matrix = torch.ones(1, 2, 2, 2, requires_grad=True)
        custom = SimpleNamespace(
            npu_ai_infra_manifold_constrained_hyper_connection_post=Mock(
                side_effect=lambda streams, _matrix, _x, _post: streams.clone()),
            npu_ai_infra_mhc_post_grad=Mock(side_effect=lambda _grad, streams, mat, block, coeff: (
                torch.full_like(streams, 2), torch.full_like(mat, 3),
                torch.full_like(block, 4), torch.full_like(coeff, 5))),
        )
        with patch.object(mhc_post, "require_package"), patch.object(torch.ops, "custom", custom):
            operation = mhc_post._load_omni()  # pylint: disable=protected-access
            operation(x, residual, post, matrix, 2).sum().backward()
        for actual, value in ((x.grad, 4), (residual.grad, 2), (post.grad, 5), (matrix.grad, 3)):
            torch.testing.assert_close(actual, torch.full_like(actual, value))
        self.assertTrue(custom.npu_ai_infra_mhc_post_grad.call_args.args[0].is_contiguous())

    def test_sinkhorn_omni_backward_uses_saved_statistics(self) -> None:
        """The moved bridge forwards numerical settings and the two saved statistics."""
        x = torch.ones(2, 3, 3, requires_grad=True)
        norm, sums = torch.zeros_like(x), torch.ones_like(x)
        custom = SimpleNamespace(
            npu_sinkhorn=Mock(return_value=(x.detach().clone(), norm, sums)),
            npu_sinkhorn_grad=Mock(side_effect=lambda grad, _norm, _sums: grad * 2),
        )
        with patch.object(sinkhorn, "require_package"), patch.object(torch.ops, "custom", custom):
            operation = sinkhorn._load_omni()  # pylint: disable=protected-access
            operation(x, 7, 0.01).sum().backward()
        self.assertEqual(custom.npu_sinkhorn.call_args.kwargs, {"out_flag": 1, "eps": 0.01, "num_iters": 7})
        self.assertIs(custom.npu_sinkhorn_grad.call_args.args[1], norm)
        self.assertIs(custom.npu_sinkhorn_grad.call_args.args[2], sums)
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))

    def test_mome_aggregate_bridge_backward(self) -> None:
        """The low-level compatibility interface retains its mask and both differentiable inputs."""
        x = torch.ones(4, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        custom = SimpleNamespace(
            npu_aggregate_hidden=Mock(side_effect=lambda states, _weight, *, mask: states.clone()),
            npu_aggregate_hidden_grad=Mock(return_value=(torch.full_like(x, 2), torch.full_like(weight, 3))),
        )
        with patch.object(aggregate_hidden, "require_package"), patch.object(torch.ops, "custom", custom):
            operation = aggregate_hidden.load_omni_aggregate_hidden()
            operation(x, weight, mask).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        torch.testing.assert_close(weight.grad, torch.full_like(weight, 3))
        self.assertIs(custom.npu_aggregate_hidden_grad.call_args.kwargs["mask"], mask)

    def test_npu_interleaved_layout_and_gradients_without_functional(self) -> None:
        """Preserve partial rotation and the NPU sequence/batch layout conversion."""
        def kernel(x, cos, sin, *, rotary_mode):
            self.assertEqual(rotary_mode, "interleave")
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack((even * cos[..., ::2] - odd * sin[..., ::2],
                                odd * cos[..., 1::2] + even * sin[..., 1::2]), dim=-1).flatten(-2)

        extension = SimpleNamespace(npu_rotary_mul=kernel)
        for case in precision_cases("rope", torch.float32):
            if case.variant != "interleaved":
                continue
            with self.subTest(case=case.name), patch.object(rope, "require_package", return_value=extension):
                inputs = [arg.detach().clone().requires_grad_(arg.requires_grad) for arg in case.args]
                operation = rope._load_npu_interleaved()  # pylint: disable=protected-access
                expected = case.reference(*inputs, **case.kwargs)
                leaves = [arg for arg in inputs if arg.requires_grad]
                expected_grads = torch.autograd.grad(sum(value.sum() for value in expected), leaves)
                actual = operation(*inputs, **case.kwargs)
                actual_grads = torch.autograd.grad(sum(value.sum() for value in actual), leaves)
                for actual_value, expected_value in zip((*actual, *actual_grads), (*expected, *expected_grads)):
                    torch.testing.assert_close(actual_value, expected_value)
