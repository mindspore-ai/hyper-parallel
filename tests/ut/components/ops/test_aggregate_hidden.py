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

"""Precision and selection contracts for the public hidden-state aggregation entry point."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.aggregate_hidden import aggregate_hidden as functional_aggregate_hidden
from hyper_parallel.components.ops import aggregate_hidden, ops
from hyper_parallel.components.ops.impl import aggregate_hidden as implementation
from hyper_parallel.components.ops.registry import ImplementationUnavailable
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestAggregateHidden(unittest.TestCase):
    """Check native mathematics and functional dispatch without accelerator requirements."""

    def setUp(self) -> None:
        """Keep process configuration and the resolution cache local to each test."""
        previous = ops.get_backends()
        self.addCleanup(ops.set_backends, previous)
        ops.set_backends({"aggregate_hidden": "native"})

    def test_native_and_functional_forward_and_gradients(self) -> None:
        """Both entries cover low precision, short windows, masking and empty input VJPs."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("aggregate_hidden", dtype):
                for entry in (aggregate_hidden, functional_aggregate_hidden):
                    with self.subTest(dtype=dtype, case=case.name, entry=entry):
                        check_case(replace(case, operation=entry))

    def test_fp64_gradcheck(self) -> None:
        """Both input and filter derivatives match finite differences in FP64."""
        generator = torch.Generator().manual_seed(36)
        x = torch.randn(3, 1, 2, generator=generator, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(2, 2, generator=generator, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, False, True]])
        self.assertTrue(torch.autograd.gradcheck(aggregate_hidden, (x, weight, mask)))

    def test_mask_only_gates_outputs_and_preserves_tap_order(self) -> None:
        """Masked source states still contribute to later unmasked output positions."""
        x = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1).requires_grad_()
        weight = torch.tensor([[10.0], [1.0]], requires_grad=True)
        actual = aggregate_hidden(x, weight, torch.tensor([[True, False, True]]))
        torch.testing.assert_close(actual.flatten(), torch.tensor([1.0, 0.0, 23.0]))
        actual.sum().backward()
        torch.testing.assert_close(x.grad.flatten(), torch.tensor([1.0, 10.0, 1.0]))
        torch.testing.assert_close(weight.grad.flatten(), torch.tensor([2.0, 4.0]))

    def test_functional_switches_native_omni_and_back(self) -> None:
        """Changing config invalidates resolution and runs the real bridge around mocked kernels."""
        def forward(x, weight, *, mask):
            return x * weight[0] * mask.T.unsqueeze(-1)

        def backward(gradient, x, weight, *, mask):
            gated = gradient * mask.T.unsqueeze(-1)
            return gated * weight[0], (gated * x).sum(dim=(0, 1)).unsqueeze(0)

        custom = SimpleNamespace(npu_aggregate_hidden=Mock(side_effect=forward),
                                 npu_aggregate_hidden_grad=Mock(side_effect=backward))
        fake_device = property(lambda _tensor: torch.device("npu:0"))
        for backend in ("native", "omni", "native"):
            ops.set_backends({"aggregate_hidden": backend})
            x = torch.ones(3, 1, 2, dtype=torch.bfloat16, requires_grad=True)
            weight = torch.full((1, 2), 2.0, dtype=torch.bfloat16, requires_grad=True)
            mask = torch.tensor([[True, False, True]])
            with patch.object(torch.Tensor, "device", fake_device), \
                    patch.object(implementation, "require_package"), patch.object(torch.ops, "custom", custom):
                output = functional_aggregate_hidden(x, weight, mask)
                output.sum().backward()
            torch.testing.assert_close(output.flatten(), torch.tensor([2, 2, 0, 0, 2, 2], dtype=x.dtype))
            torch.testing.assert_close(x.grad.flatten(), torch.tensor([2, 2, 0, 0, 2, 2], dtype=x.dtype))
            torch.testing.assert_close(weight.grad, torch.full_like(weight, 2))
            self.assertIn(("aggregate_hidden", "standard", backend), ops.used_backends())
        custom.npu_aggregate_hidden.assert_called_once()
        custom.npu_aggregate_hidden_grad.assert_called_once()

    def test_missing_omni_falls_back_only_for_auto(self) -> None:
        """Explicit selection errors are exposed; auto reports the native fallback."""
        x = torch.ones(3, 1, 2, dtype=torch.bfloat16)
        weight = torch.ones(1, 2, dtype=x.dtype)
        mask = torch.ones(1, 3, dtype=torch.bool)
        fake_device = property(lambda _tensor: torch.device("npu:0"))
        with patch.object(torch.Tensor, "device", fake_device), patch.object(
            implementation, "require_package", side_effect=ImplementationUnavailable("missing optional package omni"),
        ):
            ops.set_backends({"aggregate_hidden": "omni"})
            with self.assertRaisesRegex((ValueError, RuntimeError), "omni"):
                functional_aggregate_hidden(x, weight, mask)
            ops.set_backends({"aggregate_hidden": "auto"})
            output = functional_aggregate_hidden(x, weight, mask)
        torch.testing.assert_close(output, x)
        self.assertIn(("aggregate_hidden", "standard", "native"), ops.used_backends())

    def test_invalid_contract_is_rejected_before_dispatch(self) -> None:
        """Reject invalid ranks, widths, filters, mask types and mixed input/weight dtypes."""
        x = torch.ones(3, 2, 4)
        weight = torch.ones(2, 4)
        mask = torch.ones(2, 3, dtype=torch.bool)
        cases = ((x[0], weight, mask), (x, weight.unsqueeze(0), mask), (x, weight, mask.unsqueeze(0)),
                 (x, weight[:, :2], mask), (x, weight[:0], mask), (x, weight, mask[:, :2]),
                 (x, weight.double(), mask), (x, weight, mask.int()), (x.int(), weight.int(), mask),
                 (x, weight.to("meta"), mask))
        for inputs in cases:
            with self.subTest(shapes=[value.shape for value in inputs]):
                with self.assertRaisesRegex(ValueError, "aggregate_hidden requires"):
                    aggregate_hidden(*inputs)
