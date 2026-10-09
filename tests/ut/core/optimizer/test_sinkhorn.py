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

"""Algorithm-reference and checkpoint tests for Sinkhorn."""

import copy
import unittest

from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.sinkhorn import Sinkhorn
from tests.common.mark_utils import arg_mark


def reference_balance(
        matrix: torch.Tensor, steps: int = 11, eps: float = 1e-20, tau: float = 1e-3,
) -> torch.Tensor:
    """Slow float64 reference with explicit row/column loops."""
    result = matrix.double().clone()
    norms = torch.linalg.vector_norm(result, dim=1)
    result[norms <= tau * norms.mean()] = 0
    for iteration in range(steps):
        if iteration % 2 == 0:
            result = torch.stack([row / (torch.linalg.vector_norm(row) + eps) for row in result])
        else:
            result = torch.stack([col / (torch.linalg.vector_norm(col) + eps) for col in result.T]).T
    return result * matrix.shape[1] ** 0.5


class TestSinkhorn(unittest.TestCase):
    """Check Algorithm 1 rather than another implementation of the optimizer path."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_multistep_reference_and_single_state(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: EMA, Nesterov, masking and LR correction match the report over multiple steps.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        torch.manual_seed(41)
        param = torch.nn.Parameter(torch.randn(7, 3))
        expected = param.detach().double().clone()
        momentum = torch.zeros_like(expected)
        optimizer = Sinkhorn([param], lr=0.03, momentum=0.8)
        for _ in range(4):
            gradient = torch.randn_like(param)
            gradient[0] = 0
            gradient[1] *= 1e-8
            param.grad = gradient
            momentum = 0.8 * momentum + 0.2 * gradient.double()
            expected -= 0.03 * 0.18 * reference_balance(0.8 * momentum + 0.2 * gradient.double())
            with torch.no_grad():
                optimizer._update_parameter(param, optimizer.param_groups[0])
            torch.testing.assert_close(param.double(), expected, rtol=1e-6, atol=2e-7)
            torch.testing.assert_close(optimizer.state[param]["momentum_buffer"].double(), momentum)
        self.assertEqual(set(optimizer.state[param]), {"momentum_buffer"})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_zero_rows_and_unit_row_rms(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Zero and tiny rows stay masked, while active rows have unit RMS.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        matrix = torch.tensor([[0., 0., 0.], [1e-9, 2e-9, 3e-9], [1., -3., 4.], [-2., 1., 5.]])
        param = torch.nn.Parameter(torch.zeros_like(matrix))
        param.grad = matrix
        optimizer = Sinkhorn([param], lr=1.0, correction=1.0, momentum=0.0)
        with torch.no_grad():
            optimizer._update_parameter(param, optimizer.param_groups[0])
        torch.testing.assert_close(param[:2], torch.zeros_like(param[:2]))
        torch.testing.assert_close(param[2:].square().mean(dim=1), torch.ones(2))
        with torch.no_grad():
            param.zero_()
            param.grad = torch.zeros_like(matrix)
            optimizer._update_parameter(param, optimizer.param_groups[0])
        torch.testing.assert_close(param, torch.zeros_like(matrix))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_update_preserves_noncontiguous_gradient(self):
        """
        Feature: Independent Sinkhorn workspace.
        Description: Update with transposed FP32 and BF16 gradients and one or several normalization steps.
        Expectation: The gradient stays unchanged and the parameter update matches the serial reference.
        """
        for dtype in (torch.float32, torch.bfloat16):
            for steps in (1, 3, 11):
                with self.subTest(dtype=dtype, steps=steps):
                    matrix = torch.tensor([[0., 1., 2., -3.], [0., -2., 4., 1.]], dtype=dtype).T
                    original = matrix.clone()
                    param = torch.nn.Parameter(torch.zeros_like(matrix))
                    param.grad = matrix
                    optimizer = Sinkhorn([param], lr=1.0, correction=1.0, momentum=0.0,
                                         steps=steps, eps=0.01, tau=0.5)
                    with torch.no_grad():
                        optimizer._update_parameter(param, optimizer.param_groups[0])
                    expected = -reference_balance(original, steps, eps=0.01, tau=0.5).to(dtype)
                    torch.testing.assert_close(param, expected)
                    torch.testing.assert_close(matrix, original, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_workspace_preserves_gradient_and_momentum(self):
        """
        Feature: Single-workspace momentum update.
        Description: Run FP32 and BF16 parameters with and without Nesterov over multiple steps.
        Expectation: Parameters follow the reference, gradients stay unchanged, and state remains the EMA.
        """
        for dtype in (torch.float32, torch.bfloat16):
            for nesterov in (False, True):
                with self.subTest(dtype=dtype, nesterov=nesterov):
                    torch.manual_seed(43)
                    param = torch.nn.Parameter(torch.randn(7, 3).to(dtype))
                    expected = param.detach().clone()
                    momentum = torch.zeros_like(param, dtype=torch.float64)
                    optimizer = Sinkhorn([param], lr=0.03, momentum=0.8, nesterov=nesterov)
                    for _ in range(4):
                        gradient = torch.randn_like(param)
                        original = gradient.clone()
                        param.grad = gradient
                        momentum = 0.8 * momentum + 0.2 * gradient.double()
                        update = 0.8 * momentum + 0.2 * gradient.double() if nesterov else momentum
                        expected.add_(reference_balance(update).float(), alpha=-0.03 * 0.18)
                        with torch.no_grad():
                            optimizer._update_parameter(param, optimizer.param_groups[0])
                        torch.testing.assert_close(param, expected)
                        torch.testing.assert_close(gradient, original, rtol=0, atol=0)
                        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"].double(), momentum)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_resume_and_absent_gradient(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Resuming an EMA state yields the uninterrupted trajectory.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        first = torch.nn.Parameter(torch.randn(5, 3))
        optimizer = Sinkhorn([first])
        first.grad = torch.randn_like(first)
        optimizer.step()
        second = torch.nn.Parameter(first.detach().clone())
        resumed = Sinkhorn([second])
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        first.grad = second.grad = torch.randn_like(first)
        optimizer.step()
        resumed.step()
        torch.testing.assert_close(first, second)
        before = second.detach().clone()
        resumed.zero_grad()
        resumed.step()
        torch.testing.assert_close(second, before)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_low_precision_resume_keeps_float32_momentum(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Loading must not retain PyTorch's default parameter-dtype state cast.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        param = torch.nn.Parameter(torch.randn(4, 3).bfloat16())
        optimizer = Sinkhorn([param])
        param.grad = torch.randn_like(param)
        optimizer.step()
        checkpoint = copy.deepcopy(optimizer.state_dict())
        optimizer.load_state_dict(checkpoint)
        self.assertEqual(optimizer.state[param]["momentum_buffer"].dtype, torch.float32)
        torch.testing.assert_close(optimizer.state[param]["momentum_buffer"],
                                   checkpoint["state"][0]["momentum_buffer"], rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_options_and_groups(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Reject invalid algorithm constants including per-group overrides.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        for options in ({"steps": 2}, {"steps": 0}, {"eps": 0}, {"tau": -1},
                        {"momentum": 1}, {"lr": float("nan")}, {"correction": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Sinkhorn([torch.nn.Parameter(torch.ones(3, 2))], **options)
        with self.assertRaisesRegex(ValueError, "weight decay"):
            Sinkhorn([{"params": [torch.nn.Parameter(torch.ones(3, 2))], "weight_decay": 0.1}])
        with self.assertRaisesRegex(ValueError, "2D"):
            Sinkhorn([torch.nn.Parameter(torch.ones(3))])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_column_shards_reduce_row_statistics(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Two identical column shards reproduce a full matrix without all-gather.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        local = torch.tensor([[1., 2.], [3., -2.], [1., 1.]])
        full = torch.cat([local, local], dim=1)
        param = torch.nn.Parameter(torch.zeros_like(full))
        param.grad = full
        optimizer = Sinkhorn([param], lr=1.0, correction=1.0, momentum=0.0)
        group = object()
        optimizer._norm_groups[param] = ((), (group,))
        local_tensors = {param: param[:, :2], param.grad: param.grad[:, :2]}
        with (
            torch.no_grad(),
            patch("hyper_parallel.core.optimizer.sinkhorn.to_local_if_dtensor",
                  side_effect=lambda tensor: local_tensors.get(tensor, tensor)),
            patch("hyper_parallel.core.optimizer.sinkhorn.dist.all_reduce",
                  side_effect=lambda value, group: value.mul_(2)) as reduce,
        ):
            optimizer._update_parameter(param, optimizer.param_groups[0])
        torch.testing.assert_close(param[:, :2], -reference_balance(full)[:, :2].float())
        self.assertEqual(reduce.call_count, 6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_row_shards_reduce_column_statistics_and_mask_mean(self):
        """
        Feature: Sinkhorn and composable head-wise optimization.
        Description: Repeated row partitions reduce the global mask mean and column norms.
        Expectation: Reference updates, routing, and state invariants match the assertions.
        """
        local = torch.tensor([[0., 0.], [1., 2.], [3., -2.]])
        full = local.repeat(2, 1)
        param = torch.nn.Parameter(torch.zeros_like(full))
        param.grad = full
        optimizer = Sinkhorn([param], lr=1.0, correction=1.0, momentum=0.0)
        group = object()
        optimizer._norm_groups[param] = ((group,), ())
        local_tensors = {param: param[:3], param.grad: param.grad[:3]}
        with (
            torch.no_grad(),
            patch("hyper_parallel.core.optimizer.sinkhorn.to_local_if_dtensor",
                  side_effect=lambda tensor: local_tensors.get(tensor, tensor)),
            patch("hyper_parallel.core.optimizer.sinkhorn.dist.all_reduce",
                  side_effect=lambda value, group: value.mul_(2)) as reduce,
        ):
            optimizer._update_parameter(param, optimizer.param_groups[0])
        torch.testing.assert_close(param[:3], -reference_balance(full)[:3].float())
        self.assertEqual(reduce.call_count, 6)
