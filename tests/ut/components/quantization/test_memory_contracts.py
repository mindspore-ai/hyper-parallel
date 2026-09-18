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
"""CPU regressions: identity quantization stands in for hardware, not numerical accuracy."""
import gc
import unittest
import weakref
from typing import Any, Callable
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.quantization.tensor.hifloat8_tensor import HiFloat8Tensor, HiFloat8TensorStorage
from hyper_parallel.components.quantization.functional.hifloat8_linear_func import hifloat8_linear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import hifloat8_grouped_linear
from hyper_parallel.components.quantization.functional.npu_hifloat8 import hifloat8_matmul
from hyper_parallel.components.quantization.modules.mxfp8_grouped_linear import MXFP8GroupedExperts
from hyper_parallel.components.quantization.modules.hifloat8_grouped_linear import HiFloat8GroupedExperts

from tests.common.mark_utils import arg_mark


class IdentityOps:
    """CPU arithmetic adapter; deliberately does not emulate low-precision rounding."""

    @staticmethod
    def quant_matmul(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor,
                     **kwargs: Any) -> torch.Tensor:
        """Multiply without quantization rounding."""
        del scale
        return (left.float() @ right.float()).to(kwargs['output_dtype'])

    @staticmethod
    def quant_grouped_matmul(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor,
                            *, group_list: torch.Tensor, group_type: int,
                            group_list_type: int, **kwargs: Any) -> torch.Tensor:
        """Evaluate expert-major projections with CPU matrix operations."""
        del scale
        counts = group_list if group_list_type == 1 else torch.diff(group_list, prepend=group_list.new_zeros(1))
        if group_type == 0:
            pieces = left.split(counts.tolist())
            return torch.cat([x.float() @ w.float() for x, w in zip(pieces, right)]).to(kwargs['output_dtype'])
        xs = left.split(counts.tolist(), dim=1)
        gs = right.split(counts.tolist(), dim=0)
        return torch.stack([x.float() @ g.float() for x, g in zip(xs, gs)]).to(kwargs['output_dtype'])


class IdentityQuantizer:
    """Retain real HiFloat8 wrappers while replacing only hardware quantization."""

    npu_ops = IdentityOps()

    def quantize(self, tensor: torch.Tensor, *, rowwise: bool, colwise: bool,
                 **kwargs: Any) -> HiFloat8Tensor:
        """Wrap a copy so weak references track the actual saved payload."""
        del kwargs
        data = tensor.detach().clone()
        scale = torch.ones(1)
        return HiFloat8Tensor(tensor.shape, tensor.dtype, quantizer=self,
            row_data=data if rowwise else None, col_data=data if colwise else None, scale=scale)


class MemoryContractsTests(unittest.TestCase):
    """Exercise storage, expert initialization and HiFloat8 autograd contracts."""

    def setUp(self) -> None:
        """Isolate random state from other unit tests."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    @staticmethod
    def make_projection(grouped: bool, need_x: bool = True, need_w: bool = True,
                        empty: bool = False) -> tuple[torch.Tensor, ...]:
        """Build a CPU projection using a real autograd Function."""
        torch.manual_seed(42)
        x = torch.randn(0 if empty else 5, 4, dtype=torch.bfloat16, requires_grad=need_x)
        w = torch.randn((2, 6, 4) if grouped else (6, 4), dtype=torch.bfloat16, requires_grad=need_w)
        q = IdentityQuantizer()
        if grouped:
            counts = torch.tensor([0, 0] if empty else [2, 3])
            y = hifloat8_grouped_linear(x, w, counts, q, q, q, group_list_type=1)
        else:
            y = hifloat8_linear(x, w, q, q, q)
        return x, w, y

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_repeat_backward_retained_graph(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Repeat first-order backward for Dense/GMM and gradient subsets.
        Expectation: Repeated gradients match and a freed graph rejects reuse.
        """
        for grouped in (False, True):
            for needs in ((True, True), (True, False), (False, True)):
                for empty in (False, True) if grouped else (False,):
                    with self.subTest(grouped=grouped, needs=needs, empty=empty):
                        x, w, y = self.make_projection(grouped, *needs, empty=empty)
                        variables = [v for v in (x, w) if v.requires_grad]
                        first = torch.autograd.grad(y.sum(), variables, retain_graph=True)
                        second = torch.autograd.grad(y.sum(), variables)
                        for a, b in zip(first, second):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        with self.assertRaises(RuntimeError):
                            torch.autograd.grad(y.sum(), variables)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_saved_tensors_are_released_after_backward(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Finish backward without retain_graph.
        Expectation: Saved operands are no longer accessible.
        """
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                _, _, y = self.make_projection(grouped)
                node = y.grad_fn
                self.assertTrue(any(isinstance(t, HiFloat8Tensor) for t in node.saved_tensors))
                y.sum().backward()
                with self.assertRaises(RuntimeError):
                    _ = node.saved_tensors

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_double_backward_is_explicitly_unsupported(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Request differentiable gradients and backpropagate through them.
        Expectation: Higher-order backward raises the once_differentiable error.
        """
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                x, _, y = self.make_projection(grouped)
                dy = torch.ones_like(y, requires_grad=True)
                dx, = torch.autograd.grad(y, x, dy, create_graph=True)
                with self.assertRaisesRegex(RuntimeError, 'once_differentiable'):
                    dx.sum().backward()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_payload_lifetime_follows_retained_graph(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Track saved payloads with weak references across two backwards.
        Expectation: Payloads survive the retained graph and are released after final backward.
        """
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                _, _, y = self.make_projection(grouped)
                saved = y.grad_fn.saved_tensors
                payloads = [weakref.ref(t.col_data if t.is_colwise() else t.row_data)
                            for t in saved if isinstance(t, HiFloat8Tensor)]
                del saved
                y.sum().backward(retain_graph=True)
                gc.collect()
                self.assertTrue(all(ref() is not None for ref in payloads))
                y.sum().backward()
                gc.collect()
                self.assertTrue(all(ref() is None for ref in payloads))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_direct_constructors_initialize_each_expert(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Poison fresh allocations with NaN before constructing experts.
        Expectation: All weights are finite, nonzero and within fan-in bounds.
        """
        for cls in (MXFP8GroupedExperts, HiFloat8GroupedExperts):
            with self.subTest(cls=cls):
                # Deterministic poison proves initialization; random torch.empty values do not.
                original_empty = torch.empty
                def poisoned(*args: Any, allocator: Callable = original_empty,
                             **kwargs: Any) -> torch.Tensor:
                    """Make uninitialized storage deterministic."""
                    return allocator(*args, **kwargs).fill_(float('nan'))
                with patch('torch.empty', side_effect=poisoned):
                    module = cls(2, 4, 6)
                for weight in module.parameters():
                    self.assertTrue(torch.isfinite(weight).all())
                    self.assertGreater(weight.abs().sum().item(), 0)
                    self.assertLessEqual(weight.abs().max().item(), weight.shape[-1] ** -0.5)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_from_module_keeps_weights_and_rng(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Convert existing expert parameters.
        Expectation: Parameter identity and RNG state are preserved.
        """
        for cls in (MXFP8GroupedExperts, HiFloat8GroupedExperts):
            with self.subTest(cls=cls):
                source = nn.Module()
                source.gate_up_proj = nn.Parameter(torch.randn(2, 12, 4))
                source.down_proj = nn.Parameter(torch.randn(2, 4, 6))
                source.act_fn = nn.SiLU()
                rng = torch.get_rng_state()
                target = cls.from_module(source, fqn='experts')
                self.assertIs(target.gate_up_proj, source.gate_up_proj)
                self.assertIs(target.down_proj, source.down_proj)
                torch.testing.assert_close(torch.get_rng_state(), rng)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_expert_ids_stop_before_gmm(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Pass negative and overflowing expert indices.
        Expectation: Both fail before grouped computation.
        """
        for cls in (MXFP8GroupedExperts, HiFloat8GroupedExperts):
            for index in (-1, 2):
                with self.subTest(cls=cls, index=index):
                    module = cls(2, 4, 6)
                    with patch.object(module, '_grouped_forward') as compute:
                        with self.assertRaises((ValueError, RuntimeError)):
                            module(torch.ones(1, 4), torch.tensor([[index]]), torch.ones(1, 1))
                        compute.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_missing_scale_is_rejected_at_construction(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Construct a carrier with an unpaired data or scale field.
        Expectation: Construction raises ValueError in either direction.
        """
        for cls in (HiFloat8TensorStorage, HiFloat8Tensor):
            for field in ('row_data', 'col_data', 'row_scale', 'col_scale'):
                with self.subTest(cls=cls, field=field):
                    with self.assertRaises(ValueError):
                        cls((2, 2), torch.bfloat16, quantizer=IdentityQuantizer(),
                            **{field: torch.ones(2, 2)})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_direction_requires_both_data_and_scale(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Remove a direction scale after construction.
        Expectation: The direction is unavailable and matmul rejects it.
        """
        q = IdentityQuantizer()
        tensor = q.quantize(torch.ones(2, 2), rowwise=True, colwise=True)
        tensor.row_scale = None
        self.assertFalse(tensor.is_rowwise())
        with self.assertRaises(ValueError):
            hifloat8_matmul(tensor, tensor, layout='NN')
        tensor.col_scale = None
        self.assertFalse(tensor.is_colwise())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_shared_scale_and_usage_still_work(self) -> None:
        """
        Feature: Low-precision memory contracts.
        Description: Clone a carrier and discard its directions.
        Expectation: Shared/separate scales remain valid and source usage is unchanged.
        """
        q = IdentityQuantizer()
        for kwargs in ({'scale':torch.ones(1)}, {'row_scale':torch.ones(1),'col_scale':torch.ones(1)}):
            tensor = HiFloat8Tensor((2, 2), torch.bfloat16, quantizer=q,
                row_data=torch.ones(2, 2, dtype=torch.uint8),col_data=torch.ones(2, 2, dtype=torch.uint8),**kwargs)
            self.assertTrue(tensor.is_rowwise() and tensor.is_colwise())
            clone = torch.detach(tensor).clone()
            clone.update_usage(rowwise=False)
            self.assertFalse(clone.is_rowwise())
            self.assertTrue(clone.is_colwise())
            self.assertTrue(tensor.is_rowwise())
            clone.update_usage(rowwise=False, colwise=False)
            self.assertIsNone(clone.scale)


if __name__ == '__main__':
    unittest.main()
