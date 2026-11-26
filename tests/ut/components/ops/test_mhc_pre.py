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
"""Complete Pre and stage precision; upstream ABI simulations are CPU-only."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from examples.torch.module_family.config import MHCPreConfig, PipelinedMhcConfig
from hyper_parallel.components.modules import MHCPre, PipelinedMhc, MhcPreModule, PipelinedMhcModule
from hyper_parallel.components.ops import mhc_pre, mhc_pre_apply, mhc_split_mixes, ops
from hyper_parallel.components.ops.impl.mhc_pre import (
    MHC_PRE, MHC_PRE_APPLY, MHC_SPLIT, _load_tile_apply, _load_tile_pre, _load_tile_split, register_mhc_pre,
)
from hyper_parallel.components.ops.impl.sinkhorn import SINKHORN, _load_tile_kernels, register_sinkhorn
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec
from tests.common.op_precision import assert_precision, check_case
from tests.common.op_precision_cases import _mhc_apply, _mhc_pre, _mhc_split, _sinkhorn
from tests.common.op_precision_suite import precision_cases


def _vjp(function, arguments, gradients):
    with torch.enable_grad():
        leaves = [arg.detach().requires_grad_() for arg in arguments]
        return torch.autograd.grad(function(*leaves), leaves, gradients)


class TestMhcPrePrecision(unittest.TestCase):
    """Native contracts retain all projection, coefficient and activation gradients."""

    def test_native_forward_and_gradients(self) -> None:
        """Validate full Pre and both reusable stages against independent formulas."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("mhc_pre", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)


class TestTileKernelsMhcPre(unittest.TestCase):
    """Exercise the raw upstream ABI on CPU without claiming NPU kernel validation."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        self.calls = []
        kernels = SimpleNamespace(
            mhc_pre_split_mixes_fwd=self.split_forward, mhc_pre_split_mixes_bwd=self.split_backward,
            mhc_pre_apply_mix_fwd=self.apply_forward, mhc_pre_apply_mix_bwd=self.apply_backward,
            mhc_sinkhorn_fwd=self.sinkhorn_forward, mhc_sinkhorn_bwd=self.sinkhorn_backward,
        )
        for module in ("mhc_pre", "sinkhorn"):
            dependency = patch(f"hyper_parallel.components.ops.impl.{module}.require_package", return_value=kernels)
            dependency.start()
            self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_mhc_pre(registry)
        register_sinkhorn(registry)
        for spec, loader in ((MHC_PRE, _load_tile_pre), (MHC_SPLIT, _load_tile_split),
                             (MHC_PRE_APPLY, _load_tile_apply), (SINKHORN, _load_tile_kernels)):
            registry.register_impl(spec, name="tile_kernels", devices=("cpu",), load=loader)
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)

    def split_forward(
        self,
        mixes: Any,
        scale: Any,
        base: Any,
        pre: Any,
        post: Any,
        residual: Any,
        multiplier: Any,
        epsilon: Any,
    ) -> Any:
        """Simulate the upstream buffer-writing coefficient ABI."""
        self.assertEqual(mixes.shape[-1], 24)
        self.assertTrue(all(t.is_contiguous() for t in (mixes, scale, base)))
        for buffer, value in zip((pre, post, residual), _mhc_split(
                mixes, scale, base, 4, post_multiplier=multiplier, epsilon=epsilon)):
            buffer.copy_(value.reshape_as(buffer))
        self.calls.append("split_forward")

    def split_backward(self, pre_grad: Any, post_grad: Any, res_grad: Any, mixes: Any, post: Any, scale: Any, base: Any,
                       mixes_grad: Any, scale_grad: Any, base_grad: Any, multiplier: Any) -> Any:
        """Simulate all three input gradients, including shared parameter reductions."""
        self.assertEqual(post.shape, pre_grad.shape)

        def reference(*args: Any) -> Any:
            """Reference."""
            return _mhc_split(*args, 4, post_multiplier=multiplier)

        gradients = _vjp(reference, (mixes, scale, base), (pre_grad, post_grad, res_grad.unflatten(-1, (4, 4))))
        for buffer, value in zip((mixes_grad, scale_grad, base_grad), gradients):
            buffer.copy_(value)
        self.calls.append("split_backward")

    def apply_forward(self, hidden: Any, pre: Any, output: Any) -> Any:
        """Require the upstream explicit-stream layout."""
        self.assertEqual(hidden.ndim, 3)
        self.assertEqual(hidden.dtype, torch.bfloat16)
        output.copy_(_mhc_apply(hidden, pre))
        self.calls.append("apply_forward")

    def apply_backward(self, gradient: Any, hidden: Any, pre: Any, hidden_grad: Any) -> Any:
        """Use add_ to catch a missing zero initialization in the adapter."""
        self.assertEqual(torch.count_nonzero(hidden_grad).item(), 0)
        grad_hidden, grad_pre = _vjp(_mhc_apply, (hidden, pre), gradient)
        hidden_grad.add_(grad_hidden)
        self.calls.append("apply_backward")
        return grad_pre

    @staticmethod
    def sinkhorn_forward(x: Any, output: Any, iterations: Any, eps: Any) -> Any:
        """Simulate normalization using its independent reference."""
        output.copy_(_sinkhorn(x, iterations, eps))

    @staticmethod
    def sinkhorn_backward(gradient: Any, x: Any, result: Any, iterations: Any, eps: Any) -> Any:
        """Preserve the raw backward's output buffer contract."""
        result.copy_(_vjp(lambda value: _sinkhorn(value, iterations, eps), (x,), gradient)[0])

    def test_shared_forward_and_gradient_cases(self) -> None:
        """Check complete Pre, gamma and all activation/parameter gradients."""
        for case in precision_cases("mhc_pre", torch.bfloat16, backend="tile_kernels"):
            with self.subTest(variant=case.variant, case=case.name):
                check_case(case)
                check_case(case, backend="tile_kernels")
        for name in ("split_forward", "split_backward", "apply_forward", "apply_backward"):
            self.assertIn(name, self.calls)

    def test_independent_coefficient_outputs_and_noncontiguous_mixes(self) -> None:
        """Unused outputs must yield zero upstream VJPs, with no dropped parameter gradients."""
        for output in range(3):
            actual = [torch.randn(2, 3, 48)[..., ::2].requires_grad_(),
                      torch.randn(3, requires_grad=True), torch.randn(24, requires_grad=True)]
            reference = [t.detach().clone().requires_grad_() for t in actual]
            ops.set_backends({"mhc_pre": "tile_kernels"})
            selected = mhc_split_mixes(*actual, 4)[output]
            expected = _mhc_split(*reference, 4)[output]
            grad = torch.randn_like(expected)
            for value, target in zip(torch.autograd.grad(selected, actual, grad),
                                     torch.autograd.grad(expected, reference, grad)):
                torch.testing.assert_close(value, target)

    def test_pre_operator_checkpoint_and_update(self) -> None:
        """Standalone Pre preserves projection parameters and their updates under recompute."""
        member = MHCPre(config=MHCPreConfig(4, 128, use_gamma=True, sinkhorn_iters=3))
        x = torch.randn(2, 3, 512, dtype=torch.bfloat16, requires_grad=True)
        named = dict(member.named_parameters())
        parameters = list(named.values())
        reference = {name: value.detach().clone().requires_grad_() for name, value in {"input": x, **named}.items()}
        ops.set_backends({"mhc_pre": "tile_kernels"})
        def run(hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return mhc_pre(hidden, member.phi.weight, member.branch_alpha, member.branch_beta,
                           4, sinkhorn_iters=3, gamma=member.norm_gamma)

        actual = checkpoint(run, x, use_reentrant=False)
        expected = _mhc_pre(reference["input"], reference["phi.weight"], reference["branch_alpha"],
                            reference["branch_beta"], 4, sinkhorn_iters=3, gamma=reference["norm_gamma"])
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target)
        gradients = [torch.randn_like(tensor) for tensor in expected]
        values = torch.autograd.grad(actual, (x, *parameters), gradients)
        targets = torch.autograd.grad(expected, tuple(reference.values()), gradients)
        for value, target in zip(values, targets):
            assert_precision(value, target, dtype=torch.bfloat16, backward=True)
        with torch.no_grad():
            for name, grad, target_grad in zip(named, values[1:], targets[1:]):
                parameter, target = named[name], reference[name]
                parameter.add_(grad, alpha=-0.01)
                target.add_(target_grad, alpha=-0.01)
                torch.testing.assert_close(parameter, target, atol=2e-3, rtol=2e-3)
        self.assertIn(("mhc_pre", "standard", "tile_kernels"), ops.used_backends())

    def test_pipelined_coefficients_with_ops_aggregation(self) -> None:
        """Compose existing eager coefficients with the standalone aggregation operator."""
        member = PipelinedMhc(config=PipelinedMhcConfig(4, 128, sinkhorn_iters=3))
        hidden = torch.randn(2, 3, 4, 128, dtype=torch.bfloat16, requires_grad=True)
        supplied = torch.randn(2, 3, 4, requires_grad=True)
        ops.set_backends({"mhc_pre": "tile_kernels", "sinkhorn": "tile_kernels"})
        coefficients = member(hidden)
        actual = (mhc_pre_apply(hidden, supplied), *coefficients)
        self.assertEqual(len(coefficients), 3)
        self.assertEqual(len(actual), 4)
        normalized = hidden.flatten(2).float()
        normalized = normalized / (normalized.square().mean(-1, keepdim=True) + member.input_norm.eps).sqrt()
        mixes = (normalized * member.input_norm.weight) @ member.fn.T
        next_pre, post, residual = _mhc_split(mixes, member.scale, member.base, 4, epsilon=member.hc_eps)
        expected = (_mhc_apply(hidden, supplied), next_pre, post, _sinkhorn(residual, 3, member.hc_eps))
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target)
        for value, target in zip(actual[1:], coefficients):
            torch.testing.assert_close(value, target)
        gradient = [torch.randn_like(tensor) for tensor in expected]
        parameters = (hidden, supplied, *member.parameters())
        actual_grads = torch.autograd.grad(actual, parameters, gradient)
        expected_grads = torch.autograd.grad(expected, parameters, gradient)
        for value, target in zip(actual_grads, expected_grads):
            assert_precision(value, target, dtype=torch.bfloat16, backward=True)

    def test_missing_backward_and_unsupported_stream_count(self) -> None:
        """Reject unsupported kernel contracts without fallback or execution."""
        with self.assertRaisesRegex(ValueError, "four streams"):
            _load_tile_split()(torch.ones(2, 8), torch.ones(3), torch.ones(8), 2)
        with patch("hyper_parallel.components.ops.impl.mhc_pre.require_package", return_value=SimpleNamespace(
                mhc_pre_split_mixes_fwd=self.split_forward)):
            with self.assertRaisesRegex(ImplementationUnavailable, "mhc_pre_split_mixes_bwd"):
                _load_tile_split()

    def test_apply_mix_poisoned_allocator_and_repeated_backward(self) -> None:
        """Retain #1051's 64/2560-wide additive-buffer regression under the public Ops path."""
        generator = torch.Generator().manual_seed(1051)
        ops.set_backends({"mhc_pre": "tile_kernels"})
        for width in (64, 2560):
            with self.subTest(width=width):
                hidden = torch.randn(1, 7, 4, width, generator=generator, dtype=torch.bfloat16).requires_grad_()
                pre = torch.rand(1, 7, 4, generator=generator, requires_grad=True)
                upstream = torch.randn(1, 7, width, generator=generator, dtype=torch.bfloat16)
                expected_hidden = (upstream.unsqueeze(-2).float() * pre.detach().unsqueeze(-1)).to(hidden.dtype)
                expected_pre = (upstream.unsqueeze(-2).float() * hidden.detach().float()).sum(-1)
                with patch("torch.empty_like", side_effect=lambda value, **kwargs: torch.full_like(
                        value, float("nan"), **kwargs)):
                    for _ in range(2):
                        hidden.grad = pre.grad = None
                        mhc_pre_apply(hidden, pre).backward(upstream)
                        torch.testing.assert_close(hidden.grad, expected_hidden, rtol=0, atol=0)
                        torch.testing.assert_close(pre.grad, expected_pre, rtol=1e-5, atol=1e-5)

    def test_meta_replacement_preserves_state_registration_and_frozen_parameters(self) -> None:
        """Use real family replacement before to_empty; never allocate replacement parameters."""
        for factory, concrete, config in ((MHCPre, MhcPreModule, MHCPreConfig(4, 128, use_gamma=True)),
                                          (PipelinedMhc, PipelinedMhcModule, PipelinedMhcConfig(4, 128))):
            with self.subTest(factory=factory.__name__):
                with torch.device("meta"):
                    source = concrete(config=config).eval()
                first_name, frozen = next(source.named_parameters())
                frozen.requires_grad_(False)
                original = dict(source.named_parameters())
                keys = tuple(source.state_dict())
                container = torch.nn.Sequential(source)
                container, _ = prepare_model_components(container, {
                    "module_replacements": (ModuleReplacementSpec(("0",), factory),)})
                member = container[0]
                self.assertFalse(member.training)
                self.assertEqual(tuple(member.state_dict()), keys)
                for name, value in member.named_parameters():
                    self.assertIs(value, original[name])
                member.to_empty(device="cpu")
                with torch.no_grad():
                    for parameter in member.parameters():
                        parameter.fill_(float("nan"))
                member.reset_parameters()
                self.assertFalse(member.get_parameter(first_name).requires_grad)
                self.assertTrue(all(torch.isfinite(value).all() for value in member.parameters()))
                saved = {name: value.detach().clone() for name, value in member.state_dict().items()}
                restored = factory(config=config)
                parameters = dict(restored.named_parameters())
                restored.load_state_dict(saved, strict=True)
                self.assertEqual(tuple(restored.state_dict()), keys)
                for name, value in restored.named_parameters():
                    self.assertIs(value, parameters[name])
                    torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
