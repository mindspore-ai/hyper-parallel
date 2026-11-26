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

"""Native moe_router forward and gradient precision contracts."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import checkpoint

from examples.torch.module_family.config import GateConfig
from hyper_parallel.components.modules import Gate
from hyper_parallel.components.modules.moe.router import GateModule
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec
from hyper_parallel.components.ops import moe_topk_gate, ops
from hyper_parallel.components.ops.impl.moe_router import MOE_ROUTER, _load_tile_kernels, register_moe_router
from hyper_parallel.components.ops.impl.moe_router import (
    MOE_GATE, MOE_GATE_BACKWARD, ROUTING_NORMALIZE, _load_gate, _load_gate_backward, _load_normalize,
    _native_gate,
)
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases
from tests.common.op_precision_cases import _gate_backward_reference, _gate_reference, _normalize_reference


class TestMoeRouterPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("moe_router", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_tile_kernels_cpu_abi(self) -> None:
        """Check Top-K argument forwarding while keeping the projection and weight gradients."""
        calls = []

        def topk_gate(scores: Any, top_k: Any) -> Any:
            """Topk gate."""
            self.assertEqual(scores.ndim, 2)
            self.assertEqual(scores.dtype, torch.float32)
            self.assertTrue(scores.is_contiguous())
            calls.append(top_k)
            return scores.argsort(dim=-1, descending=True, stable=True)[:, :top_k]

        registry = OpRegistry()
        register_moe_router(registry)
        registry.register_impl(MOE_ROUTER, name="tile_kernels", devices=("cpu",), load=_load_tile_kernels)
        registry.set_auto_order(MOE_ROUTER, device="cpu", candidates=("native",))
        registry.seal()
        self.addCleanup(ops.set_backends, ops.get_backends())
        with patch("hyper_parallel.components.ops.impl.moe_router.require_package",
                   return_value=SimpleNamespace(topk_gate=topk_gate)), \
                patch("hyper_parallel.components.ops.factory.builtin_registry", registry):
            for dtype in (torch.float32, torch.bfloat16):
                for case in precision_cases("moe_router", dtype, backend="tile_kernels"):
                    if case.variant != "deepseek_v41":
                        continue
                    with self.subTest(dtype=dtype, case=case.name):
                        check_case(case)
                        check_case(case, backend="tile_kernels")
        self.assertEqual(calls, [1, 8, 8, 8] * 2)

    def test_tile_kernels_requires_topk_symbol(self) -> None:
        """A missing Top-K export must not silently fall back to native."""
        with patch("hyper_parallel.components.ops.impl.moe_router.require_package", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ImplementationUnavailable, "missing operator topk_gate"):
                _load_tile_kernels()


class TestTileKernelsRouterStages(unittest.TestCase):
    """CPU simulation for raw fused gate/backward and normalization, not device acceptance."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        self.calls = []

        def forward(
            logits: Any,
            top_k: Any,
            shared: Any,
            shared_count: Any,
            scale: Any,
            rank: Any,
            *,
            scores: Any=None,
            **options: Any,
        ) -> Any:
            """Execute the selected backend and retain the tensors needed for backward."""
            self.assertEqual(logits.dtype, torch.float32)
            self.calls.append("forward")
            if scores is not None:
                scores.copy_(torch.logaddexp(logits, torch.zeros_like(logits)).sqrt())
            if shared or any(options.get(name) is not None for name in ("to_physical_map", "fix_routing_mask")):
                return _native_gate(logits, top_k, shared, shared_count, scale, rank, **options)
            return _gate_reference(logits, top_k, shared, shared_count, scale, rank, **options)

        def backward(*args: Any, **options: Any) -> Any:
            """Return the input gradients required by the operator contract."""
            self.calls.append("backward")
            return _gate_backward_reference(*args, **options)

        def normalize(weights: Any) -> Any:
            """Normalize."""
            self.calls.append("normalize")
            return _normalize_reference(weights)

        self.kernels = SimpleNamespace(moe_topk_gate=forward, moe_topk_gate_backward=backward,
                                       normalize_weight=normalize)
        dependency = patch("hyper_parallel.components.ops.impl.moe_router.require_package", return_value=self.kernels)
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_moe_router(registry)
        for spec, loader in ((MOE_GATE, _load_gate), (MOE_GATE_BACKWARD, _load_gate_backward),
                             (ROUTING_NORMALIZE, _load_normalize)):
            registry.register_impl(spec, name="tile_kernels", devices=("cpu",), load=loader)
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_outputs_and_gradients_including_score_sum_auxiliary_gradient(self) -> None:
        for case in precision_cases("moe_router", torch.float32, backend="tile_kernels"):
            if case.variant == "deepseek_v41":
                continue
            with self.subTest(variant=case.variant, name=case.name):
                check_case(case)
                check_case(case, backend="tile_kernels")
        for name in ("forward", "backward", "normalize"):
            self.assertIn(name, self.calls)

    def test_training_rejects_preallocated_outputs(self) -> None:
        logits = torch.randn(4, 128, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "preallocated output buffers"):
            _load_gate()(logits, 8, out=(None, None))
        self.assertEqual(self.calls, [])

    def test_mapping_and_shared_experts_use_logical_routed_gradients(self) -> None:
        generator = torch.Generator().manual_seed(17)
        source = torch.randn(4, 128, generator=generator)
        upstream = torch.randn(4, 10, generator=generator)
        options = {"to_physical_map": torch.arange(130).roll(3).reshape(130, 1).int(),
                   "logical_count": torch.ones(130, dtype=torch.int32)}
        results = []
        for backend in ("native", "tile_kernels"):
            logits = source.clone().requires_grad_()
            ops.set_backends({"moe_router": backend})
            indices, weights = moe_topk_gate(logits, 8, True, 2, 1.7, 3, **options)
            order = indices.argsort(-1)
            indices, weights = indices.gather(-1, order), weights.gather(-1, order)
            results.append((indices, weights, torch.autograd.grad(weights, logits, upstream)[0]))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)

    def test_random_top1_weights_have_negligible_logit_gradient(self) -> None:
        """Use the FP32 absolute budget when relative error is undefined near a constant output."""
        generator = torch.Generator().manual_seed(971)
        source = torch.randn(16, 128, generator=generator)
        upstream = torch.randn(16, 1, generator=generator)
        for backend in ("native", "tile_kernels"):
            with self.subTest(backend=backend):
                logits = source.clone().requires_grad_()
                ops.set_backends({"moe_router": backend})
                indices, weights = moe_topk_gate(logits, 1, routed_scaling_factor=1.7)
                torch.testing.assert_close(indices, source.argmax(-1, keepdim=True), rtol=0, atol=0)
                torch.testing.assert_close(weights, torch.full_like(weights, 1.7), atol=1e-5, rtol=1e-4)
                gradient = torch.autograd.grad(weights, logits, upstream)[0]
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-4, rtol=0)

    def test_randomized_routes_do_not_backpropagate_into_logits(self) -> None:
        """Upstream marks randomized rows with -1 logical indices; their routing VJP must be zero."""
        def forward(
            logits: Any, top_k: Any, shared: Any, shared_count: Any,
            scale: Any, rank: Any, **options: Any,
        ) -> Any:
            """Execute the selected backend and retain the tensors needed for backward."""
            randomized = options.pop("force_random")
            indices, weights = _native_gate(logits, top_k, shared, shared_count, scale, rank, **options)
            options["unmapped_topk_idx"].masked_fill_(randomized[:, None], -1)
            return indices, weights.masked_fill(randomized[:, None], 0.25)

        kernels = SimpleNamespace(moe_topk_gate=forward, moe_topk_gate_backward=self.kernels.moe_topk_gate_backward)
        logits = torch.randn(4, 128, requires_grad=True)
        flags = torch.tensor([False, True, False, True])
        with patch("hyper_parallel.components.ops.impl.moe_router.require_package", return_value=kernels):
            _, weights = _load_gate()(logits, 8, force_random=flags)
            gradient = torch.autograd.grad(weights, logits, torch.randn_like(weights))[0]
        torch.testing.assert_close(gradient[flags], torch.zeros_like(gradient[flags]), rtol=0, atol=0)
        self.assertGreater(gradient[~flags].abs().sum().item(), 0)

    def test_inference_forwards_original_optional_arguments(self) -> None:
        forward = Mock(return_value=(torch.ones(2, 10, dtype=torch.int64), torch.ones(2, 10)))
        kernels = SimpleNamespace(moe_topk_gate=forward, moe_topk_gate_backward=self.kernels.moe_topk_gate_backward)
        options = {"to_physical_map": torch.ones(130, 2, dtype=torch.int32),
                   "logical_count": torch.ones(130, dtype=torch.int32),
                   "force_random": torch.tensor([False, True])}
        logits = torch.randn(2, 128)
        with patch("hyper_parallel.components.ops.impl.moe_router.require_package", return_value=kernels):
            _load_gate()(logits, 8, True, 2, 1.7, 3, **options)
        forward.assert_called_once_with(logits, 8, True, 2, 1.7, 3, **options)

    def test_unmigrated_gate_preserves_eager_checkpoint_and_gradients(self) -> None:
        """Ops selection does not migrate the existing Gate projection or checkpoint path."""
        config = GateConfig(16, 128, 8, scoring_func="sqrtsoftplus")
        member, reference = (Gate(config=config) for _ in range(2))
        reference.load_state_dict(member.state_dict())
        hidden = torch.randn(2, 4, 16, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        ops.set_backends({"moe_router": "native"})
        expected = reference(other)
        ops.set_backends({"moe_router": "tile_kernels"})
        actual = checkpoint(member, hidden, use_reentrant=False)

        def canonical(result: Any) -> Any:
            """Canonical."""
            logits, weights, indices = result
            order = indices.argsort(-1)
            return logits, weights.gather(-1, order), indices.gather(-1, order)

        actual, expected = canonical(actual), canonical(expected)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target)
        upstream = tuple(torch.randn_like(value) for value in expected[:2])
        torch.autograd.backward(actual[:2], upstream)
        torch.autograd.backward(expected[:2], upstream)
        torch.testing.assert_close(hidden.grad, other.grad, atol=1e-4, rtol=1e-3)
        for left, right in zip(member.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-4, rtol=1e-3)
        self.assertEqual(self.calls, [])
        for target in (member, reference):
            torch.optim.SGD(target.parameters(), lr=.001, foreach=False).step()
        saved = {name: value.detach().clone() for name, value in member.state_dict().items()}
        restored = Gate(config=config)
        restored.load_state_dict(saved, strict=True)
        ops.set_backends({"moe_router": "native"})
        expected = canonical(reference(hidden.detach()))
        ops.set_backends({"moe_router": "tile_kernels"})
        for target in (member, restored):
            for value, golden in zip(canonical(target(hidden.detach())), expected):
                torch.testing.assert_close(value, golden, atol=1e-5, rtol=1e-4)

    def test_meta_adoption_rebuilds_counts_and_preserves_checkpoint_bias(self) -> None:
        """Restore saved balancing bias without serializing or retaining stale expert counters."""
        config = GateConfig(16, 128, 8, scoring_func="sqrtsoftplus")
        context = {"module_configs": (SimpleNamespace(moe_router_bias_update_rate=.01),)}
        with torch.device("meta"):
            source = GateModule(config=config, context=context).eval()
        source.weight.requires_grad_(False)
        original = dict(source.named_parameters())
        keys = tuple(source.state_dict())
        container = torch.nn.Sequential(source)
        rng = torch.get_rng_state().clone()
        container, _ = prepare_model_components(container, {
            "module_replacements": (ModuleReplacementSpec(("0",), Gate),)})
        member = container[0]
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in member.named_parameters():
            self.assertIs(value, original[name])
        self.assertFalse(member.training)
        self.assertEqual(tuple(member.state_dict()), keys)
        member.to_empty(device="cpu")
        with torch.no_grad():
            for parameter in member.parameters():
                parameter.fill_(float("nan"))
        member.reset_parameters()
        rebuild_materialized_state(member, MaterializationContext("random_init", torch.device("cpu")))
        self.assertFalse(member.weight.requires_grad)
        self.assertTrue(torch.isfinite(member.weight).all())
        torch.testing.assert_close(member.bias, torch.zeros_like(member.bias), rtol=0, atol=0)
        torch.testing.assert_close(member.tokens_per_expert, torch.zeros_like(member.tokens_per_expert), rtol=0, atol=0)
        member.bias.data.copy_(torch.linspace(-.3, .3, 128))
        member.tokens_per_expert.fill_(91)
        saved = {name: value.detach().clone() for name, value in member.state_dict().items()}
        self.assertNotIn("tokens_per_expert", saved)
        restored = Gate(config=config, context=context).eval()
        parameters = dict(restored.named_parameters())
        restored.load_state_dict(saved, strict=True)
        restored.tokens_per_expert.fill_(17)
        rebuild_materialized_state(restored, MaterializationContext("checkpoint_load", torch.device("cpu")))
        torch.testing.assert_close(restored.tokens_per_expert, torch.zeros_like(restored.tokens_per_expert),
                                   rtol=0, atol=0)
        for name, value in restored.named_parameters():
            self.assertIs(value, parameters[name])
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        hidden = torch.randn(2, 4, 16)
        ops.set_backends({"moe_router": "native"})
        expected = restored(hidden)
        ops.set_backends({"moe_router": "tile_kernels"})
        actual = member(hidden)
        # Native Top-K may return a different order for the same selected experts.
        actual_order, expected_order = actual[2].argsort(-1), expected[2].argsort(-1)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual[1].gather(-1, actual_order), expected[1].gather(-1, expected_order),
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual[2].gather(-1, actual_order), expected[2].gather(-1, expected_order),
                                   rtol=0, atol=0)
