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
"""Engram reference precision and CPU-only optional-package ABI simulation."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from examples.torch.module_family.config import EngramConfig
from hyper_parallel.components.modules.engram.engram_family import Engram
from hyper_parallel.components.modules.engram import EngramModule, NgramHashMapping
from hyper_parallel.models.materialization import MaterializationContext, rebuild_materialized_state
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec
from hyper_parallel.components.ops import engram_gate, ops
from hyper_parallel.components.ops.impl.engram import ENGRAM_GATE, ENGRAM_HASH, _load_gate, _load_hash, register_engram
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_cases import _engram_gate, _engram_hash
from tests.common.op_precision_suite import precision_cases


class TestEngramPrecision(unittest.TestCase):
    """Native hashes and all gate gradients have independently expressed references."""

    def test_native_forward_and_gradients(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            for case in precision_cases("engram", dtype):
                with self.subTest(dtype=dtype, case=case.name):
                    check_case(case)


    def test_native_preserves_original_eager_gate_order(self) -> None:
        """Preserve the original module's FP32 evaluation order, including its gradients."""
        previous = ops.get_backends()
        self.addCleanup(ops.set_backends, previous)
        ops.set_backends({"engram": "native"})
        hidden = torch.randn(2, 4, 16, requires_grad=True)
        kv = torch.randn(2, 80, requires_grad=True)
        wh = torch.randn(4, 16, requires_grad=True)
        we = torch.randn(4, 16, requires_grad=True)
        key, value = kv.split((64, 16), dim=-1)
        key = key.unflatten(-1, (4, 16))
        weight = wh * we
        reciprocal_std = torch.rsqrt(hidden.square().mean(-1) + 1e-6)
        reciprocal_std = reciprocal_std * torch.rsqrt(key.square().mean(-1) + 1e-6)
        dot = (hidden * weight * key).sum(-1) * reciprocal_std * 16**-0.5
        root = dot.abs().clamp_min(1e-4).sqrt()
        gate = torch.where(dot < 0, -root, root).sigmoid()
        expected = hidden + gate.unsqueeze(-1) * value.unsqueeze(-2)
        actual = engram_gate(hidden, kv, wh, we, 1e-4, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (hidden, kv, wh, we), upstream)
        expected_grads = torch.autograd.grad(expected, (hidden, kv, wh, we), upstream)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


class TestTileKernelsEngram(unittest.TestCase):
    """Simulate the upstream autograd-enabled function; this is not kernel validation."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        self.calls = []

        def hash_kernel(*args: Any) -> Any:
            """Hash kernel."""
            self.assertEqual(args[0].dtype, torch.int32)
            self.assertEqual(args[1].dtype, torch.int64)
            self.calls.append("hash")
            return _engram_hash(*args)

        def gate_kernel(*args: Any) -> Any:
            """Gate kernel."""
            self.calls.append("gate")
            return _engram_gate(*args)

        self.kernels = SimpleNamespace(engram_hash=hash_kernel, engram_gate=gate_kernel)
        dependency = patch("hyper_parallel.components.ops.impl.engram.require_package", return_value=self.kernels)
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_engram(registry)
        for spec, loader in ((ENGRAM_HASH, _load_hash), (ENGRAM_GATE, _load_gate)):
            registry.register_impl(spec, name="tile_kernels", devices=("cpu",), load=loader)
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_shared_forward_and_gradient_cases(self) -> None:
        for case in precision_cases("engram", torch.bfloat16, backend="tile_kernels"):
            with self.subTest(variant=case.variant, case=case.name):
                check_case(case)
                check_case(case, backend="tile_kernels")
        self.assertIn("hash", self.calls)
        self.assertIn("gate", self.calls)

    def test_reuses_upstream_autograd_without_another_wrapper(self) -> None:
        self.assertIs(_load_gate(), self.kernels.engram_gate)

    def test_missing_symbol_fails(self) -> None:
        with patch("hyper_parallel.components.ops.impl.engram.require_package", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ImplementationUnavailable, "engram_gate"):
                _load_gate()

    def test_unmigrated_module_preserves_eager_checkpoint_and_gradients(self) -> None:
        assets = {"layer_ids": [0], "max_ngram_size": 3, "num_heads": 2, "token_map": list(range(16)),
                  "pad_token_id": 0, "primes": [[[5, 7], [11, 13]]], "multipliers": [[1, 3, 5]]}
        config = EngramConfig(assets, 0, 128, 4, 8)
        module, reference = (Engram(config=config).to(torch.bfloat16) for _ in range(2))
        reference.load_state_dict(module.state_dict())
        ids = torch.tensor([[1, 2, 3, 4]])
        starts, mask = torch.tensor([[True, False, True, False]]), torch.tensor([[True, False, True, True]])
        hidden = torch.randn(1, 4, 4, 128, dtype=torch.bfloat16, requires_grad=True)
        other = hidden.detach().clone().requires_grad_()
        ops.set_backends({"engram": "native"})
        expected = reference(other, ids, starts, mask)
        self.assertEqual(self.calls, [])
        ops.set_backends({"engram": "tile_kernels"})
        actual = checkpoint(module, hidden, ids, starts, mask, use_reentrant=False)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(hidden.grad, other.grad, atol=2e-2, rtol=2e-2)
        for left, right in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=2e-2, rtol=2e-2)
        self.assertEqual(self.calls, [])
        for member in (module, reference):
            torch.optim.SGD(member.parameters(), lr=.001, foreach=False).step()
        saved = {name: value.detach().clone() for name, value in module.state_dict().items()}
        restored = Engram(config=config).to(torch.bfloat16)
        restored.load_state_dict(saved, strict=True)
        ops.set_backends({"engram": "native"})
        expected = reference(hidden.detach(), ids, starts, mask)
        ops.set_backends({"engram": "tile_kernels"})
        for member in (module, restored):
            torch.testing.assert_close(member(hidden.detach(), ids, starts, mask), expected, atol=1e-2, rtol=1e-2)

    def test_meta_replacement_hash_rebuild_and_checkpoint_restore(self) -> None:
        """Preserve borrowed parameters and rebuild only nonpersistent hashes after materialization."""
        assets = {"layer_ids": [0], "max_ngram_size": 3, "num_heads": 2, "token_map": list(range(16)),
                  "pad_token_id": 0, "primes": [[[5, 7], [11, 13]]], "multipliers": [[1, 3, 5]]}
        config = EngramConfig(assets, 0, 128, 4, 8)
        with torch.device("meta"):
            source = EngramModule(config=config).eval()
        source.embed.weight.requires_grad_(False)
        original = dict(source.named_parameters())
        keys = tuple(source.state_dict())
        container = torch.nn.Sequential(source)
        rng = torch.get_rng_state().clone()
        container, _ = prepare_model_components(container, {
            "module_replacements": (ModuleReplacementSpec(("0",), Engram),)})
        member = container[0]
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in member.named_parameters():
            self.assertIs(value, original[name])
        self.assertFalse(member.training)
        self.assertEqual(tuple(member.state_dict()), keys)
        self.assertEqual(self.calls, [])
        member.to_empty(device="cpu")
        with torch.no_grad():
            for parameter in member.parameters():
                parameter.fill_(float("nan"))
        for child in member.modules():
            if callable(getattr(child, "reset_parameters", None)):
                child.reset_parameters()
        rebuild_materialized_state(member, MaterializationContext("random_init", torch.device("cpu")))
        expected_hash = NgramHashMapping(assets, 0)
        for name, value in expected_hash.named_buffers():
            torch.testing.assert_close(member.hash_mapping.get_buffer(name), value, rtol=0, atol=0)
        self.assertFalse(member.embed.weight.requires_grad)
        self.assertFalse(any("hash_mapping" in key for key in member.state_dict()))
        torch.testing.assert_close(member.q_weight, torch.ones_like(member.q_weight), rtol=0, atol=0)
        torch.testing.assert_close(member.k_weight, torch.ones_like(member.k_weight), rtol=0, atol=0)
        member.to(torch.bfloat16)
        saved = {name: value.detach().clone() for name, value in member.state_dict().items()}
        restored = Engram(config=config).to(torch.bfloat16)
        parameters = dict(restored.named_parameters())
        restored.load_state_dict(saved, strict=True)
        for name, value in restored.named_parameters():
            self.assertIs(value, parameters[name])
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        ids = torch.tensor([[1, 2, 3]])
        hidden = torch.randn(1, 3, 4, 128, dtype=torch.bfloat16)
        ops.set_backends({"engram": "native"})
        expected = restored(hidden, ids)
        ops.set_backends({"engram": "tile_kernels"})
        torch.testing.assert_close(member(hidden, ids), expected, rtol=0, atol=0)
        self.assertEqual(tuple(member.state_dict()), keys)
        self.assertEqual(self.calls, [])
