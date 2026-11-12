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
"""Precision policy preserves storage and the complete optimizer update chain."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.mixed_precision import GraphMixedPrecision
from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig


def _loss(model, *, x):
    """Keep loss accumulation in FP32 while linear computation is BF16."""
    return model(x).float().square().mean()


def _compiler(model, *, fp32_main_params=False, **policy):
    """Use the same YAML policy resolver as TextTrainer."""
    config = SimpleNamespace(fsdp_config=FSDP2Config(wrapper_class=["_Block"],
                                                   mix_precision=FSDP2MixedPrecisionConfig(**policy)),
                             optimizer=SimpleNamespace(fp32_main_params=fp32_main_params))
    return GraphCompiler(model, _loss, PassConfig(fsdp_enabled=False), trainer_config=config,
                         device=torch.device("cpu"))


class _Block(nn.Linear):
    """Explicitly selected FSDP child unit."""


class _Float32(nn.Module):
    """Model-owned explicit FP32 region, as used by routers and normalization."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Convert the activation before the FP32 projection."""
        return value.float()


class TestGraphMixedPrecision(TestCase):
    """The default input cast and per-parameter precision are graph operations."""

    def test_compute_gradients_do_not_round_through_storage_dtype(self):
        """Match eager's compute-parameter gradients before communication and accumulation."""
        for (storage, compute), reduce in product(
            ((torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)), (None, "float32")
        ):
            with self.subTest(storage=storage, compute=compute, reduce=reduce):
                model = nn.Sequential(_Block(2, 2, bias=False, dtype=storage))
                with torch.no_grad():
                    model[0].weight.fill_(1)
                reference = deepcopy(model).to(compute)
                x = torch.full((2, 2), 1 + 2 ** -9)
                expected_loss = _loss(reference, x=x.to(compute))
                expected_loss.backward()
                compiler = _compiler(model, fp32_main_params=True,
                                     param_dtype=str(compute).removeprefix("torch."), reduce_dtype=reduce)
                actual_loss, _ = compiler.forward_backward(x=x)
                torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                output = next(node for node in compiler._joint_graph.graph_module.graph.nodes if node.op == "output")
                gradient = output.args[0][1]
                self.assertEqual(gradient.meta["val"].dtype, torch.float32 if reduce else compute)
                self.assertEqual(model[0].weight.dtype, storage)
                self.assertIsNone(model[0].weight.grad)
                expected_dtype = torch.float32 if reduce else compute
                self.assertEqual(model[0].weight.main_grad.dtype, expected_dtype)
                torch.testing.assert_close(model[0].weight.main_grad, reference[0].weight.grad.to(expected_dtype),
                                           rtol=0, atol=0)

    def test_single_rank_explicit_reduction_dtype_is_captured(self):
        """Single-rank shortcuts cannot bypass reduction rounding or main-gradient storage."""
        for initialized, fsdp_enabled, main_grad in product((False, True), repeat=3):
            with self.subTest(initialized=initialized, fsdp_enabled=fsdp_enabled, main_grad=main_grad):
                model = nn.Sequential(_Block(2, 2, bias=False))
                with torch.no_grad():
                    model[0].weight.fill_(1)
                reference = deepcopy(model)
                x = torch.full((2, 2), 1 + 2 ** -9)
                expected_loss = _loss(reference, x=x)
                expected_loss.backward()
                expected = reference[0].weight.grad.bfloat16()
                self.assertFalse(torch.equal(expected.float(), reference[0].weight.grad))
                if not main_grad:
                    expected = expected.float()
                compiler = _compiler(model, fp32_main_params=main_grad, reduce_dtype="bfloat16")
                compiler.pass_config.fsdp_enabled = fsdp_enabled
                compiler.pass_config.fsdp_degree = 1
                distributed = MagicMock()
                distributed.is_initialized.return_value = initialized
                distributed.get_world_size.return_value = 1
                with (
                    patch("hyper_parallel.compile.compiler.dist", distributed),
                    patch("hyper_parallel.compile.passes.parallel.fsdp_pass.dist", distributed),
                    patch.object(compiler, "_init_device_mesh"),
                ):
                    loss, _ = compiler.forward_backward(x=x)
                torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                gradient = model[0].weight.main_grad if main_grad else model[0].weight.grad
                torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
                graph = compiler._joint_graph.graph_module
                output = next(node for node in graph.graph.nodes if node.op == "output")
                self.assertEqual(output.args[0][1].meta["val"].dtype, torch.bfloat16)
                self.assertFalse(any("_c10d_functional" in str(node.target) for node in graph.graph.nodes))

    def test_reduction_cast_preserves_trainable_state_and_named_losses(self):
        """Frozen state and buffers cannot shift gradient dtype assignments or include named losses."""
        model = nn.Sequential(_Block(2, 2))
        model[0].bias.requires_grad_(False)
        model[0].register_parameter("unused", nn.Parameter(torch.ones(2, 2)))
        model[0].register_buffer("offset", torch.ones(2))
        model[0].alias_weight = model[0].weight
        reference = deepcopy(model)
        x = torch.full((2, 2), 1 + 2 ** -9)
        expected_loss = _loss(reference, x=x)
        expected_loss.backward()
        compiler = _compiler(model, reduce_dtype="bfloat16")

        def named_loss(current, **inputs):
            """Expose a forward loss separately from trainable gradient outputs."""
            value = _loss(current, **inputs)
            return value, {"aux": value}

        compiler.train_fn = named_loss
        loss, losses = compiler.forward_backward(x=x)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(losses["aux"], expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(model[0].weight.grad, reference[0].weight.grad.bfloat16().float(), rtol=0, atol=0)
        self.assertIsNone(model[0].unused.grad)
        self.assertIsNone(model[0].bias.grad)
        self.assertIs(model[0].alias_weight, model[0].weight)
        previous = torch.ones_like(model[0].unused)
        model[0].unused.grad = previous
        compiler.forward_backward(x=x)
        self.assertIs(model[0].unused.grad, previous)
        torch.testing.assert_close(previous, torch.ones_like(previous), rtol=0, atol=0)

    def test_parameter_overrides_preserve_implicit_reduction_dtype(self):
        """Missing fields inherit defaults while None and redundant casts preserve gradient dtype."""
        cases = (
            (None, torch.bfloat16, torch.float32),
            ({"param_dtype": None}, torch.float32, torch.float32),
            ({"reduce_dtype": None}, torch.bfloat16, None),
            ({"param_dtype": None, "reduce_dtype": None}, torch.float32, None),
            ({"param_dtype": torch.float32}, torch.float32, None),
            ({"reduce_dtype": torch.bfloat16}, torch.bfloat16, None),
        )
        for custom, compute, reduce in cases:
            with self.subTest(custom=custom):
                model = nn.Linear(2, 2)
                policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    custom_params={model.weight: custom} if custom is not None else None,
                )
                precision = GraphMixedPrecision(model, policy, ())
                self.assertEqual(precision.parameter_dtypes["weight"], (compute, reduce))
                self.assertEqual(precision.parameter_dtypes["bias"], (torch.bfloat16, torch.float32))
                cast = precision.cast_state(dict(model.named_parameters()))
                self.assertEqual(cast["weight"].dtype, compute)

    def test_fp32_storage_bf16_compute_matches_ten_adamw_updates(self):
        """Compare losses, gradients, weights and moments against explicit eager casts."""
        torch.manual_seed(17)
        model = nn.Sequential(_Block(4, 4))
        reference = deepcopy(model)
        compiler = _compiler(model, param_dtype="bfloat16", reduce_dtype="float32")
        compiler.prepare_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
        for step in range(10):
            x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 16 + step / 64
            computation = deepcopy(reference).bfloat16()
            expected = computation(x.bfloat16())
            expected_loss = expected.float().square().mean()
            expected_loss.backward()
            for master, parameter in zip(reference.parameters(), computation.parameters()):
                master.grad = parameter.grad.float()
            loss, _ = compiler.forward_backward(x=x)
            torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
            for actual, expected_parameter in zip(model.parameters(), reference.parameters()):
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual.grad, expected_parameter.grad, rtol=0, atol=0)
            optimizer.step()
            expected_optimizer.step()
            for actual, expected_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected_parameter, rtol=0, atol=0)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(optimizer.state[actual][key],
                                               expected_optimizer.state[expected_parameter][key], rtol=0, atol=0)
            optimizer.zero_grad()
            expected_optimizer.zero_grad()
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model._forward_hooks)
        self.assertTrue(any(node.target == torch.ops.aten._to_copy.default
                            for node in compiler._joint_graph.graph_module.graph.nodes))

    def test_custom_parameter_dtype_and_module_output_cast(self):
        """Pattern overrides preserve an explicit FP32 compute region."""
        model = nn.Sequential(_Block(4, 4), _Float32(), nn.Linear(4, 4))
        compiler = _compiler(model, param_dtype="bfloat16", reduce_dtype="float32", output_dtype="float32",
                             custom_params={"2.*": {"param_dtype": "float32"}})
        compiler.prepare_model()
        precision = compiler._mixed_precision
        self.assertEqual(precision.parameter_dtypes["0.weight"], (torch.bfloat16, torch.float32))
        self.assertEqual(precision.parameter_dtypes["2.weight"], (torch.float32, None))
        state = dict(model.named_parameters())
        cast = precision.cast_state(state)
        self.assertEqual(cast["0.weight"].dtype, torch.bfloat16)
        self.assertIs(cast["2.weight"], model[2].weight)
        with precision.module_casts():
            # Module output casts use the same floating-leaf helper as eager.
            result = torch.func.functional_call(model, cast, (torch.ones(2, 4),))
        self.assertEqual(result.dtype, torch.float32)

    def test_fp32_main_gradient_keeps_reduction_precision(self):
        """A BF16 model can accumulate FP32 gradients without assigning them to .grad."""
        model = nn.Sequential(_Block(4, 4, dtype=torch.bfloat16))
        compiler = _compiler(model, fp32_main_params=True, param_dtype="bfloat16", reduce_dtype="float32")
        compiler.prepare_model()
        compiler.forward_backward(x=torch.ones(2, 4))
        first = [parameter.main_grad.clone() for parameter in model.parameters()]
        compiler.forward_backward(x=torch.ones(2, 4))
        for parameter, gradient in zip(model.parameters(), first):
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.main_grad.dtype, torch.float32)
            torch.testing.assert_close(parameter.main_grad, gradient * 2, rtol=0, atol=0)

    def test_microbatch_accumulation_preserves_eager_rounding(self):
        """Reduction precision and per-microbatch storage casts precede accumulation."""
        cases = (
            (torch.float32, "bfloat16", True, (1.0, 2 ** -8), 1.0, torch.bfloat16),
            (torch.bfloat16, "float32", False, (1.0, -0.998046875), 0.0, torch.bfloat16),
        )
        for storage, reduction, main, inputs, expected, dtype in cases:
            with self.subTest(storage=storage, reduction=reduction, main=main):
                model = nn.Sequential(_Block(1, 1, bias=False, dtype=storage))
                compiler = _compiler(model, fp32_main_params=main, param_dtype="float32",
                                     reduce_dtype=reduction)

                def summed_loss(current, *, x):
                    """Make the weight gradient equal the microbatch input."""
                    return current(x).sum()

                compiler.train_fn = summed_loss
                for value in inputs:
                    compiler.forward_backward(x=torch.tensor([[value]]))
                gradient = model[0].weight.main_grad if main else model[0].weight.grad
                self.assertEqual(gradient.dtype, dtype)
                torch.testing.assert_close(gradient, torch.tensor([[expected]], dtype=dtype), rtol=0, atol=0)
                if main:
                    gradient.div_(3)
                    torch.testing.assert_close(gradient, torch.tensor([[1 / 3]], dtype=dtype), rtol=0, atol=0)
