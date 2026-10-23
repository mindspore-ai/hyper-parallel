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

"""Hardware-independent checks for the three-kernel example and benchmark wiring."""

from __future__ import annotations

import importlib
import io
import sys
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hyper_parallel import DeviceMesh, DTensor, Shard, SkipDTensorDispatch
from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.multicore import MegaMoeExperts, MhcState
from hyper_parallel.core.multicore.examples import benchmark
from hyper_parallel.core.multicore.examples import mhc as mhc_module
from hyper_parallel.core.multicore.examples import moe as moe_module
from hyper_parallel.core.multicore.examples.mhc import MhcBoundary
from hyper_parallel.core.multicore.examples.model import (
    MoeTransformerConfig,
    MoeTransformerModel,
)
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.common.mark_utils import arg_mark
from tests.common.multicore.mega_mhc_golden import torch_mega_mhc

rms_norm_module = importlib.import_module("hyper_parallel.components.functional.rms_norm")


def _experts_reference(
    module: MegaMoeExperts,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate all local experts explicitly; EP is one in these CPU checks."""
    ids, weights = topk_ids, topk_weights
    shape = hidden.shape
    hidden = hidden.reshape(-1, shape[-1])
    if tokens_per_expert is not None:
        expected_counts = torch.bincount(ids.flatten(), minlength=module.num_experts).to(torch.int32)
        torch.testing.assert_close(tokens_per_expert, expected_counts, rtol=0, atol=0)
    gate_up = torch.einsum("th,ehj->tej", hidden, module.gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    output = torch.einsum("tei,eid->ted", F.silu(gate) * up, module.down_weight)
    scores = torch.zeros(hidden.shape[0], module.num_experts).scatter_add(1, ids, weights)
    return (output.float() * scores.unsqueeze(-1)).sum(dim=1).to(hidden.dtype).reshape(shape)


def _rms_reference(hidden: torch.Tensor, weight: torch.Tensor, *, epsilon: float) -> tuple:
    """Replace only the NPU RMSNorm dispatch with its CPU mathematical equivalent."""
    normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(dim=-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(hidden.dtype), None


def _native_mhc_reference(*args: torch.Tensor, token_tile: int, **kwargs: float) -> tuple:
    """Bridge native dispatch to the independent oracle for the CPU wiring check."""
    del token_tile
    return torch_mega_mhc(*args, **kwargs)


def _rotary_reference(
    hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, rotary_mode: str,
) -> torch.Tensor:
    """Replace only the NPU half-rotation dispatch for the CPU graph check."""
    if rotary_mode != "half":
        raise ValueError(f"Unexpected rotary mode: {rotary_mode}.")
    first, second = hidden.chunk(2, dim=-1)
    return hidden * cos + torch.cat((-second, first), dim=-1) * sin


class TestMoeTransformerExample(unittest.TestCase):
    """Exercise identical architectures, selective dispatch, and frozen router bias."""

    @staticmethod
    def _config() -> MoeTransformerConfig:
        """Keep native shape contracts but avoid accelerator allocations."""
        return MoeTransformerConfig(
            vocab_size=32, hidden_size=128, num_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_seq_len=128, num_experts=4, top_k=2,
            intermediate_size=128, shared_expert_intermediate_size=128,
            local_num_tokens=128, ep_size=1,
        )

    @staticmethod
    def _model(kernels: tuple[str, ...]) -> MoeTransformerModel:
        """Initialize equal BF16 weights while keeping mHC predictor parameters FP32."""
        torch.manual_seed(30)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            return MoeTransformerModel(TestMoeTransformerExample._config(), mega_kernels=kernels)
        finally:
            torch.set_default_dtype(original_dtype)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_router_scores_and_gradients(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Router scores and gradients.
        Expectation: Match sqrt-softplus TopK, unbiased selected scores, and projection gradients.
        """
        model = self._model(())
        self.addCleanup(model.close)
        gate = model.layers[0].mlp.gate
        with torch.no_grad():
            gate.bias.copy_(torch.linspace(-0.1, 0.1, gate.num_experts))
        hidden = torch.randn(1, 128, 128, dtype=torch.bfloat16, requires_grad=True)
        _, weights, ids = gate.native_forward(hidden)
        _, expected_weights, expected_ids = gate.mega_forward(hidden)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
        probe = torch.randn_like(weights)
        actual_grads = torch.autograd.grad((weights * probe).sum(), (hidden, gate.weight))
        expected_grads = torch.autograd.grad((expected_weights * probe).sum(), (hidden, gate.weight))
        for got, want in zip(actual_grads, expected_grads):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_two_layer_training_and_native_boundary_wiring(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Two layer training and native boundary wiring.
        Expectation: Keep cross-layer mHC state and gradients equal across execution choices.
        """
        reference = self._model(())
        optimized = self._model(("moe", "gate", "mhc"))
        self.addCleanup(reference.close)
        self.addCleanup(optimized.close)
        inputs = torch.arange(128).reshape(1, -1) % 32
        with (
            patch.object(MegaMoeExperts, "forward", _experts_reference),
            patch.object(rms_norm_module.torch_npu, "npu_rms_norm", _rms_reference),
            patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul", _rotary_reference),
            patch("hyper_parallel.core.multicore.modules.mega_mhc.module.hyper_mega_mhc",
                  side_effect=_native_mhc_reference) as native_mhc,
        ):
            results = [model(inputs, labels=inputs) for model in (reference, optimized)]
            for result in results:
                result["loss"].backward()
        self.assertEqual(native_mhc.call_count, 4)
        torch.testing.assert_close(results[0]["logits"], results[1]["logits"], rtol=0, atol=0)
        torch.testing.assert_close(results[0]["loss"], results[1]["loss"], rtol=0, atol=0)
        for model in (reference, optimized):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, msg=f"Missing gradient for {name}")
                    self.assertTrue(torch.isfinite(parameter.grad).all(), msg=f"Nonfinite gradient for {name}")
            tensors = benchmark._canonical_tensors(model, gradients=True)  # pylint: disable=protected-access
            self.assertNotIn("dense.layers.0.mlp.gate.bias", tensors)
        for got, want in zip(reference.parameters(), optimized.parameters()):
            if got.requires_grad:
                torch.testing.assert_close(got.grad, want.grad, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_selective_backend_construction(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Selective backend construction.
        Expectation: Use common EP unless moe is explicitly selected in the optimized backend.
        """
        for backend, kernels in (("common", ("moe", "gate", "mhc")), ("mega", ("gate",)), ("mega", ("moe",))):
            with (
                self.subTest(backend=backend, kernels=kernels),
                patch.object(benchmark, "_reset_seed"),
                patch.object(benchmark, "_build_model", side_effect=lambda c, d, k: self._model(k)) as build,
                patch.object(benchmark, "init_device_mesh"),
                patch.object(
                    moe_module, "NativeExperts", side_effect=lambda *args: torch.nn.Identity(),
                ) as common,
            ):
                model = benchmark._build_backend_model(  # pylint: disable=protected-access
                    self._config(), torch.device("cpu"), 30, backend, kernels,
                )
                selected = kernels if backend == "mega" else ()
                self.assertEqual(build.call_args.args[2], selected)
                self.assertEqual(common.call_count, 0 if "moe" in selected else 2)
                self.assertEqual(model.layers[0].mlp.gate.use_mega_gate, "gate" in selected)
                self.assertEqual(model.layers[0].attention_mhc.use_mega_mhc, "mhc" in selected)
                for layer in model.layers:
                    if isinstance(layer.mlp.experts, MegaMoeExperts):
                        layer.mlp.experts.close()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mixed_dtype_gradient_synchronization(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Mixed dtype gradient synchronization.
        Expectation: Keep FP32 mHC predictors and BF16 dense weights in separate collectives.
        """
        gradients = [torch.ones(3, dtype=torch.bfloat16), torch.ones(2), torch.full((5,), 2.0)]
        expected = [gradient.clone() for gradient in gradients]

        def sum_gradients(bucket: list[torch.Tensor], *, op: object) -> None:
            """Emulate the dense gradient reduction without a process group."""
            self.assertEqual(op, benchmark.dist.ReduceOp.SUM)
            self.assertEqual(len({gradient.dtype for gradient in bucket}), 1)
            for gradient in bucket:
                gradient.mul_(8)

        with (
            patch.object(benchmark, "_dense_gradients", return_value=gradients),
            patch.object(benchmark.dist, "all_reduce_coalesced", side_effect=sum_gradients) as all_reduce,
        ):
            benchmark._synchronize_dense_gradients(None, 8)  # pylint: disable=protected-access
        self.assertEqual(all_reduce.call_count, 2)
        for got, want in zip(gradients, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_main_gradient_layout_and_optimizer_update(self, mock_rank: object) -> None:
        """Feature: MegaKernel model integration.
        Description: Ep main gradient layout and optimizer update.
        Expectation: Bridge actual local autograd gradients to a DTensor main-parameter update.
        """
        del mock_rank
        mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("ep",), _init_backend=False)
        parameter = torch.nn.Parameter(DTensor.from_local(
            torch.ones(2, dtype=torch.bfloat16), mesh, (Shard(0),),
        ))
        parameter.model_name = "weight"
        model = torch.nn.Module()
        model.register_parameter("weight", parameter)
        adamw = AdamW([parameter], lr=0.1)
        optimizer = Float16OptimizerWithFloat16Params(ChainedOptimizer(model, {"adamw": adamw}), model)
        parameter.to_local().sum().backward()
        with patch.object(model, "expert_parameters", return_value=(parameter,), create=True):
            benchmark._prepare_expert_main_gradients(model)  # pylint: disable=protected-access
        self.assertIsInstance(parameter.main_grad, DTensor)
        self.assertEqual(tuple(parameter.main_grad.shape), (4,))
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            optimizer.step()
        self.assertTrue(torch.all(parameter.to_local() < 1))
        self.assertIsInstance(adamw.state[parameter.main_param]["exp_avg"], DTensor)
        self.assertEqual(tuple(adamw.state[parameter.main_param]["exp_avg"].shape), (4,))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_arguments_and_topology_validation(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Arguments and topology validation.
        Expectation: Validate selected kernels and reject incompatible fixed workload shapes.
        """
        args = benchmark.parse_args(["--dispatch-mode", "pull", "--mega-kernels", "gate,mhc"])
        self.assertEqual(args.mega_kernels, ("gate", "mhc"))
        self.assertIsNone(args.initial_capacity_factor)
        self.assertIsNone(args.capacity_growth_factor)
        self.assertEqual(benchmark.parse_args([]).mega_kernels, ("moe", "gate", "mhc"))
        self.assertEqual(benchmark.parse_args([]).accuracy_mode, "components")
        self.assertEqual(benchmark.parse_args(["--accuracy-mode", "full_step"]).accuracy_mode, "full_step")
        self.assertEqual(replace(self._config(), local_num_tokens=129).local_num_tokens, 129)
        with self.assertRaisesRegex(ValueError, "Unsupported Mega kernels"):
            MoeTransformerModel(self._config(), mega_kernels=("invalid",))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_kernel_selection_rejects_invalid_comma_separated_values(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Kernel selection rejects invalid comma separated values.
        Expectation: CLI component selection is explicit, nonempty and free of duplicate names.
        """
        for value in ("", "moe,", "unknown", "gate,gate"):
            with (self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                benchmark.parse_args(["--mega-kernels", value])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_component_comparison_checks_all_outputs_and_gradients(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Component comparison checks all outputs and gradients.
        Expectation: Use one shared input set to check five outputs and all nine gradient groups.
        """
        models = {"common": self._model(()), "mega": self._model(("moe", "gate", "mhc"))}
        for model in models.values():
            self.addCleanup(model.close)
        context = SimpleNamespace(config=self._config(), args=SimpleNamespace(seed=42), rank=0,
                                  device=torch.device("cpu"), models=models)

        def compare(name: str, expected: dict, actual: dict, device: object, **tolerances: float) -> dict:
            """Compare every captured tensor with the production tolerances."""
            del name, device
            self.assertEqual(set(expected), set(actual))
            for key, value in expected.items():
                torch.testing.assert_close(actual[key], value, **tolerances)
            return {"passed": True, "tensor_count": len(actual)}

        with (
            patch.object(benchmark, "_SEQUENCE_LENGTH", 128),
            patch.object(benchmark, "_compare_tensor_maps", side_effect=compare),
            patch("hyper_parallel.core.multicore.modules.mega_mhc.module.hyper_mega_mhc",
                  side_effect=_native_mhc_reference),
        ):
            result = benchmark._compare_mhc_component(context)  # pylint: disable=protected-access
        for backend in ("common", "mega"):
            self.assertEqual(result[f"{backend}_forward"]["tensor_count"], 5)
            self.assertEqual(result[f"{backend}_gradients"]["tensor_count"], 9)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_component_scope_restores_native_for_complete_training(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Component scope restores native for complete training.
        Expectation: Keep strict controlled checks separate from fully enabled training steps.
        """
        models = {"common": self._model(()), "mega": self._model(("moe", "gate", "mhc"))}
        for model in models.values():
            self.addCleanup(model.close)
        args = SimpleNamespace(accuracy_mode="components", mega_kernels=("moe", "gate", "mhc"))
        context = SimpleNamespace(args=args, models=models, device=torch.device("cpu"))
        workloads = {"common": object(), "mega": object()}

        def compare(*unused: object) -> tuple:
            """Observe the selected boundary paths during strict comparison."""
            del unused
            self.assertFalse(models["mega"].layers[0].attention_mhc.use_mega_mhc)
            self.assertFalse(models["mega"].layers[0].ffn_mhc.use_mega_mhc)
            return {}, {}

        def validate(workload: object) -> tuple:
            """Observe the restored paths during the complete training check."""
            self.assertIn(workload, workloads.values())
            self.assertTrue(models["mega"].layers[0].attention_mhc.use_mega_mhc)
            return {"loss_finite": True}, 1.0, None, None, {}

        with (
            patch.object(benchmark, "_compare_mhc_component", return_value={"passed": True}),
            patch.object(benchmark, "_compare_first_step_accuracy", side_effect=compare),
            patch.object(benchmark, "_validate_first_step", side_effect=validate) as training,
            patch.object(torch.npu, "empty_cache"),
        ):
            accuracy, steps = benchmark._validate_accuracy(context, workloads)  # pylint: disable=protected-access
        self.assertEqual(accuracy["scope"], "mhc_component_and_full_step_with_common_mhc")
        self.assertEqual(training.call_count, 2)
        self.assertEqual(set(steps), {"common", "mega"})
        with (
            patch.object(benchmark, "_compare_mhc_component", return_value={"passed": True}),
            patch.object(benchmark, "_compare_first_step_accuracy", side_effect=RuntimeError("controlled failure")),
            self.assertRaisesRegex(RuntimeError, "controlled failure"),
        ):
            benchmark._validate_accuracy(context, workloads)  # pylint: disable=protected-access
        self.assertTrue(models["mega"].layers[0].attention_mhc.use_mega_mhc)


class TestMhcBoundary(unittest.TestCase):
    """Compare all outputs and gradients with the independent semantic oracle."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reference_forward_and_backward(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Reference forward and backward.
        Expectation: Preserve shifted input mappings and FP32 predictor gradients.
        """
        torch.manual_seed(20)
        boundary = MhcBoundary(128, 1e-6, use_mega_mhc=False)
        state = (
            torch.randn(1, 128, 128, dtype=torch.bfloat16),
            torch.randn(1, 128, 4, 128, dtype=torch.bfloat16),
            torch.rand(1, 128, 4), torch.rand(1, 128, 4), torch.rand(1, 128, 4, 4),
        )
        actual_inputs = tuple(t.detach().clone().requires_grad_() for t in (*state, *boundary.parameters()))
        expected_inputs = tuple(t.detach().clone().requires_grad_() for t in actual_inputs)
        # Parameter order is phi, alpha, bias, norm_weight, identical to the public module.
        actual_state = actual_inputs[:5]
        actual = boundary(*actual_state)
        expected = torch_mega_mhc(*expected_inputs)
        probes = tuple(torch.randn_like(t) for t in actual)
        actual_grads = torch.autograd.grad(
            sum((t.float() * p.float()).sum() for t, p in zip(actual, probes)),
            (*actual_state, *boundary.parameters()),
        )
        expected_grads = torch.autograd.grad(
            sum((t.float() * p.float()).sum() for t, p in zip(expected, probes)), expected_inputs,
        )
        for got, want in zip((*actual, *actual_grads), (*expected, *expected_grads)):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertEqual(boundary.phi.dtype, torch.float32)
        self.assertEqual(boundary.norm_weight.dtype, torch.bfloat16)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_projection_restores_hf32_after_failure(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Projection restores hf32 after failure.
        Expectation: Never leave the process-wide NPU setting changed when projection raises.
        """
        residual = SimpleNamespace(
            float=lambda: torch.ones(1, 2, 4, 128), device=SimpleNamespace(type="npu"),
        )
        for previous in (False, True):
            with self.subTest(previous=previous):
                matmul_config = SimpleNamespace(allow_hf32=previous)
                with (patch.object(torch, "npu", SimpleNamespace(matmul=matmul_config), create=True),
                      patch.object(torch, "matmul", side_effect=RuntimeError("projection failed")),
                      self.assertRaisesRegex(RuntimeError, "projection failed")):
                    # Exercise the NPU setting boundary without an accelerator.
                    # pylint: disable-next=protected-access
                    mhc_module._mapping_logits(residual, torch.ones(24, 512), 1e-6)
                self.assertEqual(matmul_config.allow_hf32, previous)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_common_runs_without_omni_registration(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Common runs without omni registration.
        Expectation: The ordinary comparison is a Torch formula without an extra vendor dependency.
        """
        boundary = MhcBoundary(128, 1e-6, use_mega_mhc=False)
        hidden = torch.randn(1, 64, 128, dtype=torch.bfloat16, requires_grad=True)
        with patch.dict(sys.modules, {"omni_training_custom_ops": None}):
            state = MhcState.initialize(hidden)
            result = boundary(state)
            result.hidden_states.float().square().sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bootstrap_and_final_flush(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Bootstrap and final flush.
        Expectation: Do not drop the final block's output or pending residual mappings.
        """
        hidden = torch.randn(2, 128, 128, dtype=torch.bfloat16)
        state = MhcState.initialize(hidden)
        torch.testing.assert_close(state.finish(), hidden, rtol=0, atol=0)
        output = torch.randn_like(hidden, requires_grad=True)
        residual, pre, _, mix = state[1:]
        post = torch.ones_like(pre)
        final = MhcState(output, residual, pre, post, mix).finish()
        torch.testing.assert_close(final, (hidden.float() + output.float()).to(hidden.dtype), rtol=0, atol=0)
        final.float().sum().backward()
        torch.testing.assert_close(output.grad, torch.ones_like(output), rtol=0, atol=0)
