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

"""Hardware-independent contracts for the MoE model and its EP benchmark."""

from __future__ import annotations

import importlib
import io
import json
import sys
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hyper_parallel import DeviceMesh, DTensor, Shard, SkipDTensorDispatch
from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.multicore import MegaMhc, MegaMoeExperts
from hyper_parallel.core.multicore.examples import _training as training
from hyper_parallel.core.multicore.examples.moe import benchmark
from hyper_parallel.core.multicore.examples.moe import experts as moe_module
from hyper_parallel.core.multicore.examples.moe.model import (
    MoeTransformerConfig,
    MoeTransformerModel,
)
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer

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
            ep_size=1,
        )

    @staticmethod
    def _model(kernels: tuple[str, ...]) -> MoeTransformerModel:
        """Initialize equal BF16 weights while keeping Router correction bias FP32."""
        torch.manual_seed(30)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            return MoeTransformerModel(TestMoeTransformerExample._config(), mega_kernels=kernels)
        finally:
            torch.set_default_dtype(original_dtype)

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
        for use_mega_gate in (False, True):
            with self.subTest(use_mega_gate=use_mega_gate), patch.object(gate, "use_mega_gate", use_mega_gate):
                with self.assertRaisesRegex(ValueError, "does not support image_mask"):
                    gate(hidden, image_mask=torch.ones(1, 128, dtype=torch.bool))
        _, weights, ids = gate.native_forward(hidden)
        _, expected_weights, expected_ids = gate.mega_forward(hidden)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
        probe = torch.randn_like(weights)
        actual_grads = torch.autograd.grad((weights * probe).sum(), (hidden, gate.weight))
        expected_grads = torch.autograd.grad((expected_weights * probe).sum(), (hidden, gate.weight))
        for got, want in zip(actual_grads, expected_grads):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_two_layer_training_with_plain_residuals(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Two-layer training with ordinary residual connections.
        Expectation: Keep ordinary residuals, logits and all gradients equal across expert choices.
        """
        reference = self._model(())
        optimized = self._model(("moe", "gate"))
        self.addCleanup(reference.close)
        self.addCleanup(optimized.close)
        inputs = torch.arange(128).reshape(1, -1) % 32
        with (
            patch.object(MegaMoeExperts, "forward", _experts_reference),
            patch.object(rms_norm_module.torch_npu, "npu_rms_norm", _rms_reference),
            patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul", _rotary_reference),
        ):
            results = [model(inputs, labels=inputs) for model in (reference, optimized)]
            for result in results:
                result["loss"].backward()
        self.assertFalse(any(isinstance(module, MegaMhc) for module in optimized.modules()))
        self.assertTrue(hasattr(optimized.layers[0], "input_norm"))
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

    def test_selective_backend_construction(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Selective backend construction.
        Expectation: Use common EP unless moe is explicitly selected in the optimized backend.
        """
        for backend, kernels in (("common", ("moe", "gate")), ("mega", ("gate",)), ("mega", ("moe",))):
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
                self.assertFalse(any(isinstance(module, MegaMhc) for module in model.modules()))
                for layer in model.layers:
                    if isinstance(layer.mlp.experts, MegaMoeExperts):
                        layer.mlp.experts.close()

    def test_mixed_dtype_gradient_synchronization(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Mixed dtype gradient synchronization.
        Expectation: Average dense and owner expert gradients exactly once, with dtype-specific collectives.
        """
        gradients = [torch.ones(3, dtype=torch.bfloat16), torch.ones(2), torch.full((5,), 2.0)]
        expected = [gradient.clone() for gradient in gradients]
        expert = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
        source_losses = (torch.arange(1, 9).reshape(-1, 1) * expert).sum(-1)
        expert.grad, = torch.autograd.grad(source_losses.sum(), expert, retain_graph=True)
        expected_expert, = torch.autograd.grad(source_losses.mean(), expert)
        unused = torch.nn.Parameter(torch.ones(1))
        model = SimpleNamespace(expert_parameters=lambda: (expert, unused))

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
            benchmark._synchronize_gradients(model, 8)  # pylint: disable=protected-access
        self.assertEqual(all_reduce.call_count, 2)
        torch.testing.assert_close(expert.grad, expected_expert, rtol=0, atol=0)
        self.assertIsNone(unused.grad)
        for got, want in zip(gradients, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    @patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
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
        with (patch.object(model, "expert_parameters", return_value=(parameter,), create=True),
              patch.object(benchmark, "_dense_gradients", return_value=[])):
            benchmark._synchronize_gradients(model, 2)
            benchmark._prepare_expert_main_gradients(model)  # pylint: disable=protected-access
        self.assertIsInstance(parameter.main_grad, DTensor)
        self.assertEqual(tuple(parameter.main_grad.shape), (4,))
        torch.testing.assert_close(
            parameter.main_grad.to_local(), torch.full((2,), 0.5, dtype=parameter.dtype), rtol=0, atol=0)
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            optimizer.step()
        self.assertTrue(torch.all(parameter.to_local() < 1))
        self.assertIsInstance(adamw.state[parameter.main_param]["exp_avg"], DTensor)
        self.assertEqual(tuple(adamw.state[parameter.main_param]["exp_avg"].shape), (4,))

    def test_arguments_and_topology_validation(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Arguments and topology validation.
        Expectation: Validate selected kernels and reject incompatible workload shapes.
        """
        args = benchmark.parse_args(["--dispatch-mode", "pull", "--mega-kernels", "gate"])
        self.assertEqual(args.mega_kernels, ("gate",))
        self.assertIsNone(args.initial_capacity_factor)
        self.assertIsNone(args.capacity_growth_factor)
        self.assertEqual(benchmark.parse_args([]).mega_kernels, ("moe", "gate"))
        self.assertEqual(replace(self._config(), max_seq_len=129).max_seq_len, 129)
        with self.assertRaisesRegex(ValueError, "Unsupported Mega kernels"):
            MoeTransformerModel(self._config(), mega_kernels=("invalid",))
        for value in ("", "moe,", "unknown", "gate,gate", "mhc"):
            with (self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                benchmark.parse_args(["--mega-kernels", value])

    def test_expert_configuration_tracks_runtime_world_size(self) -> None:
        """Feature: Configurable MoE benchmark EP.
        Description: Prepare different runtime sizes and explicit expert-count overrides.
        Expectation: EP follows actual ranks, defaults preserve six local experts and overrides are validated.
        """
        for world_size in (1, 2, 4, 8, 16):
            with (self.subTest(world_size=world_size),
                  patch.object(benchmark, "_init_runtime", return_value=(0, world_size, torch.device("cpu"), False))):
                context = benchmark._prepare_benchmark([])  # pylint: disable=protected-access
                self.assertEqual(context.world_size, world_size)
                self.assertEqual(context.config.ep_size, world_size)
                self.assertEqual(context.config.num_experts, 6 * world_size)
                self.assertEqual(context.config.top_k, min(8, 6 * world_size))
        with patch.object(benchmark, "_init_runtime", return_value=(0, 4, torch.device("cpu"), False)):
            context = benchmark._prepare_benchmark(  # pylint: disable=protected-access
                ["--num-experts", "48", "--top-k", "4"])
        self.assertEqual((context.config.ep_size, context.config.num_experts, context.config.top_k), (4, 48, 4))

    def test_invalid_expert_configuration_releases_only_owned_group(self) -> None:
        """Feature: Configurable MoE benchmark EP.
        Description: Reject non-divisible experts and invalid TopK after runtime initialization.
        Expectation: Failed preparation cleans up its own process group and preserves caller-owned groups.
        """
        cases = [(["--num-experts", "25"], "divisible"), (["--num-experts", "0"], "positive"),
                 (["--top-k", "0"], "positive"), (["--top-k", "25"], "cannot exceed")]
        for owned in (False, True):
            for argv, error in cases:
                with (self.subTest(owned=owned, argv=argv),
                      patch.object(benchmark, "_init_runtime", return_value=(0, 4, torch.device("cpu"), owned)),
                      patch.object(benchmark.dist, "destroy_process_group") as destroy,
                      self.assertRaisesRegex(ValueError, error)):
                    benchmark._prepare_benchmark(argv)  # pylint: disable=protected-access
                self.assertEqual(destroy.call_count, int(owned))

    def test_runtime_accepts_non_eight_rank_groups(self) -> None:
        """Feature: Configurable MoE benchmark EP.
        Description: Bind the last local device of an EP16 group without executing device operations.
        Expectation: Runtime accepts the actual topology and reports group ownership correctly.
        """
        for initialized in (False, True):
            with (self.subTest(initialized=initialized),
                  patch.dict(benchmark.os.environ, {"LOCAL_RANK": "15"}),
                  patch.object(torch.npu, "set_device") as select,
                  patch.object(benchmark.dist, "is_initialized", return_value=initialized),
                  patch.object(benchmark.dist, "init_process_group") as initialize,
                  patch.object(benchmark.dist, "get_rank", return_value=15),
                  patch.object(benchmark.dist, "get_world_size", return_value=16)):
                rank, size, device, owned = benchmark._init_runtime()  # pylint: disable=protected-access
            self.assertEqual((rank, size, device.index, owned), (15, 16, 15, not initialized))
            select.assert_called_once_with(15)
            self.assertEqual(initialize.call_count, int(not initialized))

    def test_result_records_configured_topology_and_experts(self) -> None:
        """Feature: Configurable MoE benchmark EP.
        Description: Write a result for an EP16 run using the default expert count.
        Expectation: JSON records the actual EP and expert shape instead of the old eight-rank constant.
        """
        backends = {name: {"steady_state_optimizer_step_ms": {"median": value}}
                    for name, value in (("common", 2.0), ("mega", 1.0))}
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parse_args(["--output", str(Path(directory) / "result.json")])
            config = benchmark._build_config(args, 16)  # pylint: disable=protected-access
            with redirect_stderr(io.StringIO()), patch("builtins.print"), patch.object(benchmark.dist, "barrier"):
                benchmark._write_result(args, config, 0, {}, backends)  # pylint: disable=protected-access
            result = json.loads(Path(args.output).read_text(encoding="utf-8"))
        self.assertEqual(result["topology"], {"world_size": 16, "tp": 1, "ep": 16})
        self.assertEqual((result["shape"]["num_experts"], result["shape"]["top_k"]), (96, 8))


    def test_import_does_not_load_mhc_implementation(self) -> None:
        """Feature: Independent MoE example.
        Description: Import the model and benchmark in a fresh interpreter.
        Expectation: Neither the mHC example nor managed mHC implementation is imported.
        """
        code = """import sys
from hyper_parallel.core.multicore.examples.moe import benchmark, model
assert not any(name.startswith('hyper_parallel.core.multicore.examples.mhc') for name in sys.modules)
assert 'hyper_parallel.core.multicore.modules.mega_mhc.module' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_tensor_comparison_rejects_a_peer_failure(self) -> None:
        """Feature: Distributed model comparison.
        Description: Emulate a failed peer while the local common/Mega tensors are equal.
        Expectation: Shared comparison still converges failures and maximum errors across EP ranks.
        """
        operations = []

        def _merge_peer(tensor: torch.Tensor, *, op: object) -> None:
            """Inject one remote failure and its worst-case diagnostics."""
            operations.append(op)
            tensor.fill_(0 if op == training.dist.ReduceOp.MIN else 5)

        with (patch.object(training.dist, "is_initialized", return_value=True),
              patch.object(training.dist, "all_reduce", side_effect=_merge_peer),
              self.assertRaisesRegex(RuntimeError, "peer validation") as failure):
            # pylint: disable-next=protected-access
            training._compare_tensor_maps("peer validation", {"weight": torch.ones(2)},
                                          {"weight": torch.ones(2)}, torch.device("cpu"), rtol=0.02, atol=0.002)
        self.assertIn("'max_absolute_error': 5.0", str(failure.exception))
        self.assertEqual(operations, [training.dist.ReduceOp.MIN, training.dist.ReduceOp.MAX,
                                      training.dist.ReduceOp.MAX])
