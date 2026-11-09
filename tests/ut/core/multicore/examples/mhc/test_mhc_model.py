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

"""CPU contracts for the independent dense mHC model and single-device benchmark."""

from __future__ import annotations

from contextlib import redirect_stderr
import importlib
import io
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.examples._training import build_optimizer
from hyper_parallel.core.multicore.examples.mhc import benchmark
from hyper_parallel.core.multicore.examples.mhc import mhc as mhc_module
from hyper_parallel.core.multicore.examples.mhc.mhc import MhcBoundary
from hyper_parallel.core.multicore.examples.mhc.model import (
    MhcTransformerConfig, MhcTransformerModel, _finish_state, _initialize_state,
)
from tests.common.multicore.mega_mhc_golden import torch_mega_mhc

rms_norm_module = importlib.import_module("hyper_parallel.components.functional.rms_norm")


def _rms_reference(hidden: torch.Tensor, weight: torch.Tensor, *, epsilon: float) -> tuple:
    """Keep the dense CPU model independent of the NPU RMSNorm registration."""
    normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(dim=-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(hidden.dtype), None


def _rotary_reference(hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, rotary_mode: str) -> torch.Tensor:
    """Apply the half-rotation used by the dense model on CPU."""
    if rotary_mode != "half":
        raise ValueError(f"Unexpected rotary mode: {rotary_mode}")
    first, second = hidden.chunk(2, dim=-1)
    return hidden * cos + torch.cat((-second, first), dim=-1) * sin


def _native_mhc_reference(*args: torch.Tensor, token_tile: int, **kwargs: float) -> tuple:
    """Execute the native boundary's mathematical oracle in CPU model tests."""
    del token_tile
    return torch_mega_mhc(*args, **kwargs)


class TestDenseMhcModel(unittest.TestCase):
    """Check dense architecture isolation, complete training and fail-fast launch rules."""

    @staticmethod
    def _config() -> MhcTransformerConfig:
        return MhcTransformerConfig(vocab_size=32, hidden_size=128, num_layers=2, num_attention_heads=4,
                                    num_key_value_heads=2, max_seq_len=128, intermediate_size=128)

    def _models(self) -> dict:
        result = {}
        for name in ("common", "mega"):
            torch.manual_seed(32)
            result[name] = MhcTransformerModel(self._config(), use_mega_mhc=name == "mega").bfloat16()
        return result

    def _workloads(self) -> dict:
        inputs = torch.arange(128).reshape(1, -1) % 32
        return {name: benchmark._Workload(  # pylint: disable=protected-access
            model, build_optimizer(model, learning_rate=1e-3, weight_decay=0.01), inputs, inputs.clone(),
            torch.device("cpu")) for name, model in self._models().items()}

    def test_state_bootstrap_and_final_pending_output_preserve_gradients(self) -> None:
        """Feature: Model-owned mHC state.
        Description: Initialize independent states and apply the final pending block output.
        Expectation: Preserve leading dimensions, FP32 mappings and gradients into both endpoints.
        """
        hidden = torch.randn(2, 3, 128, requires_grad=True)
        first, second = _initialize_state(hidden), _initialize_state(hidden * 2)
        torch.testing.assert_close(_finish_state(first), hidden, rtol=0, atol=0)
        torch.testing.assert_close(_finish_state(second), hidden * 2, rtol=0, atol=0)
        _finish_state(first).sum().backward()
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
        self.assertEqual(first[2].dtype, torch.float32)
        hidden = torch.randn(2, 128, 128, dtype=torch.bfloat16)
        output = torch.randn_like(hidden, requires_grad=True)
        state = _initialize_state(hidden)
        torch.testing.assert_close(_finish_state(state), hidden, rtol=0, atol=0)
        _, residual, pre, _, mix = state
        final = _finish_state((output, residual, pre, torch.ones_like(pre), mix))
        torch.testing.assert_close(final, (hidden.float() + output.float()).to(hidden.dtype), rtol=0, atol=0)
        final.float().sum().backward()
        torch.testing.assert_close(output.grad, torch.ones_like(output), rtol=0, atol=0)

    def test_dense_architecture_contains_no_router_or_experts(self) -> None:
        """Feature: Independent mHC example.
        Description: Construct a dense model without an expert-parallel configuration.
        Expectation: Every layer has a dense FFN and mHC boundaries but no Router or expert module.
        """
        model = self._models()["mega"]
        self.assertFalse(hasattr(model.config, "ep_size"))
        self.assertFalse(hasattr(model, "expert_parameters"))
        for layer in model.layers:
            self.assertFalse(hasattr(layer.mlp, "gate"))
            self.assertFalse(hasattr(layer.mlp, "experts"))
            self.assertTrue(hasattr(layer.mlp, "gate_up"))
            self.assertIsInstance(layer.attention_mhc, MhcBoundary)
        for layer in model.layers:
            self.assertEqual(layer.attention_mhc.phi.dtype, torch.float32)
            self.assertEqual(layer.mlp.gate_up.weight.dtype, torch.bfloat16)

    def test_complete_dense_training_and_optimizer_parity(self) -> None:
        """Feature: Dense mHC training parity.
        Description: Compare actual model loss, logits, every gradient and FP32-master AdamW updates.
        Expectation: Native-oracle and ordinary boundaries agree without disabling mHC during training.
        """
        workloads = self._workloads()
        with (
            patch.object(rms_norm_module.torch_npu, "npu_rms_norm", _rms_reference),
            patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul", _rotary_reference),
            patch("hyper_parallel.core.multicore.modules.mega_mhc.module.hyper_mega_mhc",
                  side_effect=_native_mhc_reference) as native,
        ):
            result = benchmark._validate_models(workloads)  # pylint: disable=protected-access
        self.assertEqual(native.call_count, 4)
        self.assertEqual(result["scope"], "full_dense_model_with_selected_mhc")
        for name in ("initial_parameters", "forward", "gradients", "updated_parameters"):
            self.assertTrue(result[name]["passed"])
        expected = sum(p.requires_grad for p in workloads["mega"].model.parameters())
        self.assertEqual(result["gradients"]["tensor_count"], expected)
        self.assertTrue(all(layer.attention_mhc.use_mega_mhc for layer in workloads["mega"].model.layers))

    def test_boundary_comparison_keeps_all_outputs_and_gradients(self) -> None:
        """Feature: mHC boundary precision.
        Description: Compare the five outputs and nine input/parameter gradients before full-model validation.
        Expectation: Both implementations exactly match the independent oracle for all outputs and gradients.
        """
        models = self._models()
        with (patch("hyper_parallel.core.multicore.modules.mega_mhc.module.hyper_mega_mhc",
                    side_effect=_native_mhc_reference),
              patch.object(benchmark, "_RTOL", 0), patch.object(benchmark, "_ATOL", 0)):
            result = benchmark._compare_mhc_component(models, self._config(), 42, torch.device("cpu"))
        for backend in ("common", "mega"):
            self.assertEqual(result[f"{backend}_forward"]["tensor_count"], 5)
            self.assertEqual(result[f"{backend}_gradients"]["tensor_count"], 9)

    def test_corrupted_initial_parameters_are_rejected(self) -> None:
        """Feature: Matched initial state.
        Description: Change one dense parameter before paired training.
        Expectation: Full-model validation fails before accepting unequal initial weights.
        """
        workloads = self._workloads()
        with torch.no_grad():
            workloads["mega"].model.lm_head.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "initial parameters"):
            benchmark._validate_models(workloads)  # pylint: disable=protected-access

    def test_multi_rank_launch_is_rejected_before_device_initialization(self) -> None:
        """Feature: Local mHC execution.
        Description: Attempt the single-device example under a multi-rank launcher.
        Expectation: Reject it before selecting a device or creating distributed state.
        """
        with (patch.dict(os.environ, {"WORLD_SIZE": "8"}), patch.object(torch.npu, "set_device") as select,
              self.assertRaisesRegex(ValueError, "one device")):
            benchmark._build_workloads(benchmark.parse_args([]))  # pylint: disable=protected-access
        select.assert_not_called()

    def test_small_inputs_use_the_native_token_limit(self) -> None:
        """Feature: Device-dependent mHC training shapes.
        Description: Reject inputs below each device's native backward reduction minimum.
        Expectation: Device limits are checked before allocating the example model or input batch.
        """
        for cores, tokens, minimum in (((20, 40), 39, 40), ((24, 48), 40, 48)):
            with (self.subTest(cores=cores), patch.dict(os.environ, {"WORLD_SIZE": "1"}),
                  patch.object(torch.npu, "set_device"),
                  patch.object(benchmark, "_resolve_core_counts", return_value=cores),
                  patch.object(benchmark, "MhcTransformerModel") as build,
                  self.assertRaisesRegex(ValueError, f"at least {minimum} tokens")):
                benchmark._build_workloads(benchmark.parse_args(["--tokens", str(tokens)]))
            build.assert_not_called()

    def test_cli_has_no_expert_or_communication_options(self) -> None:
        """Feature: Independent mHC command line.
        Description: Validate local workload arguments and reject MoE-only controls.
        Expectation: mHC has no kernel-subset, communication-mode or EP configuration.
        """
        args = benchmark.parse_args([])
        self.assertFalse(hasattr(args, "mega_kernels"))
        self.assertEqual(args.device_index, 0)
        for option in ("--mega-kernels", "--dispatch-mode"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parse_args([option, "moe"])
        with self.assertRaisesRegex(ValueError, "tokens must be positive"):
            benchmark.parse_args(["--tokens", "0"])

    def test_import_does_not_load_moe_or_initialize_distributed(self) -> None:
        """Feature: Independent dense example.
        Description: Import the model and benchmark in a fresh interpreter.
        Expectation: No MoE implementation or process group is needed by the dense example.
        """
        code = """import sys
import torch.distributed as dist
from hyper_parallel.core.multicore.examples.mhc import benchmark, model
assert not any(name.startswith('hyper_parallel.core.multicore.examples.moe') for name in sys.modules)
assert 'hyper_parallel.core.multicore.modules.mega_moe.module' not in sys.modules
assert not dist.is_initialized()
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


class TestMhcBoundary(unittest.TestCase):
    """Compare all outputs and gradients with the independent semantic oracle."""


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

    def test_common_runs_without_omni_registration(self) -> None:
        """Feature: MegaKernel model integration.
        Description: Common runs without omni registration.
        Expectation: The ordinary comparison is a Torch formula without an extra vendor dependency.
        """
        boundary = MhcBoundary(128, 1e-6, use_mega_mhc=False)
        hidden = torch.randn(1, 64, 128, dtype=torch.bfloat16, requires_grad=True)
        with patch.dict(sys.modules, {"omni_training_custom_ops": None}):
            state = _initialize_state(hidden)
            *_, hidden_states = boundary(*state)
            hidden_states.float().square().sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
