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
"""Low-precision recovery contracts and CPU state-repair continuation."""

import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_linear_func import HiFloat8LinearStrategy
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import MXFP8LinearStrategy
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.modules.linear import LowPrecisionLinear
from hyper_parallel.components.quantization.quantizers.hifloat8 import HiFloat8Quantizer
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.components.quantization.tensor.mxfp8_tensor import MXFP8Tensor
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.high_availability.model_recovery import ModelRecoveryAdapter
from hyper_parallel.high_availability.recovery_state import ModelOptimizerState
from tests.ut.components.quantization.test_memory_contracts import IdentityQuantizer
from tests.ut.components.quantization.test_mxfp8_memory import IdentityMXOps


class TestLowPrecisionRecovery(unittest.TestCase):
    """Use native state codecs; replace only hardware quantization and matmul."""

    def setUp(self) -> None:
        """Keep deterministic probes from changing other tests' RNG state."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(42)

    def test_builtin_strategies_and_family_quantizer_preserve_parameter_identity(self) -> None:
        """Feature: Admit dynamic low precision independently of config selectors.

        Description: Bind each built-in strategy and a directly held family quantizer.
        Expectation: Adapter creation preserves persistent parameter objects and values.
        """
        for strategy_type in (
            MXFP8LinearStrategy,
            HiFloat8LinearStrategy,
            MXFP8GroupedLinear,
            HiFloat8GroupedLinear,
            W4A8GroupedLinear,
            FakeW4A8GroupedLinear,
        ):
            with self.subTest(strategy=strategy_type.__name__):
                model = torch.nn.Linear(32, 32)
                model.strategy = strategy_type()
                original = model.weight
                expected = original.detach().clone()
                ModelRecoveryAdapter(model)
                self.assertIs(model.weight, original)
                torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
        model = torch.nn.Linear(32, 32)
        model.quantizer = MXFP8Quantizer()
        ModelRecoveryAdapter(model)

    def test_custom_quantizers_and_strategy_subclasses_fail_before_attach(self) -> None:
        """Feature: Reject unregistered quantization history before recovery starts.

        Description: Replace a strategy or its quantizer, including a directly held quantizer.
        Expectation: Construction identifies the unsupported object's module path.
        """
        class HistoryStrategy(MXFP8LinearStrategy):
            """Represent an extension with state outside registered buffers."""

        class HistoryQuantizer(MXFP8Quantizer):
            """Represent a delayed-scaling extension."""

        for strategy in (
            HistoryStrategy(),
            MXFP8LinearStrategy(HistoryQuantizer()),
            HiFloat8LinearStrategy(input_quantizer=object()),
        ):
            model = LowPrecisionLinear.from_linear(torch.nn.Linear(32, 32), strategy=strategy)
            with self.subTest(strategy=type(strategy).__name__):
                with self.assertRaisesRegex(ValueError, "<root>.strategy.*recovery adapter"):
                    ModelRecoveryAdapter(model)
        model = torch.nn.Linear(32, 32)
        model.quantizer = HistoryQuantizer()
        with self.assertRaisesRegex(ValueError, "<root>.quantizer.*recovery adapter"):
            ModelRecoveryAdapter(model)

    def test_persistent_packed_weights_and_quantized_buffers_are_rejected(self) -> None:
        """Feature: Keep quantization payloads outside the persistent repair protocol.

        Description: Bind packed integer weights or a registered quantized tensor buffer.
        Expectation: Fail before state fingerprinting or communication can begin.
        """
        model = torch.nn.Linear(32, 32)
        model.strategy = MXFP8LinearStrategy()
        model.weight = torch.nn.Parameter(torch.ones(32, 32, dtype=torch.uint8), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "high-precision parameters"):
            ModelRecoveryAdapter(model)
        model = torch.nn.Linear(32, 32)
        payload = torch.ones(32, 32)
        model.register_buffer("packed", MXFP8Tensor(
            payload.shape,
            payload.dtype,
            quantizer=MXFP8Quantizer(),
            row_data=payload,
            row_scale=torch.ones_like(payload),
        ))
        with self.assertRaisesRegex(ValueError, "persistent quantized buffers"):
            ModelRecoveryAdapter(model)

    @staticmethod
    def _model(format_name: str, optimizer_type: type) -> tuple:
        """Construct ordinary persistent weights with a native dynamic strategy."""
        strategy = (MXFP8LinearStrategy(MXFP8Quantizer(npu_ops=IdentityMXOps()))
                    if format_name == "mxfp8" else HiFloat8LinearStrategy())
        model = LowPrecisionLinear.from_linear(
            torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16),
            strategy=strategy,
        )
        optimizer = (AdamW(model.parameters(), lr=0.001) if optimizer_type is AdamW
                     else Muon(model.parameters(), lr=0.001, ns_steps=2))
        adapter = ModelRecoveryAdapter(model)
        return model, optimizer, ModelOptimizerState(model, optimizer), adapter

    @staticmethod
    def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, value: torch.Tensor) -> None:
        """Run native low-precision autograd and a real optimizer update on CPU."""
        optimizer.zero_grad(set_to_none=True)
        model(value).float().square().mean().backward()
        optimizer.step()

    def test_repair_and_repeated_continuation_match_donor(self) -> None:
        """Feature: Repair dynamic computation using only high-precision persistent state.

        Description: Interrupt a cold receiver's forward, then corrupt a later update and repair again.
        Expectation: Subsequent parameters and AdamW/Muon slots exactly match the donor;
            no quantization buffers or extra state are mirrored.
        """
        quantizer = IdentityQuantizer()
        with patch.object(HiFloat8Quantizer, "quantize", side_effect=quantizer.quantize):
            for format_name, optimizer_type in (
                ("mxfp8", AdamW),
                ("mxfp8", Muon),
                ("hifloat8", AdamW),
                ("hifloat8", Muon),
            ):
                with self.subTest(format=format_name, optimizer=optimizer_type.__name__):
                    donor, donor_optimizer, donor_state, _ = self._model(format_name, optimizer_type)
                    target, target_optimizer, target_state, adapter = self._model(format_name, optimizer_type)
                    value = torch.randn(5, 32, dtype=torch.bfloat16)
                    self._step(donor, donor_optimizer, value)
                    for attempt in range(2):
                        rank_state = target_state.rank_state()
                        with self.assertRaisesRegex(RuntimeError, "injected failure"):
                            loss = target(value).float().square().mean()
                            if attempt:
                                loss.backward()
                                with torch.no_grad():
                                    target.weight.add_(1)
                                    for slot in target_optimizer.state[target.weight].values():
                                        if isinstance(slot, torch.Tensor):
                                            slot.add_(1)
                            raise RuntimeError("injected failure")
                        del loss
                        adapter.abort_iteration()
                        adapter.reset_after_device_restart(Mock())
                        target_state.prepare_receive(donor_state.metadata())
                        destination = dict(target_state.tensor_items())
                        with torch.no_grad():
                            for name, tensor in donor_state.tensor_items():
                                destination[name].copy_(tensor)
                        target_state.restore_model_state(rank_state["buffers"], rank_state["extra"])
                        self._step(donor, donor_optimizer, value)
                        self._step(target, target_optimizer, value)
                        destination = dict(target_state.tensor_items())
                        for name, tensor in donor_state.tensor_items():
                            torch.testing.assert_close(destination[name], tensor, rtol=0, atol=0)
                        self.assertEqual(target_state.metadata(), donor_state.metadata())
                        self.assertEqual(target_state.buffers, {})
                        self.assertEqual(target_state.extra_modules, {})
