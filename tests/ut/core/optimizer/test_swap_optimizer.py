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
"""Unit tests for the swap optimizer public API."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.core.optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer import swap_optimizer_base
from hyper_parallel.core.optimizer.adamw import AdamW as NewAdamW
from hyper_parallel.core.optimizer.swap_optimizer import SwapOptimizer, is_swap_optimizer


def _materialize_adam_state(optimizer, param):
    """Populate deterministic Adam state without executing an optimizer update."""
    values = torch.arange(1, param.numel() + 1, dtype=param.dtype, device=param.device).view_as(param)
    state = optimizer.state[param]
    state["step"] = torch.tensor(1.0)
    state["exp_avg"] = values.clone()
    state["exp_avg_sq"] = values.square()
    return state


class TestSwapOptimizerConfig(unittest.TestCase):
    """Configuration validation and defaults."""

    def test_packed_swap_defaults_on_and_is_overridable(self):
        """Packed staging is enabled unless the caller opts out."""
        self.assertTrue(SwapOptimizerConfig().packed_swap)
        self.assertFalse(SwapOptimizerConfig(packed_swap=False).packed_swap)

    def test_defaults_match_the_documented_pipeline_shape(self):
        """A default config uses the documented partition and threshold values."""
        config = SwapOptimizerConfig()

        self.assertEqual(config.swap_times, 16)
        self.assertEqual(config.min_numel, 1024)
        self.assertFalse(config.include_master_params)
        self.assertIsNone(config.state_keys)

    def test_swap_times_must_be_positive(self):
        """A non-positive partition count is rejected at construction."""
        with self.assertRaisesRegex(ValueError, "swap_times must be positive"):
            SwapOptimizerConfig(swap_times=0)

    def test_min_numel_must_be_non_negative(self):
        """A negative swap threshold is rejected at construction."""
        with self.assertRaisesRegex(ValueError, "min_numel must be non-negative"):
            SwapOptimizerConfig(min_numel=-1)

    def test_state_keys_are_validated_and_normalized_to_a_tuple(self):
        """Logical state keys are validated and frozen into a tuple."""
        config = SwapOptimizerConfig(state_keys=["exp_avg_sq", "exp_avg"])

        self.assertEqual(config.state_keys, ("exp_avg_sq", "exp_avg"))

    def test_reject_invalid_state_key(self):
        """Unknown logical state keys are rejected before any optimizer is wrapped."""
        with self.assertRaisesRegex(ValueError, "only supports Adam/AdamW logical slots"):
            SwapOptimizerConfig(state_keys=("exp_avg", "momentum_buffer"))

    def test_master_param_key_is_accepted_for_master_copy_optimizers(self):
        """``master_param`` passes validation even though Adam itself never uses it."""
        config = SwapOptimizerConfig(include_master_params=True, state_keys=("exp_avg", "master_param"))

        self.assertEqual(config.state_keys, ("exp_avg", "master_param"))

    def test_config_is_frozen(self):
        """A validated config cannot be mutated after construction."""
        config = SwapOptimizerConfig()

        with self.assertRaises(Exception):
            config.swap_times = 4


class TestSwapOptimizerFacade(unittest.TestCase):
    """Creation, wrapping and predicate behaviour of the public API."""

    @staticmethod
    def _adam(params):
        """Build a plain Adam optimizer over ``params``."""
        return torch.optim.Adam(list(params), lr=0.01)

    def test_swap_optimizer_wraps_a_supported_optimizer(self):
        """Wrapping an Adam optimizer returns the swap wrapper."""
        param = torch.nn.Parameter(torch.ones(8))

        wrapped = swap_optimizer(self._adam([param]), SwapOptimizerConfig())

        self.assertIsInstance(wrapped, swap_optimizer_base.SwapOptimizer)
        self.assertTrue(is_swap_optimizer(wrapped))
        self.assertIs(wrapped.optimizer, wrapped.adapter.optimizer)

    def test_swap_optimizer_builds_a_torch_optimizer_wrapper(self):
        """The wrapper is a real Torch optimizer that delegates to the base one."""
        param = torch.nn.Parameter(torch.ones(8))
        base = self._adam([param])

        wrapped = swap_optimizer(base, SwapOptimizerConfig())

        self.assertIsInstance(wrapped, torch.optim.Optimizer)
        self.assertEqual(wrapped.param_groups, base.param_groups)
        self.assertTrue(hasattr(wrapped, "state_dict"))

    def test_default_config_is_created_when_omitted(self):
        """Omitting the config falls back to the documented defaults."""
        param = torch.nn.Parameter(torch.ones(8))

        wrapped = swap_optimizer(self._adam([param]))

        self.assertIsInstance(wrapped.config, SwapOptimizerConfig)
        self.assertEqual(wrapped.config.swap_times, 16)
        self.assertTrue(wrapped.config.packed_swap)

    def test_swap_optimizer_class_delegates_to_the_factory(self):
        """``SwapOptimizer(...)`` produces the same wrapper as the factory function."""
        param = torch.nn.Parameter(torch.ones(8))

        wrapped = SwapOptimizer(self._adam([param]), SwapOptimizerConfig(swap_times=2, min_numel=1))

        self.assertIsInstance(wrapped, swap_optimizer_base.SwapOptimizer)
        self.assertEqual(wrapped.config.swap_times, 2)

    def test_is_swap_optimizer_rejects_plain_optimizers(self):
        """A plain Torch optimizer is not reported as a swap optimizer."""
        param = torch.nn.Parameter(torch.ones(8))

        self.assertFalse(is_swap_optimizer(self._adam([param])))
        self.assertFalse(is_swap_optimizer(object()))
        self.assertFalse(is_swap_optimizer(None))
        self.assertFalse(is_swap_optimizer(SimpleNamespace(_is_swap_optimizer=0)))

    def test_unsupported_optimizer_type_is_rejected(self):
        """Optimizers without an Adam/AdamW adapter fail fast."""
        param = torch.nn.Parameter(torch.ones(8))

        with self.assertRaisesRegex(ValueError, "only supports"):
            swap_optimizer(torch.optim.SGD([param], lr=0.01), SwapOptimizerConfig())

    def test_native_adam_rejects_unsupported_true_flags(self):
        """Native Adam adapters reject execution modes the swap pipeline cannot preserve."""
        unsupported_flags = {
            torch.optim.Adam: ("foreach", "fused", "capturable", "differentiable"),
            torch.optim.AdamW: ("foreach", "capturable", "differentiable"),
        }

        for optimizer_type, flags in unsupported_flags.items():
            for flag in flags:
                with self.subTest(optimizer=optimizer_type.__name__, flag=flag):
                    param = torch.nn.Parameter(torch.ones(8))
                    optimizer = optimizer_type([param], **{flag: True})

                    with self.assertRaisesRegex(ValueError, rf"{flag}=True"):
                        swap_optimizer(optimizer, SwapOptimizerConfig())

    def test_native_adamw_accepts_fused(self):
        """Fused AdamW stays supported because its adapter forwards the flag."""
        param = torch.nn.Parameter(torch.ones(8))

        wrapped = swap_optimizer(torch.optim.AdamW([param], lr=0.01, fused=True), SwapOptimizerConfig())

        self.assertIsInstance(wrapped.adapter, swap_optimizer_base.TorchNativeAdamWAdapter)

    def test_prefetch_batches_is_not_configurable(self):
        """The pipeline shape is fixed, so no prefetch knob is exposed."""
        self.assertFalse(hasattr(SwapOptimizerConfig(), "prefetch_batches"))
        self.assertFalse(hasattr(SwapOptimizerConfig(), "num_prefetch"))

    def test_preinitialized_state_is_registered_on_wrap(self):
        """AdamW state materialized before wrapping is picked up immediately."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=0.01)
        _materialize_adam_state(optimizer, param)

        wrapped = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=2, min_numel=1))

        slots = tuple(wrapped.adapter.all_slots())
        self.assertEqual({slot.name for slot in slots}, {"exp_avg", "exp_avg_sq"})
        self.assertEqual(
            {id(slot.tensor) for slot in slots},
            {
                id(optimizer.state[param]["exp_avg"]),
                id(optimizer.state[param]["exp_avg_sq"]),
            },
        )

    def test_new_adamw_optimizer_is_supported(self):
        """hyper-parallel's fused AdamW is accepted by the factory."""
        param = torch.nn.Parameter(torch.ones(8))

        wrapped = swap_optimizer(NewAdamW([param], lr=0.01), SwapOptimizerConfig(swap_times=4, min_numel=1))

        self.assertIsInstance(wrapped, swap_optimizer_base.SwapOptimizer)

    def test_torch_update_units_use_parameter_group_as_adapter_index(self):
        """Units keep the index of the parameter group they came from."""
        params = [torch.nn.Parameter(torch.ones(4)) for _ in range(3)]
        optimizer = torch.optim.Adam([
            {"params": params[:2], "lr": 0.01},
            {"params": params[2:], "lr": 0.02},
        ])
        wrapped = swap_optimizer(optimizer, SwapOptimizerConfig(packed_swap=False, min_numel=1024))
        for param in params:
            param.grad = torch.ones_like(param)

        units = wrapped.adapter.prepare_step()["units"]

        self.assertEqual([unit.adapter_index for unit in units], [0, 0, 1])
        self.assertEqual([id(unit.param) for unit in units], [id(param) for param in params])

    def test_new_adamw_prepare_step_increments_group_once(self):
        """Unit preparation advances the new AdamW group counter exactly once."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = NewAdamW([param], lr=0.01)
        wrapped = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=4, min_numel=1))
        param.grad = torch.ones_like(param)

        units = wrapped.adapter.prepare_step()["units"]

        self.assertEqual(optimizer.param_groups[0]["step"], 1)
        self.assertEqual([id(unit.param) for unit in units], [id(param)])

    def test_packed_fused_adamw_initializes_step_on_the_parameter_device(self):
        """Packed fused AdamW keeps its scalar step beside the fused kernel's parameters."""
        param = torch.nn.Parameter(torch.ones(8, device="meta"))
        optimizer = torch.optim.AdamW([param], lr=0.01, fused=True)
        runtime = swap_optimizer_base.PipelineSwapRuntime(SwapOptimizerConfig(packed_swap=True, min_numel=1))
        runtime.is_packable_template = mock.Mock(return_value=False)
        adapter = swap_optimizer_base.TorchNativeAdamWAdapter(optimizer, runtime.config, runtime)

        adapter._init_param_state(param, object(), optimizer.param_groups[0])

        self.assertEqual(optimizer.state[param]["step"].device, param.device)


class TestSwapOptimizerCheckpoint(unittest.TestCase):
    """Checkpoint roundtrips through the public wrapper."""

    @staticmethod
    def _wrapped(param, **config):
        """Build a swap optimizer over a fresh Adam optimizer with materialized state."""
        optimizer = torch.optim.Adam([param], lr=0.01)
        _materialize_adam_state(optimizer, param)
        return swap_optimizer(optimizer, SwapOptimizerConfig(**config))

    def test_state_dict_roundtrip(self):
        """Materialized Adam moments roundtrip through the wrapper checkpoint API."""
        param = torch.nn.Parameter(torch.ones(8))
        wrapped = self._wrapped(param, swap_times=2, min_numel=1)

        state_dict = wrapped.state_dict()
        new_param = torch.nn.Parameter(torch.ones(8))
        new_wrapped = self._wrapped(new_param, swap_times=2, min_numel=1)
        new_wrapped.load_state_dict(state_dict)

        self.assertIn("exp_avg", new_wrapped.state[new_param])
        self.assertTrue(torch.allclose(
            state_dict["state"][0]["exp_avg"],
            new_wrapped.state[new_param]["exp_avg"].detach().cpu(),
        ))

    def test_checkpoint_state_dict_never_exposes_released_storage(self):
        """A checkpoint taken after an offload still carries the real moments."""
        param = torch.nn.Parameter(torch.ones(8))
        wrapped = self._wrapped(param, swap_times=2, min_numel=1)
        wrapped.adapter.runtime.offload_initial_slots(wrapped.adapter.initial_slots())

        state_dict = wrapped.state_dict()

        for key in ("exp_avg", "exp_avg_sq"):
            self.assertGreater(state_dict["state"][0][key].untyped_storage().size(), 0)

    def test_load_state_dict_respects_configured_state_keys(self):
        """Only configured Adam buffers are stripped and re-registered as swap slots."""
        param = torch.nn.Parameter(torch.ones(8))
        wrapped = self._wrapped(param, swap_times=1, min_numel=0, state_keys=("exp_avg",))

        state_dict = wrapped.state_dict()
        new_param = torch.nn.Parameter(torch.ones(8))
        new_wrapped = self._wrapped(new_param, swap_times=1, min_numel=0, state_keys=("exp_avg",))
        new_wrapped.load_state_dict(state_dict)

        self.assertEqual([slot.name for slot in new_wrapped.adapter.all_slots()], ["exp_avg"])
        self.assertIn("exp_avg_sq", new_wrapped.state[new_param])

    def test_load_state_dict_casts_swappable_state_like_torch(self):
        """Loaded Adam buffers follow the current parameter dtype like native PyTorch."""
        param = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
        wrapped = self._wrapped(param, swap_times=1, min_numel=0)

        state_dict = wrapped.state_dict()
        new_param = torch.nn.Parameter(torch.ones(8, dtype=torch.float64))
        new_wrapped = self._wrapped(new_param, swap_times=1, min_numel=0)
        new_wrapped.load_state_dict(state_dict)

        self.assertEqual(new_wrapped.state[new_param]["exp_avg"].dtype, torch.float64)
        self.assertEqual(new_wrapped.state[new_param]["exp_avg_sq"].dtype, torch.float64)

    def test_load_state_dict_registers_reloaded_moments_as_swap_slots(self):
        """Reloaded moments become tracked slots instead of orphan optimizer state."""
        param = torch.nn.Parameter(torch.ones(8))
        wrapped = self._wrapped(param, swap_times=1, min_numel=0)

        state_dict = wrapped.state_dict()
        new_param = torch.nn.Parameter(torch.ones(8))
        new_wrapped = self._wrapped(new_param, swap_times=1, min_numel=0)
        new_wrapped.load_state_dict(state_dict)

        slots = tuple(new_wrapped.adapter.all_slots())
        self.assertEqual({slot.name for slot in slots}, {"exp_avg", "exp_avg_sq"})
        for slot in slots:
            # The stale slots from wrapping are replaced, so every registered
            # slot must own the optimizer state tensor that was just loaded.
            self.assertIs(new_wrapped.state[new_param][slot.name], slot.tensor)
            self.assertTrue(torch.count_nonzero(slot.checkpoint_tensor).item() > 0)


if __name__ == "__main__":
    unittest.main()
