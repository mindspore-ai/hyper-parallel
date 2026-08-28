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
"""Unit tests for swap optimizer public API and Torch backend."""

import importlib
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.core.optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.adamw import AdamW as HyperAdamW
from hyper_parallel.core.optimizer.swap_optimizer_base import (
    PipelineSwapRuntime,
    SwapSlot,
    UpdateUnit,
    _iter_unique_slot_objects,
)
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.platform.torch.swap_optimizer.swap_optimizer import TorchSwapOptimizer, TorchSwapRuntime
from hyper_parallel.platform.torch.swap_optimizer import adapters as torch_swap_adapters
from hyper_parallel.platform.torch.swap_optimizer.adapters import TorchAdamBaseAdapter

swap_optimizer_module = importlib.import_module("hyper_parallel.core.optimizer.swap_optimizer")


class _DummyConfig:
    swap_times = 1


def _materialize_adam_state(optimizer, param):
    """Populate deterministic Adam state without executing an optimizer update."""
    values = torch.arange(1, param.numel() + 1, dtype=param.dtype, device=param.device).view_as(param)
    state = optimizer.state[param]
    state["step"] = torch.tensor(1.0)
    state["exp_avg"] = values.clone()
    state["exp_avg_sq"] = values.square()
    return state


class _DummySwapRuntime(PipelineSwapRuntime):
    """Small runtime that records state transitions without a device backend."""

    def __init__(self):
        super().__init__(_DummyConfig())
        self.make_cpu_calls = 0
        self.to_device = []
        self.to_cpu = []

    def make_cpu_tensor(self, tensor):
        self.make_cpu_calls += 1
        return f"cpu:{tensor}"

    def copy_to_device(self, slot):
        self.to_device.append(slot.name)
        slot.state = "device"

    def wait_prefetch_slot(self, slot):
        slot.state = "device"

    def copy_to_cpu(self, slot):
        self.to_cpu.append(slot.name)
        slot.cpu_tensor = self.make_cpu_tensor(slot.tensor)
        slot.state = "d2h"

    def wait_offload_slot(self, slot):
        slot.state = "host"

    def current_stream(self):
        """Synchronous test runtime has no compute stream."""
        return None

    def new_stream(self):
        """Synchronous test runtime has no copy stream."""
        return None


class _DummyPackedRuntime(_DummySwapRuntime):
    """Record the backend-neutral two-staging-buffer schedule."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def supports_packed_pipeline(self, batches):
        del batches
        return True

    def begin_packed_step(self, batches):
        self.calls.append(f"begin:{len(batches)}")

    def enqueue_packed_prefetch(self, batch_index, staging_index):
        self.calls.append(f"prefetch:{batch_index}:{staging_index}")

    def wait_packed_prefetch(self, batch_index, staging_index):
        self.calls.append(f"wait_prefetch:{batch_index}:{staging_index}")

    def activate_packed_batch(self, batch_index, staging_index):
        self.calls.append(f"activate:{batch_index}:{staging_index}")

    def enqueue_packed_offload_prefetch(self, batch_index, next_index, staging_index):
        self.calls.append(f"chain:{batch_index}:{next_index}:{staging_index}")

    def wait_packed_offload(self, batch_index):
        self.calls.append(f"wait_offload:{batch_index}")

    def finish_packed_offload(self, batch_index):
        self.calls.append(f"finish_offload:{batch_index}")

    def end_packed_step(self):
        self.calls.append("end")


class TestSwapOptimizerConfig(unittest.TestCase):
    """Config validation should fail fast for unsupported modes."""

    @mock.patch.object(swap_optimizer_module, "get_platform")
    def test_packed_swap_default_depends_on_platform(self, mock_get_platform):
        """Packed staging defaults off on MindSpore and on for PyTorch."""
        for platform_type, expected in (
            (PlatformType.MINDSPORE, False),
            (PlatformType.PYTORCH, True),
        ):
            with self.subTest(platform_type=platform_type):
                mock_get_platform.return_value = SimpleNamespace(platform_type=platform_type)
                self.assertIs(SwapOptimizerConfig().packed_swap, expected)

    @mock.patch.object(swap_optimizer_module, "get_platform")
    def test_packed_swap_explicit_value_overrides_platform_default(self, mock_get_platform):
        """Callers can explicitly override either platform default."""
        mock_get_platform.return_value = SimpleNamespace(platform_type=PlatformType.MINDSPORE)
        self.assertTrue(SwapOptimizerConfig(packed_swap=True).packed_swap)
        mock_get_platform.return_value = SimpleNamespace(platform_type=PlatformType.PYTORCH)
        self.assertFalse(SwapOptimizerConfig(packed_swap=False).packed_swap)

    @mock.patch.object(swap_optimizer_module, "get_platform")
    def test_mindformers_adamw_defaults_to_packed_swap(self, mock_get_platform):
        """MindFormers AdamW opts into packed swap unless the caller overrides it."""
        optimizer_type = type(
            "AdamW",
            (),
            {"__module__": "mindformers.pynative.optimizer.adamw"},
        )
        optimizer = optimizer_type()
        backend_wrapper = mock.Mock(side_effect=lambda _optimizer, config: config)
        mock_get_platform.return_value = SimpleNamespace(
            platform_type=PlatformType.MINDSPORE,
            get_swap_optimizer=mock.Mock(return_value=backend_wrapper),
        )

        default_config = SwapOptimizerConfig()
        explicit_config = SwapOptimizerConfig(packed_swap=False)

        resolved_default = swap_optimizer(optimizer, default_config)
        resolved_explicit = swap_optimizer(optimizer, explicit_config)
        resolved_native = swap_optimizer(object(), default_config)

        self.assertTrue(resolved_default.packed_swap)
        self.assertFalse(resolved_explicit.packed_swap)
        self.assertFalse(resolved_native.packed_swap)

    def test_reject_invalid_state_key(self):
        """Only Adam/AdamW logical state keys are accepted."""
        with self.assertRaisesRegex(ValueError, "state_keys"):
            SwapOptimizerConfig(state_keys=("momentum_buffer",))

    def test_prefetch_batches_is_not_configurable(self):
        """The runtime always uses one-batch-ahead prefetch."""
        with self.assertRaisesRegex(TypeError, "prefetch_batches"):
            SwapOptimizerConfig(prefetch_batches=1)


class TestPipelineSwapRuntime(unittest.TestCase):
    """Common swap runtime state-machine behavior."""

    def test_unique_slot_objects_preserves_first_slot_for_each_tensor(self):
        """Tensor aliases are deduplicated by identity in input order."""
        shared_tensor = object()
        first = SwapSlot(name="first", tensor=shared_tensor)
        duplicate = SwapSlot(name="duplicate", tensor=shared_tensor)
        distinct = SwapSlot(name="distinct", tensor=object())

        slots = list(_iter_unique_slot_objects([first, duplicate, distinct]))

        self.assertEqual(slots, [first, distinct])

    def test_partition_balances_state_bytes_across_requested_batches(self):
        """Each partition tracks the average cost of the remaining batches."""
        runtime = _DummySwapRuntime()
        runtime.config.swap_times = 4
        unit_costs = [16, 16, 16, 16, 16, 1, 16, 16, 4, 2, 4, 2]
        units = [
            UpdateUnit(
                adapter_index=index,
                param=object(),
                grad=object(),
                slots=[SwapSlot(name=f"moment{index}", tensor=object(), storage_nbytes=cost)],
            )
            for index, cost in enumerate(unit_costs)
        ]

        batches = runtime.partition(units)

        batch_costs = [
            sum(slot.storage_nbytes for unit in batch for slot in unit.slots if slot.swappable)
            for batch in batches
        ]
        self.assertEqual(batch_costs, [32, 32, 33, 28])
        partitioned_units = [unit for batch in batches for unit in batch]
        self.assertEqual([id(unit) for unit in partitioned_units], [id(unit) for unit in units])

    def test_cpu_mirror_is_created_by_first_offload(self):
        """Cold slots stay device-resident until their first offload."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="moment", tensor="device-tensor", storage_nbytes=16, state="device")
        unit = UpdateUnit(
            adapter_index=0,
            param=object(),
            grad=object(),
            slots=[slot],
        )

        self.assertIsNone(slot.cpu_tensor)
        self.assertEqual(runtime.make_cpu_calls, 0)
        runtime.prefetch([unit])
        self.assertEqual(runtime.to_device, [])
        self.assertEqual(slot.state, "device")

        result = runtime.run_pipeline([[unit]], object(), lambda _batch, _ctx: "updated")

        self.assertEqual(result, ["updated"])
        self.assertEqual(slot.state, "host")
        self.assertEqual(slot.cpu_tensor, "cpu:device-tensor")
        self.assertEqual(runtime.to_device, [])
        self.assertEqual(runtime.to_cpu, ["moment"])

        runtime.prefetch([unit])
        self.assertEqual(slot.state, "h2d")
        runtime.wait_prefetch([unit])
        self.assertEqual(slot.state, "device")
        self.assertEqual(runtime.to_device, ["moment"])

    def test_initial_offload_makes_first_prefetch_load_state(self):
        """Initial offload lets the first optimizer update prefetch host-resident slots."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="moment", tensor="device-tensor", storage_nbytes=16, state="device")
        unit = UpdateUnit(
            adapter_index=0,
            param=object(),
            grad=object(),
            slots=[slot],
        )

        runtime.offload_initial_slots([slot])

        self.assertEqual(slot.state, "host")
        self.assertEqual(slot.cpu_tensor, "cpu:device-tensor")
        self.assertEqual(runtime.to_cpu, ["moment"])

        runtime.prefetch([unit])
        self.assertEqual(slot.state, "h2d")
        runtime.wait_prefetch([unit])
        self.assertEqual(slot.state, "device")
        self.assertEqual(runtime.to_device, ["moment"])

    def test_wait_offload_orders_release_after_copy_event(self):
        """Device storage release is stream-ordered after the D2H copy event."""
        runtime = _DummySwapRuntime()
        compute_stream = object()
        copy_event = object()
        slot = SwapSlot(name="moment", tensor=object(), state="d2h", event=copy_event)
        calls = []

        runtime.current_stream = mock.Mock(return_value=compute_stream)
        runtime.wait_event = mock.Mock(
            side_effect=lambda event, stream: calls.append(("wait", event, stream))
        )
        runtime.wait_offload_slot = mock.Mock(
            side_effect=lambda current_slot: calls.append(("release", current_slot))
        )

        runtime._wait_offload_slots([slot])

        self.assertEqual(calls, [
            ("wait", copy_event, compute_stream),
            ("release", slot),
        ])
        self.assertIs(slot.event, copy_event)

    def test_synchronize_cpu_mirrors_waits_before_checkpoint_read(self):
        """Checkpoint host reads wait after stream-ordered D2H and release."""
        runtime = _DummySwapRuntime()
        compute_stream = object()
        copy_event = object()
        checkpoint_event = object()
        slot = SwapSlot(
            name="moment",
            tensor=object(),
            cpu_tensor=object(),
            state="host",
            event=copy_event,
        )
        calls = []

        runtime.current_stream = mock.Mock(return_value=compute_stream)
        runtime.record_event = mock.Mock(
            side_effect=lambda stream: calls.append(("record", stream)) or checkpoint_event
        )
        runtime.wait_event = mock.Mock(
            side_effect=lambda event, stream: calls.append(("wait", event, stream))
        )

        runtime.synchronize_cpu_mirrors([slot])

        self.assertEqual(calls, [
            ("wait", copy_event, compute_stream),
            ("record", compute_stream),
            ("wait", checkpoint_event, None),
        ])
        self.assertIsNone(slot.event)

    def test_pipeline_prefetches_next_batch_and_releases_two_batches_back(self):
        """Prefetch runs one batch ahead and waits old offload before widening the window."""
        runtime = _DummySwapRuntime()
        slots = [
            SwapSlot(
                name=f"moment{index}",
                tensor=f"device-tensor{index}",
                cpu_tensor=f"cpu:device-tensor{index}",
                storage_nbytes=16,
                state="host",
            )
            for index in range(3)
        ]
        units = [
            UpdateUnit(
                adapter_index=index,
                param=object(),
                grad=object(),
                slots=[slot],
            )
            for index, slot in enumerate(slots)
        ]
        calls = []

        def _recording_step(batch, context):
            del context
            calls.append(f"update:{batch[0].slots[0].name}")

        def _copy_to_device(slot):
            calls.append(f"prefetch:{slot.name}")
            slot.state = "h2d"

        def _wait_prefetch(slot):
            calls.append(f"wait_prefetch:{slot.name}")
            slot.state = "device"

        def _copy_to_cpu(slot):
            calls.append(f"offload:{slot.name}")
            slot.state = "d2h"

        def _wait_offload(slot):
            calls.append(f"wait_offload:{slot.name}")
            slot.state = "host"

        runtime.copy_to_device = _copy_to_device
        runtime.wait_prefetch_slot = _wait_prefetch
        runtime.copy_to_cpu = _copy_to_cpu
        runtime.wait_offload_slot = _wait_offload

        runtime.run_pipeline([[unit] for unit in units], object(), _recording_step)

        self.assertEqual(calls, [
            "prefetch:moment0",
            "wait_prefetch:moment0",
            "prefetch:moment1",
            "update:moment0",
            "offload:moment0",
            "wait_prefetch:moment1",
            "wait_offload:moment0",
            "prefetch:moment2",
            "update:moment1",
            "offload:moment1",
            "wait_prefetch:moment2",
            "wait_offload:moment1",
            "update:moment2",
            "offload:moment2",
            "wait_offload:moment2",
        ])

    def test_packed_pipeline_overlaps_copy_chain_with_other_arena_update(self):
        """D2H batch i and H2D batch i+2 share a chain while the other arena updates."""
        runtime = _DummyPackedRuntime()
        units = [
            UpdateUnit(
                adapter_index=index,
                param=object(),
                grad=object(),
                slots=[],
            )
            for index in range(3)
        ]

        def _recording_step(batch, context):
            del context
            runtime.calls.append(f"update:{batch[0].adapter_index}")

        runtime.run_pipeline([[unit] for unit in units], object(), _recording_step)

        self.assertEqual(runtime.calls, [
            "begin:3",
            "prefetch:0:0",
            "prefetch:1:1",
            "wait_prefetch:0:0",
            "activate:0:0",
            "update:0",
            "chain:0:2:0",
            "wait_prefetch:1:1",
            "activate:1:1",
            "update:1",
            "chain:1:None:1",
            "wait_prefetch:2:0",
            "wait_offload:0",
            "finish_offload:0",
            "activate:2:0",
            "update:2",
            "chain:2:None:0",
            "wait_offload:1",
            "finish_offload:1",
            "wait_offload:2",
            "finish_offload:2",
            "end",
        ])

    def test_packed_results_can_be_released_before_staging_teardown(self):
        """Backend hooks may drop update stubs before staging storage is released."""

        class _ReleasingRuntime(_DummyPackedRuntime):
            def release_packed_step_results(self, results):
                self.calls.append("release_results")
                results.clear()

            def end_packed_step(self):
                self.calls.append(f"result_alive_at_end:{result_ref() is not None}")
                super().end_packed_step()

        class _Result:
            pass

        runtime = _ReleasingRuntime()
        unit = UpdateUnit(0, object(), object(), [])
        placeholder_result = _Result()
        result_ref = weakref.ref(placeholder_result)
        del placeholder_result

        def _step(batch, context):
            nonlocal result_ref
            del batch, context
            result = _Result()
            result_ref = weakref.ref(result)
            return result

        result = runtime.run_pipeline([[unit]], object(), _step)

        self.assertEqual(result, [])
        self.assertIn("release_results", runtime.calls)
        self.assertIn("result_alive_at_end:False", runtime.calls)


class TestTorchSwapOptimizer(unittest.TestCase):
    """Torch backend numerical and checkpoint smoke coverage."""

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

    def test_native_adamw_accepts_fused_and_forwards_it_to_functional(self):
        """Native AdamW keeps fused execution when its state is updated through swap slots."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=0.01, fused=True)
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=False, min_numel=1024))
        adapter = torch_swap_adapters.TorchNativeAdamWAdapter(optimizer, runtime.config, runtime)
        state = optimizer.state[param]
        state["step"] = torch.zeros(())
        state["exp_avg"] = torch.zeros_like(param)
        state["exp_avg_sq"] = torch.zeros_like(param)
        unit = UpdateUnit(0, param, torch.ones_like(param), [])

        adapter.validate()
        with mock.patch.object(torch.optim._functional, "adamw") as functional_adamw:
            adapter.step_batch([unit], {})

        self.assertTrue(functional_adamw.call_args.kwargs["fused"])

    def test_fused_adamw_selects_native_adamw_adapter(self):
        """AdamW must not be captured by the Adam adapter through inheritance."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=0.01, fused=True)

        wrapped = TorchSwapOptimizer(optimizer, SwapOptimizerConfig(packed_swap=False))

        self.assertIsInstance(wrapped.adapter, torch_swap_adapters.TorchNativeAdamWAdapter)

    def test_packed_fused_adamw_initializes_step_on_parameter_device(self):
        """Packed fused AdamW keeps its scalar step beside parameters for the fused kernel."""
        param = torch.nn.Parameter(torch.ones(8, device="meta"))
        optimizer = torch.optim.AdamW([param], lr=0.01, fused=True)
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True, min_numel=1))
        runtime.is_packable_template = mock.Mock(return_value=False)
        adapter = torch_swap_adapters.TorchNativeAdamWAdapter(optimizer, runtime.config, runtime)

        adapter._init_param_state(param, object(), optimizer.param_groups[0])

        self.assertEqual(optimizer.state[param]["step"].device, param.device)

    def test_packed_swap_config_controls_candidate_path(self):
        """Parameter-specific packed eligibility is deferred to slot and step checks."""
        packed_runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True))
        legacy_runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=False))

        self.assertTrue(packed_runtime.packed_enabled)
        self.assertFalse(legacy_runtime.packed_enabled)

    def test_dtensor_local_shard_is_packable(self):
        """DTensor-like optimizer state uses its local shard for packed metadata and storage."""

        class _DTensorLike:
            def __init__(self, local_tensor):
                self.local_tensor = local_tensor

            def to_local(self):
                return self.local_tensor

        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True, min_numel=1))
        param = _DTensorLike(torch.empty(8, device="meta"))
        adapter = object.__new__(TorchAdamBaseAdapter)
        adapter.runtime = runtime
        adapter.config = SimpleNamespace(min_numel=1)
        adapter.optimizer = SimpleNamespace(state={param: {}})

        slot = adapter._make_slot("exp_avg", None, template=param)

        self.assertTrue(slot.swappable)
        self.assertTrue(slot.packed)
        self.assertEqual(slot.shape, (8,))
        self.assertEqual(slot.device, param.to_local().device)

    def test_packed_dtensor_lazy_state_starts_pending_without_zeros_like(self):
        """Lazy DTensor moments are zeroed on the packed host buffer, not the device shard."""

        class _DTensorLike:
            device = torch.device("meta")

            def __init__(self, local_tensor):
                self.local_tensor = local_tensor

            def to_local(self):
                return self.local_tensor

        param = _DTensorLike(torch.empty(8, device="meta"))
        optimizer = SimpleNamespace(state={param: {}}, param_groups=[{"params": [param]}])
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True, min_numel=1))
        adapter = object.__new__(TorchAdamBaseAdapter)
        adapter.optimizer = optimizer
        adapter.runtime = runtime
        adapter.config = runtime.config
        adapter._slots = {}
        adapter.is_hyper_adamw = True

        with mock.patch.object(torch, "zeros_like", wraps=torch.zeros_like) as zeros_like:
            adapter._init_param_state(param, object(), {"amsgrad": False})
            slots = tuple(adapter.all_slots())
            self.assertEqual({slot.name for slot in slots}, {"exp_avg", "exp_avg_sq"})
            self.assertTrue(all(slot.state == "pending" for slot in slots))
            self.assertTrue(all(slot.tensor is None for slot in slots))
            self.assertTrue(all(slot.logical_tensor is None for slot in slots))

            original_empty = torch.empty

            def _cpu_empty(*args, **kwargs):
                kwargs.pop("pin_memory", None)
                return original_empty(*args, **kwargs)

            with mock.patch("torch.empty", side_effect=_cpu_empty):
                runtime.prepare_packed_host(slots)
                self.assertNotIn("exp_avg", optimizer.state[param])
                self.assertNotIn("exp_avg_sq", optimizer.state[param])
                adapter.publish_packed_state()

            self.assertTrue(all(slot.state == "host" for slot in slots))
            self.assertTrue(all(torch.count_nonzero(slot.cpu_tensor) == 0 for slot in slots))
            self.assertIs(optimizer.state[param]["exp_avg"], slots[0].cpu_tensor)
            self.assertIs(optimizer.state[param]["exp_avg_sq"], slots[1].cpu_tensor)
            zeros_like.assert_not_called()

    def test_supports_packed_pipeline_makes_final_device_decision(self):
        """The final gate rejects otherwise complete packed slots spanning local devices."""
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True))
        runtime._host_buffers = {torch.float32: torch.empty(2)}

        def _unit(index, device, logical_tensor=None):
            slot = SwapSlot(
                name="exp_avg",
                tensor=torch.empty(1),
                cpu_tensor=torch.empty(1),
                swappable=True,
                packed=True,
                dtype=torch.float32,
                device=torch.device(device),
                logical_tensor=logical_tensor,
            )
            return UpdateUnit(index, object(), object(), [slot])

        same_device = [[_unit(0, "meta:0"), _unit(1, "meta:0")]]
        mixed_devices = [[_unit(0, "meta:0"), _unit(1, "meta:1")]]

        self.assertTrue(runtime.supports_packed_pipeline(same_device))
        self.assertFalse(runtime.supports_packed_pipeline(mixed_devices))

        mixed_dtensor_devices = [[
            _unit(0, "meta:0", logical_tensor=object()),
            _unit(1, "meta:1", logical_tensor=object()),
        ]]
        with self.assertRaisesRegex(RuntimeError, "cannot fall back to per-tensor swap"):
            runtime.supports_packed_pipeline(mixed_dtensor_devices)

    def test_wait_packed_offload_uses_compute_stream_dependency(self):
        """Intermediate packed offload waits do not synchronize the CPU."""
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True))
        offload_event = object()
        compute_stream = object()
        runtime._packed_offload_events = {3: offload_event}
        runtime.current_stream = mock.Mock(return_value=compute_stream)
        runtime.wait_event = mock.Mock()

        runtime.wait_packed_offload(3)

        runtime.wait_event.assert_called_once_with(offload_event, compute_stream)

    def test_end_packed_step_cpu_synchronizes_only_tail_event(self):
        """One tail event drains all work serialized on the packed copy stream."""
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True))
        earlier_event = object()
        tail_event = object()
        runtime._packed_ready_events = {0: earlier_event}
        runtime._packed_offload_events = {0: earlier_event, 1: tail_event}
        runtime._packed_tail_event = tail_event
        runtime.wait_event = mock.Mock()

        runtime.end_packed_step()

        runtime.wait_event.assert_called_once_with(tail_event, None)
        self.assertIsNone(runtime._packed_tail_event)

    def test_torch_update_units_use_parameter_group_as_adapter_index(self):
        """Torch units use the adapter index to retain parameter-group membership."""
        params = [torch.nn.Parameter(torch.ones(4)) for _ in range(3)]
        optimizer = torch.optim.Adam([
            {"params": params[:2], "lr": 0.01},
            {"params": params[2:], "lr": 0.02},
        ])
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=False, min_numel=1024))
        adapter = TorchAdamBaseAdapter(optimizer, runtime.config, runtime)
        for param in params:
            param.grad = torch.ones_like(param)

        units = adapter.prepare_step()["units"]

        self.assertEqual([unit.adapter_index for unit in units], [0, 0, 1])
        self.assertEqual([id(unit.param) for unit in units], [id(param) for param in params])

    def test_per_tensor_lazy_state_starts_from_zero_cpu_mirrors(self):
        """Swappable lazy moments avoid zero-initializing every device tensor."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=False, min_numel=1))
        runtime.is_swappable_tensor = mock.Mock(return_value=True)
        runtime.make_zero_cpu_tensor_like = mock.Mock(return_value=torch.zeros_like(param, device="cpu"))
        runtime.release_device_storage = mock.Mock()
        adapter = TorchAdamBaseAdapter(optimizer, runtime.config, runtime)
        param.grad = torch.ones_like(param)

        with mock.patch.object(torch, "zeros_like", wraps=torch.zeros_like) as zeros_like:
            adapter.prepare_step()

        slots = tuple(adapter.all_slots())
        self.assertEqual({slot.name for slot in slots}, {"exp_avg", "exp_avg_sq"})
        self.assertTrue(all(slot.state == "host" for slot in slots))
        self.assertTrue(all(torch.count_nonzero(slot.cpu_tensor) == 0 for slot in slots))
        self.assertEqual(runtime.make_zero_cpu_tensor_like.call_count, 2)
        self.assertEqual(runtime.release_device_storage.call_count, 2)
        zeros_like.assert_not_called()

    def test_packed_runtime_restores_cpu_slots_and_releases_two_staging_arenas(self):
        """Packed slot views roundtrip through two raw staging storages."""
        runtime = TorchSwapRuntime(_DummyConfig())
        runtime._packed_enabled = True
        runtime._get_copy_stream = lambda: None
        runtime.current_stream = lambda: None
        runtime.record_event = lambda _stream=None: object()
        runtime.wait_event = lambda _event, _stream=None: None

        sources = [
            torch.arange(4, dtype=torch.float32),
            torch.arange(4, dtype=torch.bfloat16),
            torch.arange(4, dtype=torch.float32) + 10,
            torch.arange(4, dtype=torch.bfloat16) + 10,
        ]
        slots = []
        for source in sources:
            slot = SwapSlot(
                name="exp_avg",
                tensor=source,
                swappable=True,
                packed=True,
            )
            runtime.populate_slot_metadata(slot, source)
            slots.append(slot)

        original_empty = torch.empty

        def _cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return original_empty(*args, **kwargs)

        copy_counts = {"h2d": 0, "d2h": 0}
        original_h2d = runtime._copy_packed_to_device
        original_d2h = runtime._copy_packed_to_host

        def _record_h2d(batch_index, staging_index):
            copy_counts["h2d"] += len(runtime._packed_batch_plans[batch_index].regions)
            return original_h2d(batch_index, staging_index)

        def _record_d2h(batch_index, staging_index):
            copy_counts["d2h"] += len(runtime._packed_batch_plans[batch_index].regions)
            return original_d2h(batch_index, staging_index)

        runtime._copy_packed_to_device = _record_h2d
        runtime._copy_packed_to_host = _record_d2h
        units = [
            UpdateUnit(
                adapter_index=index,
                param=object(),
                grad=object(),
                slots=[slot],
            )
            for index, slot in enumerate(slots)
        ]
        batches = [units[:2], units[2:]]

        def _update(batch, context):
            del context
            for unit in batch:
                unit.slots[0].tensor.add_(1)

        with mock.patch("torch.empty", side_effect=_cpu_empty):
            runtime.prepare_packed_host(slots)
            runtime.run_pipeline(batches, object(), _update)

        for slot, source in zip(slots, sources):
            self.assertIs(slot.tensor, slot.cpu_tensor)
            self.assertTrue(torch.equal(slot.cpu_tensor, source + 1))
        self.assertEqual(copy_counts, {"h2d": 4, "d2h": 4})
        self.assertEqual(len(runtime._staging_arenas), 2)
        self.assertTrue(all(arena.raw_buffer.untyped_storage().size() == 0 for arena in runtime._staging_arenas))

    def test_prepare_packed_step_retains_inactive_materialized_state(self):
        """Packed unit preparation retains state whose parameter has no gradient."""
        params = [torch.nn.Parameter(torch.ones(8)), torch.nn.Parameter(torch.ones(8))]
        optimizer = torch.optim.AdamW(params, lr=0.01)
        for param in params:
            _materialize_adam_state(optimizer, param)
        params[0].grad = torch.ones_like(params[0])

        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True, min_numel=1))
        runtime.is_swappable_tensor = mock.Mock(return_value=True)
        runtime.prepare_packed_host = mock.Mock()
        adapter = TorchAdamBaseAdapter(optimizer, runtime.config, runtime)

        units = adapter.prepare_step()["units"]

        self.assertEqual([id(unit.param) for unit in units], [id(param) for param in params])
        self.assertIsNotNone(units[0].grad)
        self.assertIsNone(units[1].grad)
        self.assertTrue(all(slot.packed for unit in units for slot in unit.slots))
        runtime.prepare_packed_host.assert_called_once()

    def test_torch_adapter_prefers_active_slot_tensor_and_falls_back_to_state(self):
        """Functional updates use active staging views without requiring state rebinding."""
        state_tensor = torch.zeros(4)
        device_tensor = torch.ones(4)
        slot = SwapSlot(name="exp_avg", tensor=device_tensor, swappable=True, state="device")
        unit = UpdateUnit(0, object(), object(), [slot])

        self.assertIs(TorchAdamBaseAdapter._slot_tensor(unit, "exp_avg", state_tensor), device_tensor)

        slot.state = "host"
        self.assertIs(TorchAdamBaseAdapter._slot_tensor(unit, "exp_avg", state_tensor), state_tensor)
        self.assertIs(TorchAdamBaseAdapter._slot_tensor(unit, "exp_avg_sq", state_tensor), state_tensor)

    def test_torch_step_batch_reads_active_slots_without_rebinding_optimizer_state(self):
        """Torch functional Adam receives staging views while public state keeps CPU mirrors."""
        param = torch.nn.Parameter(torch.ones(4))
        optimizer = torch.optim.Adam([param], lr=0.01)
        state = optimizer.state[param]
        state["step"] = torch.zeros(())
        cpu_exp_avg = torch.zeros(4)
        cpu_exp_avg_sq = torch.zeros(4)
        state["exp_avg"] = cpu_exp_avg
        state["exp_avg_sq"] = cpu_exp_avg_sq
        active_exp_avg = torch.ones(4)
        active_exp_avg_sq = torch.full((4,), 2.0)
        slots = [
            SwapSlot(name="exp_avg", tensor=active_exp_avg, swappable=True, state="device"),
            SwapSlot(name="exp_avg_sq", tensor=active_exp_avg_sq, swappable=True, state="device"),
        ]
        unit = UpdateUnit(0, param, torch.ones_like(param), slots)
        runtime = TorchSwapRuntime(SwapOptimizerConfig(packed_swap=True, min_numel=1))
        adapter = TorchAdamBaseAdapter(optimizer, runtime.config, runtime)

        with mock.patch.object(torch.optim._functional, "adam") as functional_adam:
            adapter.step_batch([unit], {})

        call_args = functional_adam.call_args.args
        self.assertIs(call_args[2][0], active_exp_avg)
        self.assertIs(call_args[3][0], active_exp_avg_sq)
        self.assertIs(optimizer.state[param]["exp_avg"], cpu_exp_avg)
        self.assertIs(optimizer.state[param]["exp_avg_sq"], cpu_exp_avg_sq)

    def test_preinitialized_adamw_state_is_registered_on_wrap(self):
        """AdamW states materialized before wrapping are registered immediately."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=0.01)
        _materialize_adam_state(optimizer, param)

        swapped = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=2, min_numel=1))

        slots = tuple(swapped.adapter.all_slots())
        self.assertEqual({slot.name for slot in slots}, {"exp_avg", "exp_avg_sq"})
        self.assertEqual({id(slot.tensor) for slot in slots}, {
            id(optimizer.state[param]["exp_avg"]),
            id(optimizer.state[param]["exp_avg_sq"]),
        })

    def test_state_dict_roundtrip(self):
        """Materialized Adam moments roundtrip through the wrapper checkpoint API."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        _materialize_adam_state(optimizer, param)
        opt = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=2, min_numel=1))
        state_dict = opt.state_dict()

        new_param = torch.nn.Parameter(torch.ones(8))
        new_opt = swap_optimizer(
            torch.optim.Adam([new_param], lr=0.01),
            SwapOptimizerConfig(swap_times=2, min_numel=1),
        )
        new_opt.load_state_dict(state_dict)
        self.assertIn("exp_avg", new_opt.state[new_param])
        self.assertTrue(torch.allclose(
            state_dict["state"][0]["exp_avg"],
            new_opt.state[new_param]["exp_avg"].detach().cpu(),
        ))

    def test_load_state_dict_respects_configured_state_keys(self):
        """Only configured Adam buffers are stripped and re-registered as swap slots."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        _materialize_adam_state(optimizer, param)
        opt = swap_optimizer(
            optimizer,
            SwapOptimizerConfig(swap_times=1, min_numel=0, state_keys=("exp_avg",)),
        )
        state_dict = opt.state_dict()

        new_param = torch.nn.Parameter(torch.ones(8))
        new_opt = swap_optimizer(
            torch.optim.Adam([new_param], lr=0.01),
            SwapOptimizerConfig(swap_times=1, min_numel=0, state_keys=("exp_avg",)),
        )
        new_opt.load_state_dict(state_dict)

        self.assertEqual([slot.name for slot in new_opt.adapter.all_slots()], ["exp_avg"])
        self.assertIn("exp_avg_sq", new_opt.state[new_param])

    def test_load_state_dict_casts_swappable_state_like_torch(self):
        """Loaded Adam buffers follow the current parameter dtype like native PyTorch."""
        param = torch.nn.Parameter(torch.ones(8, dtype=torch.float32))
        optimizer = torch.optim.Adam([param], lr=0.01)
        _materialize_adam_state(optimizer, param)
        opt = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=1, min_numel=0))
        state_dict = opt.state_dict()

        new_param = torch.nn.Parameter(torch.ones(8, dtype=torch.float64))
        new_opt = swap_optimizer(
            torch.optim.Adam([new_param], lr=0.01),
            SwapOptimizerConfig(swap_times=1, min_numel=0),
        )
        new_opt.load_state_dict(state_dict)

        self.assertEqual(new_opt.state[new_param]["exp_avg"].dtype, torch.float64)
        self.assertEqual(new_opt.state[new_param]["exp_avg_sq"].dtype, torch.float64)

    def test_load_packed_dtensor_state_retains_independent_logical_wrapper(self):
        """Checkpoint restore keeps a DTensor state wrapper for later packed activation."""

        class _Param:
            dtype = torch.float32

            @staticmethod
            def is_floating_point():
                return True

        param = _Param()
        logical_state = object()
        released = []
        prepared = []
        runtime = SimpleNamespace(
            packed_enabled=True,
            is_packable_template=lambda tensor, _min_numel: tensor is param,
            is_distributed_tensor=lambda tensor: tensor is param or tensor is logical_state,
            is_swappable_tensor=lambda tensor, _min_numel: tensor is logical_state,
            populate_slot_metadata=lambda slot, _template: (
                setattr(slot, "dtype", torch.float32),
                setattr(slot, "numel", 4),
                setattr(slot, "storage_nbytes", 16),
            ),
            release_device_storage=released.append,
            prepare_packed_host=prepared.extend,
        )
        optimizer = SimpleNamespace(
            param_groups=[{"params": [param]}],
            state={param: {}},
        )
        adapter = object.__new__(TorchAdamBaseAdapter)
        adapter.optimizer = optimizer
        adapter.runtime = runtime
        adapter.config = SimpleNamespace(min_numel=1, state_keys=None)
        adapter._slots = {}
        fake_torch = SimpleNamespace(
            Tensor=torch.Tensor,
            preserve_format=torch.preserve_format,
            zeros_like=mock.Mock(return_value=logical_state),
        )

        with mock.patch.object(torch_swap_adapters, "torch", fake_torch):
            adapter.load_swappable_state(
                {"param_groups": [{"params": [0]}]},
                {0: {"exp_avg": torch.arange(4, dtype=torch.float32)}},
            )

        slot = adapter._slots[(id(param), "exp_avg")]
        self.assertIs(slot.logical_tensor, logical_state)
        self.assertIs(optimizer.state[param]["exp_avg"], slot.cpu_tensor)
        self.assertIsNot(slot.logical_tensor, param)
        self.assertEqual(released, [slot])
        self.assertEqual(prepared, [slot])

    def test_hyper_adamw_prepare_step_increments_group_once(self):
        """Hyper AdamW unit preparation advances the group counter once."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = HyperAdamW([param], lr=0.01)
        swapped = swap_optimizer(optimizer, SwapOptimizerConfig(swap_times=4, min_numel=1))
        param.grad = torch.ones_like(param)

        units = swapped.adapter.prepare_step()["units"]

        self.assertEqual(optimizer.param_groups[0]["step"], 1)
        self.assertEqual([id(unit.param) for unit in units], [id(param)])

    def test_unsupported_optimizer_fails_fast(self):
        """Unsupported optimizers are not silently wrapped."""
        param = torch.nn.Parameter(torch.ones(8))
        with self.assertRaisesRegex(ValueError, "only supports"):
            swap_optimizer(torch.optim.SGD([param], lr=0.01), SwapOptimizerConfig())


if __name__ == "__main__":
    unittest.main()
