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
"""Unit tests for the swap optimizer runtime and adapter internals."""

import contextlib
import os
import types
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.core.optimizer.swap_optimizer_base import (
    OptimizerSwapAdapter,
    PipelineSwapRuntime,
    SwapSlot,
    TorchNativeAdamAdapter,
    TorchNativeAdamWAdapter,
    UpdateUnit,
    _PackedBatchPlan,
    _PackedBatchRegion,
    _StagingArena,
    _iter_unique_events,
    _iter_unique_slot_objects,
    _iter_unique_slots,
    validate_state_keys,
)

# Adam state keys created by the lazy per-tensor initialization path.
ADAM_STATE_KEYS = ("exp_avg", "exp_avg_sq")
# Every logical key the adapter knows about, including the optional amsgrad moment.
ALL_STATE_KEYS = ADAM_STATE_KEYS + ("max_exp_avg_sq",)

# Real Torch allocators, captured before the tests patch them for CPU-only runs.
_TORCH_EMPTY = torch.empty
_TORCH_EMPTY_LIKE = torch.empty_like
_TORCH_ZEROS = torch.zeros
_TORCH_ZEROS_LIKE = torch.zeros_like


def _unpinned(allocator):
    """Wrap ``allocator`` so the ``pin_memory`` request is dropped."""
    def _allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return allocator(*args, **kwargs)

    return _allocate


# ``pin_memory`` needs an accelerator allocator. These replacements let a
# CPU-only UT host exercise the same pageable-memory fallback a device run takes.
_unpinned_empty = _unpinned(_TORCH_EMPTY)
_unpinned_empty_like = _unpinned(_TORCH_EMPTY_LIKE)
_unpinned_zeros = _unpinned(_TORCH_ZEROS)
_unpinned_zeros_like = _unpinned(_TORCH_ZEROS_LIKE)


def _patch_allocators():
    """Patch every Torch allocator the runtime may call."""
    patchers = [
        mock.patch.object(torch, "empty", _unpinned_empty),
        mock.patch.object(torch, "empty_like", _unpinned_empty_like),
        mock.patch.object(torch, "zeros", _unpinned_zeros),
        mock.patch.object(torch, "zeros_like", _unpinned_zeros_like),
    ]
    for patcher in patchers:
        patcher.start()
    return patchers


class _LogicalTensor:
    """Stand-in for a DTensor wrapper that carries a local shard."""

    def __init__(self, device, data):
        self.device = device
        self.data = data
        self._local_tensor = None


class _FakeDevice:
    """Device descriptor that reports a type other than ``cpu``.

    The bundled ``synchronize`` hook keeps
    :meth:`PipelineSwapRuntime.begin_packed_step` off a real accelerator runtime.
    """

    def __init__(self, device_type="cuda"):
        self.type = device_type

    def synchronize(self, *_args, **_kwargs):
        """Accept the device-wide synchronization that opens a packed step."""

    def __str__(self):
        return self.type


class _DeviceLikeTensor:
    """Proxy over a real tensor that claims to live on a device.

    Storage resizing and H2D/D2H copies are device-only code paths. A CPU-only
    test host cannot allocate such a tensor, so this proxy forwards storage,
    metadata and element-wise operations to a real tensor while reporting a
    non-CPU device to the runtime.
    """

    def __init__(self, tensor, device=None):
        self._tensor = tensor
        self.device = _FakeDevice() if device is None else device

    @property
    def shape(self):
        """Forward the tensor shape."""
        return self._tensor.shape

    @property
    def dtype(self):
        """Forward the tensor dtype."""
        return self._tensor.dtype

    def numel(self):
        """Forward the element count."""
        return self._tensor.numel()

    def element_size(self):
        """Forward the element size."""
        return self._tensor.element_size()

    def untyped_storage(self):
        """Forward the underlying storage so it can still be resized."""
        return self._tensor.untyped_storage()

    def is_floating_point(self):
        """Forward the dtype check."""
        return self._tensor.is_floating_point()

    def is_contiguous(self):
        """Forward the layout check."""
        return self._tensor.is_contiguous()

    def detach(self):
        """Forward detachment onto the wrapped tensor."""
        return _DeviceLikeTensor(self._tensor.detach(), self.device)

    def clone(self):
        """Forward cloning onto the wrapped tensor."""
        return _DeviceLikeTensor(self._tensor.clone(), self.device)

    def reshape(self, *shape):
        """Forward reshape onto the wrapped tensor."""
        return _DeviceLikeTensor(self._tensor.reshape(*shape), self.device)

    def to(self, *args, **kwargs):
        """Forward a dtype/device cast onto the wrapped tensor."""
        return _DeviceLikeTensor(self._tensor.to(*args, **kwargs), self.device)

    def view(self, *shape):
        """Forward a view onto the wrapped tensor."""
        return _DeviceLikeTensor(self._tensor.view(*shape), self.device)

    def zero_(self):
        """Zero the wrapped tensor."""
        self._tensor.zero_()
        return self

    def copy_(self, source, **kwargs):
        """Forward an in-place copy onto the wrapped tensor."""
        self._tensor.copy_(getattr(source, "_tensor", source), **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def _fake_device_slot(**overrides):
    """Build a slot whose metadata describes a device-resident state tensor."""
    values = {"dtype": torch.float32, "numel": 4, "shape": (4,), "device": _FakeDevice()}
    values.update(overrides)
    slot = SwapSlot(
        name=overrides.pop("name", "exp_avg"),
        tensor=None,
        swappable=True,
        packed=True,
        state="pending",
    )
    for key, value in values.items():
        setattr(slot, key, value)
    return slot


def _like_device_runtime(**config):
    """Build a runtime whose optimizer state pretends to live on a device."""
    values = {"swap_times": 1, "packed_swap": True, "min_numel": 0, "state_keys": None}
    values.update(config)
    runtime = PipelineSwapRuntime(_DummyConfig(**values))
    runtime._storage_tensor = staticmethod(lambda tensor: tensor)
    return runtime


class _DummyConfig:
    """Config stub covering every attribute the runtime reads."""

    def __init__(self, swap_times=1, packed_swap=False, min_numel=0, state_keys=None):
        self.swap_times = swap_times
        self.packed_swap = packed_swap
        self.min_numel = min_numel
        self.state_keys = state_keys


def _unit(slot, index=0, param=None, grad=None):
    """Build one update unit from a slot, a mock slot, or a list of either."""
    slots = list(slot) if isinstance(slot, (list, tuple)) else [slot]
    return UpdateUnit(
        adapter_index=index,
        param=object() if param is None else param,
        grad=object() if grad is None else grad,
        slots=slots,
    )


class _FakeScheduleTensor(torch.Tensor):
    """Real CPU tensor whose reported device, shape and storage are overridden.

    Swappability is decided before any storage is allocated, so a CPU-only UT
    host needs tensors that merely *claim* to live on a device. Subclassing a
    real tensor keeps ``isinstance`` checks and ``untyped_storage`` working.
    """

    @staticmethod
    def make(*, numel=8, element_size=4, storage_size=32, storage_error=None, device_type="cuda",
             floating=True, contiguous=True, sparse=False):
        """Build a fake tensor exposing the requested metadata."""
        tensor = _TORCH_ZEROS(numel, dtype=torch.float32 if floating else torch.int32)
        tensor = tensor.as_subclass(_FakeScheduleTensor)
        tensor.fake_numel = numel
        tensor.fake_element_size = element_size
        tensor.fake_storage_size = storage_size
        tensor.fake_storage_error = storage_error
        tensor.fake_device = SimpleNamespace(type=device_type)
        tensor.fake_contiguous = contiguous
        tensor.fake_sparse = sparse
        return tensor

    @property
    def device(self):
        """Report the simulated device."""
        return self.fake_device

    @property
    def is_sparse(self):
        """Report the simulated sparsity."""
        return self.fake_sparse

    def numel(self):
        """Report the simulated element count."""
        return self.fake_numel

    def element_size(self):
        """Report the simulated element size."""
        return self.fake_element_size

    def is_contiguous(self):
        """Report the simulated contiguity."""
        return self.fake_contiguous

    def untyped_storage(self):
        """Report the simulated storage."""
        if self.fake_storage_error is not None:
            raise self.fake_storage_error
        storage = mock.Mock()
        storage.size.return_value = self.fake_storage_size
        return storage


class _DummySwapRuntime(PipelineSwapRuntime):
    """Runtime that records state transitions without touching a real backend."""

    def __init__(self, config=None):
        super().__init__(_DummyConfig() if config is None else config)
        self.to_device = []
        self.to_cpu = []

    def make_cpu_tensor(self, tensor):
        return f"cpu:{tensor}"

    def copy_to_device(self, slot):
        self.to_device.append(slot.name)
        slot.state = "h2d"

    def wait_prefetch_slot(self, slot):
        slot.state = "device"

    def copy_to_cpu(self, slot):
        self.to_cpu.append(slot.name)
        if slot.cpu_tensor is None:
            slot.cpu_tensor = self.make_cpu_tensor(slot.tensor)
        slot.state = "d2h"

    def wait_offload_slot(self, slot):
        slot.state = "host"

    def current_stream(self):
        """A synchronous test runtime owns no compute stream."""
        return None

    def new_stream(self):
        """A synchronous test runtime owns no copy stream."""
        return None

    def stream_context(self, stream):
        """A synchronous test runtime has no device stream to enter."""
        return contextlib.nullcontext()

    def restore_device_storage(self, slot):
        """Test slots are plain sentinels, so there is no storage to restore."""

    def release_device_storage(self, slot):
        """Test slots are plain sentinels, so there is no storage to release."""


class _DummyPackedRuntime(_DummySwapRuntime):
    """Record the backend-neutral two-staging-buffer schedule."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.finished = []

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
        self.finished.append(batch_index)

    def end_packed_step(self):
        self.calls.append("end")


class TestSwapSlot(unittest.TestCase):
    """Swap slot metadata and tensor rebinding."""

    def test_bind_tensor_assigns_plain_tensors_directly(self):
        """A slot without a logical DTensor rebinds by plain assignment."""
        sentinel = object()
        slot = SwapSlot(name="exp_avg", tensor=None, cpu_tensor="cpu-mirror")

        slot.bind_tensor(sentinel)

        self.assertIs(slot.tensor, sentinel)
        self.assertIs(slot.checkpoint_tensor, "cpu-mirror")

    def test_bind_tensor_updates_the_local_shard_in_place(self):
        """A DTensor-like slot keeps its logical wrapper and replaces the local shard."""
        replacement = _DeviceLikeTensor(torch.zeros(4, dtype=torch.float32))
        logical = _LogicalTensor(device=_FakeDevice(), data=torch.arange(4, dtype=torch.float32))
        slot = SwapSlot(name="exp_avg", tensor=torch.arange(4, dtype=torch.float32), logical_tensor=logical)

        slot.bind_tensor(replacement)

        self.assertIs(slot.logical_tensor._local_tensor, replacement)
        self.assertIs(slot.tensor, slot.logical_tensor)
        self.assertIs(slot.tensor.data, replacement)

    def test_bind_tensor_keeps_plain_assignment_for_cpu_targets(self):
        """A CPU-bound logical tensor never enters the local-shard rebinding path."""
        host_view = torch.ones(4)
        logical = _LogicalTensor(device=torch.device("cpu"), data=torch.zeros(4))
        slot = SwapSlot(name="exp_avg", tensor=torch.zeros(4), logical_tensor=logical)

        slot.bind_tensor(host_view)

        self.assertIs(slot.tensor, host_view)
        self.assertIsNot(slot.tensor, slot.logical_tensor)
        self.assertIsNone(logical._local_tensor)
        self.assertEqual(torch.count_nonzero(logical.data).item(), 0)

    def test_checkpoint_tensor_prefers_the_cpu_mirror(self):
        """Checkpoint reads use the CPU mirror whenever one exists."""
        device_tensor = torch.ones(4)
        cpu_tensor = torch.zeros(4)
        slot = SwapSlot(name="exp_avg", tensor=device_tensor)

        self.assertIs(slot.checkpoint_tensor, device_tensor)
        slot.cpu_tensor = cpu_tensor
        self.assertIs(slot.checkpoint_tensor, cpu_tensor)

    def test_defaults_describe_a_swappable_device_slot(self):
        """A freshly built slot is device-resident, swappable and unpacked."""
        slot = SwapSlot(name="exp_avg", tensor=torch.ones(4))

        self.assertTrue(slot.swappable)
        self.assertFalse(slot.packed)
        self.assertEqual(slot.state, "device")
        self.assertIsNone(slot.cpu_tensor)
        self.assertIsNone(slot.event)
        self.assertIsNone(slot.logical_tensor)
        self.assertEqual(slot.storage_nbytes, 0)
        self.assertEqual(slot.host_offset, 0)

    def test_update_unit_keeps_the_adapter_index_and_slots(self):
        """Update units carry the parameter group index beside param, grad and slots."""
        slot = SwapSlot(name="exp_avg", tensor=torch.ones(4))

        unit = UpdateUnit(adapter_index=2, param="param", grad="grad", slots=[slot])

        self.assertEqual(unit.adapter_index, 2)
        self.assertEqual(unit.param, "param")
        self.assertEqual(unit.grad, "grad")
        self.assertEqual(unit.slots, [slot])


class TestSwapRuntimePartition(unittest.TestCase):
    """Cost-balanced batching of update units."""

    def test_packed_enabled_follows_the_config_and_defaults_on(self):
        """Packed staging is opt-out: an unset attribute still enables it."""
        explicit = PipelineSwapRuntime(_DummyConfig(packed_swap=False))
        defaulted = PipelineSwapRuntime(SimpleNamespace(swap_times=1, min_numel=0, state_keys=None))

        self.assertFalse(explicit.packed_enabled)
        self.assertTrue(defaulted.packed_enabled)

    def test_partition_returns_nothing_for_an_empty_step(self):
        """An optimizer step with no update units produces no batches."""
        self.assertEqual(_DummySwapRuntime().partition([]), [])

    def test_partition_clamps_swap_times_to_the_unit_count(self):
        """Requesting more partitions than units yields one batch per unit."""
        runtime = _DummySwapRuntime(_DummyConfig(swap_times=8))
        units = [_unit(SwapSlot(name=f"moment{index}", tensor=object(), storage_nbytes=16)) for index in range(3)]

        batches = runtime.partition(units)

        self.assertEqual([len(batch) for batch in batches], [1, 1, 1])
        self.assertEqual(
            [id(unit) for batch in batches for unit in batch],
            [id(unit) for unit in units],
        )

    def test_partition_floors_zero_cost_units_at_one_byte(self):
        """Units without swappable state still advance the partition cursor."""
        runtime = _DummySwapRuntime(_DummyConfig(swap_times=2))
        units = [_unit(SwapSlot(name="small", tensor=object(), storage_nbytes=0)) for _ in range(4)]

        batches = runtime.partition(units)

        self.assertEqual([len(batch) for batch in batches], [2, 2])

    def test_partition_balances_batches_on_swappable_state_bytes(self):
        """Batches balance on swappable bytes rather than on unit counts."""
        runtime = _DummySwapRuntime(_DummyConfig(swap_times=2))
        units = [
            _unit(SwapSlot(name="big", tensor=object(), storage_nbytes=512)),
            _unit(SwapSlot(name="small", tensor=object(), storage_nbytes=16)),
            _unit(SwapSlot(name="small", tensor=object(), storage_nbytes=16)),
        ]

        batches = runtime.partition(units)

        self.assertEqual([len(batch) for batch in batches], [1, 2])
        self.assertEqual(runtime._unit_cost(batches[0][0]), 512)
        self.assertEqual([runtime._unit_cost(unit) for unit in batches[1]], [16, 16])


class TestSwapRuntimePipeline(unittest.TestCase):
    """The one-batch-ahead per-tensor pipeline."""

    @staticmethod
    def _host_slot(name, cpu_tensor="cpu-mirror"):
        """Build a host-resident swappable slot."""
        return SwapSlot(name=name, tensor=f"{name}-tensor", cpu_tensor=cpu_tensor, storage_nbytes=16, state="host")

    def test_run_pipeline_returns_without_batches(self):
        """No batches means no prefetch or offload at all."""
        runtime = _DummySwapRuntime()

        results = runtime.run_pipeline([], object(), lambda _batch, _ctx: "never")

        self.assertEqual(results, [])
        self.assertEqual(runtime.to_device, [])
        self.assertEqual(runtime.to_cpu, [])

    def test_run_pipeline_prefetches_then_offloads_a_single_batch(self):
        """A one-batch step prefetches cold state and offloads it again at the end."""
        runtime = _DummySwapRuntime()
        slot = self._host_slot("moment")
        seen_states = []

        results = runtime.run_pipeline(
            [[_unit(slot)]],
            object(),
            lambda _batch, _ctx: seen_states.append(slot.state) or "updated",
        )

        self.assertEqual(seen_states, ["device"])
        self.assertEqual(results, ["updated"])
        self.assertEqual(slot.state, "host")
        self.assertEqual(runtime.to_device, ["moment"])
        self.assertEqual(runtime.to_cpu, ["moment"])

    def test_run_pipeline_skips_slots_that_are_not_swappable(self):
        """Unswappable state is never prefetched, updated through swap or offloaded."""
        runtime = _DummySwapRuntime()
        tiny = SwapSlot(name="tiny", tensor="tiny-tensor", swappable=False, state="device")
        live = self._host_slot("moment")

        results = runtime.run_pipeline([[_unit([tiny, live])]], object(), lambda _batch, _ctx: "updated")

        self.assertEqual(results, ["updated"])
        self.assertEqual(runtime.to_device, ["moment"])
        self.assertEqual(runtime.to_cpu, ["moment"])
        self.assertEqual(tiny.state, "device")
        self.assertIsNone(tiny.cpu_tensor)

    def test_run_pipeline_prefetches_the_next_batch_while_updating(self):
        """Two batches overlap: batch 1 is on device before batch 0 is offloaded."""
        runtime = _DummySwapRuntime()
        first = self._host_slot("first")
        second = self._host_slot("second")
        observed = []

        def _step(batch, ctx):
            del ctx
            observed.append(tuple(slot.state for slot in batch[0].slots))
            return "updated"

        runtime.run_pipeline([[_unit(first)], [_unit(second)]], object(), _step)

        self.assertEqual(observed, [("device",), ("device",)])
        self.assertEqual(runtime.to_device, ["first", "second"])
        self.assertEqual(runtime.to_cpu, ["first", "second"])
        self.assertEqual([first.state, second.state], ["host", "host"])

    def test_run_pipeline_defers_to_the_packed_pipeline_when_eligible(self):
        """An eligible step never enters the per-tensor path."""
        runtime = _DummySwapRuntime()
        slot = self._host_slot("moment")
        runtime.supports_packed_pipeline = mock.Mock(return_value=True)
        runtime._run_packed_pipeline = mock.Mock(return_value=["packed"])

        results = runtime.run_pipeline([[_unit(slot)]], object(), lambda _batch, _ctx: "updated")

        self.assertEqual(results, ["packed"])
        runtime._run_packed_pipeline.assert_called_once()
        self.assertEqual(runtime.to_device, [])

    def test_run_pipeline_keeps_the_packed_path_off_when_ineligible(self):
        """A disabling gate routes the step through per-tensor swap only."""
        runtime = _DummySwapRuntime()
        slot = self._host_slot("moment")
        runtime.supports_packed_pipeline = mock.Mock(return_value=False)
        runtime._run_packed_pipeline = mock.Mock(side_effect=AssertionError("packed pipeline entered"))

        results = runtime.run_pipeline([[_unit(slot)]], object(), lambda _batch, _ctx: "updated")

        self.assertEqual(results, ["updated"])
        runtime.supports_packed_pipeline.assert_called_once()
        runtime._run_packed_pipeline.assert_not_called()

    def test_base_hooks_keep_results_and_slot_state(self):
        """The base runtime retains results and defers refresh to subclasses."""
        runtime = _DummySwapRuntime()
        results = [object()]
        slot = SwapSlot(name="moment", tensor=object(), state="device")

        runtime.refresh_swappable_slots([_unit(slot)])
        runtime.release_packed_step_results(results)

        self.assertEqual(len(results), 1)
        self.assertEqual(slot.state, "device")


class TestSwapRuntimePackedPipeline(unittest.TestCase):
    """The packed two-staging-buffer schedule without a device backend."""

    def setUp(self):
        self.runtime = _DummyPackedRuntime()
        self.runtime._run_packed_pipeline = types.MethodType(PipelineSwapRuntime._run_packed_pipeline, self.runtime)

    @staticmethod
    def _batches(count):
        """Build ``count`` single-unit batches with distinct slots."""
        return [[_unit(SwapSlot(name=f"moment{index}", tensor=object()))] for index in range(count)]

    def _step(self, batch, ctx):
        """Record the batch that is currently being updated."""
        del ctx
        self.runtime.calls.append(f"update:{batch[0].slots[0].name}")
        return "updated"

    def test_two_batches_alternate_staging_buffers_and_drain_together(self):
        """Parity follows the batch index and both batches are drained at the end."""
        runtime = self.runtime

        results = runtime._run_packed_pipeline(self._batches(2), object(), self._step)

        self.assertEqual(results, ["updated", "updated"])
        self.assertEqual(runtime.finished, [0, 1])
        self.assertEqual(runtime.calls, [
            "begin:2",
            "prefetch:0:0",
            "prefetch:1:1",
            "wait_prefetch:0:0",
            "activate:0:0",
            "update:moment0",
            "chain:0:None:0",
            "wait_prefetch:1:1",
            "activate:1:1",
            "update:moment1",
            "chain:1:None:1",
            "wait_offload:0",
            "finish_offload:0",
            "wait_offload:1",
            "finish_offload:1",
            "end",
        ])

    def test_single_batch_waits_for_its_own_offload(self):
        """A one-batch packed step drains the batch it just offloaded."""
        runtime = self.runtime
        batch = [_unit(SwapSlot(name="moment0", tensor=object()))]

        results = runtime.run_pipeline([batch], object(), self._step)

        self.assertEqual(results, ["updated"])
        self.assertEqual(runtime.calls, [
            "begin:1",
            "prefetch:0:0",
            "wait_prefetch:0:0",
            "activate:0:0",
            "update:moment0",
            "chain:0:None:0",
            "wait_offload:0",
            "finish_offload:0",
            "end",
        ])

    def test_three_batches_release_the_first_once_the_third_is_ready(self):
        """A completed batch is drained two steps later, not only at the very end."""
        runtime = self.runtime

        runtime._run_packed_pipeline(self._batches(3), object(), self._step)

        self.assertEqual(runtime.finished, [0, 1, 2])
        drain_positions = [index for index, call in enumerate(runtime.calls) if call == "wait_offload:0"]
        self.assertEqual(len(drain_positions), 1)
        self.assertLess(drain_positions[0], runtime.calls.index("update:moment2"))

    def test_packed_pipeline_tears_down_after_a_failing_update(self):
        """A failing update still releases staging state and re-raises."""
        runtime = self.runtime

        def _step(batch, ctx):
            del batch, ctx
            raise RuntimeError("update failed")

        with self.assertRaisesRegex(RuntimeError, "update failed"):
            runtime._run_packed_pipeline(self._batches(1), object(), _step)

        self.assertEqual(runtime.calls[-1], "end")


class TestSwapRuntimeOffloadWindow(unittest.TestCase):
    """Copy-stream bookkeeping for prefetch, offload and their waits."""

    def test_initial_offload_drains_only_device_resident_swappable_slots(self):
        """Slots already on the host are left untouched by the first offload."""
        runtime = _DummySwapRuntime()
        cold = SwapSlot(name="cold", tensor="cold-tensor", storage_nbytes=16, state="device")
        already_host = SwapSlot(
            name="host",
            tensor="host-tensor",
            cpu_tensor="cpu:host-tensor",
            storage_nbytes=16,
            state="host",
        )

        runtime.offload_initial_slots([cold, already_host])

        self.assertEqual(runtime.to_cpu, ["cold"])
        self.assertEqual(cold.state, "host")
        self.assertEqual(already_host.state, "host")

    def test_prefetch_ignores_host_slots_without_swappable_state(self):
        """Nothing is copied when a batch only holds unswappable state."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="tiny", tensor="tiny-tensor", swappable=False, state="host")

        runtime.prefetch([_unit(slot)])

        self.assertEqual(runtime.to_device, [])
        self.assertEqual(slot.state, "host")

    def test_prefetch_requires_a_cpu_mirror_for_host_slots(self):
        """A host-resident slot without a mirror is a hard error, not a silent skip."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="moment", tensor="moment-tensor", cpu_tensor=None, state="host")

        with self.assertRaisesRegex(RuntimeError, "host-resident but has no CPU mirror"):
            runtime.prefetch([_unit(slot)])

    def test_prefetch_restores_device_storage_before_copying(self):
        """H2D runs after device storage is re-allocated and leaves slots in h2d."""
        runtime = _DummySwapRuntime()
        restored = []
        slot = SwapSlot(
            name="moment",
            tensor="moment-tensor",
            cpu_tensor="cpu-mirror",
            storage_nbytes=16,
            state="host",
        )
        runtime.restore_device_storage = restored.append

        runtime.prefetch([_unit(slot)])

        self.assertEqual(restored, [slot])
        self.assertEqual(slot.state, "h2d")
        self.assertEqual(runtime.to_device, ["moment"])

    def test_prefetch_copies_each_slot_alias_once(self):
        """State reachable through several units is transferred a single time."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="moment", tensor="moment-tensor", cpu_tensor="cpu-mirror", storage_nbytes=16, state="host")
        other = SwapSlot(name="other", tensor="other-tensor", storage_nbytes=16, state="device")

        runtime.prefetch([_unit([slot, slot]), _unit(other)])

        self.assertEqual(runtime.to_device, ["moment"])

    def test_wait_prefetch_clears_events_after_ordering(self):
        """Host-to-device completion marks slots device-resident and clears their event."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="moment", tensor="moment-tensor", storage_nbytes=16, state="h2d", event=mock.Mock())

        runtime.wait_prefetch([_unit(slot)])

        self.assertEqual(slot.state, "device")
        self.assertIsNone(slot.event)

    def test_offload_ignores_host_and_unswappable_slots(self):
        """Only device-resident swappable slots enter the D2H window."""
        runtime = _DummySwapRuntime()
        host_slot = SwapSlot(name="host", tensor="t0", cpu_tensor="cpu:t0", storage_nbytes=16, state="host")
        tiny_slot = SwapSlot(name="tiny", tensor="t1", swappable=False, storage_nbytes=16, state="device")
        live_slot = SwapSlot(name="live", tensor="t2", storage_nbytes=16, state="device")

        runtime.offload([_unit([host_slot, tiny_slot, live_slot])])

        self.assertEqual(runtime.to_cpu, ["live"])
        self.assertEqual(host_slot.state, "host")
        self.assertEqual(tiny_slot.state, "device")
        self.assertEqual(live_slot.state, "d2h")

    def test_synchronize_cpu_mirrors_requires_host_slots_to_keep_a_mirror(self):
        """Checkpointing fails loudly when a host slot lost its CPU mirror."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="moment", tensor=object(), cpu_tensor=None, swappable=True, state="host")

        with self.assertRaisesRegex(RuntimeError, "host-resident but has no CPU mirror"):
            runtime.synchronize_cpu_mirrors([slot])

    def test_synchronize_cpu_mirrors_copies_device_slots_before_checkpoint(self):
        """Device-resident state is copied to its mirror, then released."""
        runtime = _DummySwapRuntime()
        runtime.record_event = mock.Mock(return_value=None)
        device_slot = SwapSlot(name="device-slot", tensor="t0", cpu_tensor="cpu:t0", storage_nbytes=16, state="device")
        pending_slot = SwapSlot(name="pending-slot", tensor="t1", storage_nbytes=16, state="h2d")

        runtime.synchronize_cpu_mirrors([device_slot, pending_slot])

        self.assertEqual(runtime.to_cpu, ["device-slot", "pending-slot"])
        self.assertEqual(device_slot.state, "host")
        self.assertEqual(pending_slot.state, "host")
        self.assertIsNone(device_slot.event)
        runtime.record_event.assert_not_called()

    def test_synchronize_cpu_mirrors_skips_unswappable_and_empty_inputs(self):
        """State outside swap management is never copied for checkpointing."""
        runtime = _DummySwapRuntime()
        slot = SwapSlot(name="tiny", tensor="t0", swappable=False, state="device")

        runtime.synchronize_cpu_mirrors([slot])
        runtime.synchronize_cpu_mirrors([])

        self.assertEqual(runtime.to_cpu, [])
        self.assertEqual(slot.state, "device")


class TestSwapRuntimeTensorTransfer(unittest.TestCase):
    """Real Torch storage handling on a CPU-only host."""

    def setUp(self):
        # ``pin_memory`` needs an accelerator allocator. Drop it so the CPU-only
        # UT host exercises the same pageable-memory fallback a device run takes.
        patcher = mock.patch.object(torch, "empty_like", _unpinned_empty_like)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runtime = PipelineSwapRuntime(_DummyConfig())

    def test_make_cpu_tensor_preserves_shape_dtype_and_values(self):
        """The CPU mirror is a copy, not a view of the source tensor."""
        source = torch.arange(8, dtype=torch.float32).view(2, 4)

        mirror = self.runtime.make_cpu_tensor(source)

        self.assertEqual(mirror.device.type, "cpu")
        self.assertEqual(mirror.dtype, source.dtype)
        self.assertEqual(mirror.shape, source.shape)
        self.assertTrue(torch.equal(mirror, source))
        mirror.add_(1)
        self.assertTrue(torch.equal(source, torch.arange(8, dtype=torch.float32).view(2, 4)))

    def test_make_cpu_tensor_falls_back_to_pageable_memory(self):
        """A host without pinned memory still gets a usable CPU mirror."""
        source = torch.arange(4, dtype=torch.float32)

        with mock.patch.object(
                torch,
                "empty_like",
                side_effect=[RuntimeError("no pinned allocator"), _TORCH_EMPTY_LIKE(source)],
        ):
            mirror = self.runtime.make_cpu_tensor(source)

        self.assertTrue(torch.equal(mirror, source))

    def test_make_cpu_tensor_rejects_non_tensor_state(self):
        """Non-tensor optimizer state cannot be mirrored."""
        with self.assertRaisesRegex(ValueError, "Expected torch.Tensor for CPU mirror"):
            self.runtime.make_cpu_tensor(SimpleNamespace())

    def test_make_zero_cpu_tensor_like_allocates_zeros(self):
        """Lazily created swappable state starts from a zeroed CPU mirror."""
        mirror = self.runtime.make_zero_cpu_tensor_like(torch.full((4,), 3.0))

        self.assertEqual(mirror.device.type, "cpu")
        self.assertEqual(mirror.shape, (4,))
        self.assertEqual(torch.count_nonzero(mirror).item(), 0)

    def test_make_zero_cpu_tensor_like_rejects_non_tensor_state(self):
        """The zeroed mirror helper validates its input type as well."""
        with self.assertRaisesRegex(ValueError, "Expected torch.Tensor for CPU mirror"):
            self.runtime.make_zero_cpu_tensor_like(None)

    def test_make_device_tensor_like_detaches_and_clones(self):
        """The live state tensor never aliases the saved checkpoint tensor."""
        param = SimpleNamespace(device=torch.device("cpu"))
        saved = torch.arange(4, dtype=torch.float64)

        tensor = self.runtime.make_device_tensor_like(param, saved)

        self.assertTrue(torch.equal(tensor, saved))
        self.assertEqual(tensor.dtype, saved.dtype)
        tensor.add_(1)
        self.assertTrue(torch.equal(saved, torch.arange(4, dtype=torch.float64)))

    def test_make_device_tensor_like_rejects_non_tensor_state(self):
        """Checkpoint state that is not a tensor is rejected."""
        param = SimpleNamespace(device=torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected torch.Tensor in optimizer state"):
            self.runtime.make_device_tensor_like(param, 1.0)

    def test_make_empty_device_tensor_like_keeps_shape_and_dtype(self):
        """The device placeholder is an uninitialized shell, not a copy."""
        param = SimpleNamespace(device=torch.device("cpu"))
        saved = torch.arange(6, dtype=torch.float32).view(2, 3)

        tensor = self.runtime.make_empty_device_tensor_like(param, saved)

        self.assertEqual(tensor.shape, saved.shape)
        self.assertEqual(tensor.dtype, saved.dtype)
        self.assertEqual(tensor.device.type, "cpu")

    def test_make_empty_device_tensor_like_rejects_non_tensor_state(self):
        """An uninitialized placeholder still needs a tensor template."""
        param = SimpleNamespace(device=torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected torch.Tensor in optimizer state"):
            self.runtime.make_empty_device_tensor_like(param, object())

    def test_copy_to_device_skips_slots_that_are_not_host_resident(self):
        """H2D only runs for a slot that is currently on the host."""
        runtime = self.runtime
        slot = SwapSlot(name="moment", tensor=torch.zeros(4), cpu_tensor=torch.ones(4), state="device")

        runtime.copy_to_device(slot)

        self.assertEqual(slot.state, "device")
        self.assertEqual(torch.count_nonzero(slot.tensor).item(), 0)

    def test_copy_to_device_skips_host_slots_without_a_cpu_mirror(self):
        """A host slot without a mirror cannot be copied to the device."""
        runtime = self.runtime
        slot = SwapSlot(name="moment", tensor=torch.zeros(4), cpu_tensor=None, state="host")

        runtime.copy_to_device(slot)

        self.assertEqual(slot.state, "host")

    def test_copy_to_device_reloads_host_values(self):
        """H2D restores the device tensor from the CPU mirror and marks it h2d."""
        runtime = self.runtime
        slot = SwapSlot(name="moment", tensor=torch.zeros(4), cpu_tensor=torch.ones(4), state="host")

        runtime.copy_to_device(slot)

        self.assertEqual(slot.state, "h2d")
        self.assertTrue(torch.equal(slot.tensor, torch.ones(4)))

    def test_copy_to_cpu_creates_the_mirror_then_updates_it_in_place(self):
        """The first D2H allocates the mirror, later offloads reuse it."""
        runtime = self.runtime
        slot = SwapSlot(name="moment", tensor=torch.ones(4), state="device")

        runtime.copy_to_cpu(slot)
        mirror = slot.cpu_tensor
        self.assertEqual(slot.state, "d2h")
        self.assertTrue(torch.equal(mirror, torch.ones(4)))

        slot.tensor = torch.full((4,), 7.0)
        runtime.copy_to_cpu(slot)

        self.assertIs(slot.cpu_tensor, mirror)
        self.assertTrue(torch.equal(mirror, torch.full((4,), 7.0)))

    def test_wait_prefetch_slot_marks_slots_device_resident(self):
        """Torch copies are stream ordered, so the wait only flips the state."""
        slot = SwapSlot(name="moment", tensor=torch.ones(4), state="h2d")

        self.runtime.wait_prefetch_slot(slot)

        self.assertEqual(slot.state, "device")

    def test_release_and_restore_device_storage_resize_the_live_storage(self):
        """Offload shrinks device storage to zero and prefetch grows it back."""
        runtime = _like_device_runtime()
        tensor = torch.ones(9)
        slot = SwapSlot(name="moment", tensor=_DeviceLikeTensor(tensor, _FakeDevice()), state="device")
        runtime.populate_slot_metadata(slot, slot.tensor)
        storage = tensor.untyped_storage()

        runtime.release_device_storage(slot)
        self.assertEqual(storage.size(), 0)

        runtime.restore_device_storage(slot)
        self.assertEqual(storage.size(), slot.storage_nbytes)
        self.assertEqual(slot.storage_nbytes, 36)

    def test_restore_device_storage_leaves_allocated_storage_alone(self):
        """A prefetch that follows an unreleased offload keeps the same storage."""
        runtime = self.runtime
        slot = SwapSlot(name="moment", tensor=torch.ones(4), state="host")
        runtime.populate_slot_metadata(slot, slot.tensor)
        storage = slot.tensor.untyped_storage()

        runtime.restore_device_storage(slot)

        self.assertIs(slot.tensor.untyped_storage(), storage)
        self.assertEqual(storage.size(), slot.storage_nbytes)

    def test_storage_helpers_skip_cpu_resident_slots(self):
        """Packed host views have no device storage to resize."""
        runtime = self.runtime
        slot = SwapSlot(name="moment", tensor=torch.ones(4), state="host")
        runtime.populate_slot_metadata(slot, slot.tensor)

        runtime.release_device_storage(slot)
        runtime.restore_device_storage(slot)

        self.assertEqual(slot.tensor.untyped_storage().size(), slot.storage_nbytes)

    def test_wait_offload_slot_releases_device_storage_and_marks_host(self):
        """D2H completion is the point where device storage is freed."""
        runtime = _like_device_runtime()
        tensor = torch.ones(4)
        slot = SwapSlot(name="moment", tensor=_DeviceLikeTensor(tensor, _FakeDevice()),
                        cpu_tensor=torch.zeros(4), state="d2h")
        runtime.populate_slot_metadata(slot, slot.tensor)

        runtime.wait_offload_slot(slot)

        self.assertEqual(slot.state, "host")
        self.assertEqual(tensor.untyped_storage().size(), 0)

    def test_wait_offload_slot_skips_release_for_cpu_tensors(self):
        """A slot already bound to a host view only flips its state."""
        runtime = self.runtime
        slot = SwapSlot(name="moment", tensor=torch.ones(4), state="d2h")

        runtime.wait_offload_slot(slot)

        self.assertEqual(slot.state, "host")
        self.assertNotEqual(slot.tensor.untyped_storage().size(), 0)

    def test_initial_offload_then_prefetch_round_trips_storage(self):
        """A cold device slot is released on offload and restored on prefetch."""
        runtime = _like_device_runtime(swap_times=1)
        runtime._get_copy_stream = lambda: None
        runtime.current_stream = lambda: None
        tensor = torch.ones(8)
        slot = SwapSlot(name="moment", tensor=tensor, swappable=True, state="device")
        runtime.populate_slot_metadata(slot, slot.tensor)

        runtime.offload_initial_slots([slot])

        # The slot binds a device-like tensor, so the mirror is filled from it.
        self.assertEqual(slot.state, "host")
        self.assertEqual(tensor.untyped_storage().size(), slot.storage_nbytes)
        self.assertTrue(torch.equal(slot.cpu_tensor, torch.ones(8)))

        runtime.prefetch([_unit(slot)])
        runtime.wait_prefetch([_unit(slot)])

        self.assertEqual(slot.state, "device")
        self.assertEqual(tensor.untyped_storage().size(), slot.storage_nbytes)


class TestSwapRuntimeTensorEligibility(unittest.TestCase):
    """Swappability and storage-size accounting rules."""

    def setUp(self):
        self.runtime = PipelineSwapRuntime(_DummyConfig())

    def test_storage_tensor_unwraps_dtensor_like_state(self):
        """Distributed state is measured and copied through its local shard."""
        local = torch.ones(4)
        wrapper = SimpleNamespace(to_local=lambda: local)

        self.assertIs(self.runtime._storage_tensor(wrapper), local)
        self.assertIs(self.runtime._storage_tensor(local), local)

    def test_is_distributed_tensor_detects_the_local_shard_protocol(self):
        """Only objects exposing ``to_local`` are treated as DTensor-like."""
        self.assertTrue(self.runtime.is_distributed_tensor(SimpleNamespace(to_local=lambda: None)))
        self.assertFalse(self.runtime.is_distributed_tensor(torch.ones(4)))
        self.assertFalse(self.runtime.is_distributed_tensor(None))

    def test_storage_nbytes_uses_the_full_storage_size(self):
        """State bytes follow the untyped storage, not the view element count."""
        base = torch.zeros(16, dtype=torch.float32)
        view = base.narrow(0, 0, 4)

        self.assertEqual(self.runtime.storage_nbytes(view), 64)
        self.assertEqual(self.runtime.storage_nbytes(base), 64)

    def test_storage_nbytes_rejects_non_tensors(self):
        """Non-tensor state reports zero bytes instead of raising."""
        self.assertEqual(self.runtime.storage_nbytes(None), 0)
        self.assertEqual(self.runtime.storage_nbytes(SimpleNamespace()), 0)

    def test_storage_nbytes_falls_back_when_storage_is_unavailable(self):
        """Tensors that cannot report storage fall back to numel times element size."""
        fake = _FakeScheduleTensor.make(numel=4, element_size=4, storage_error=RuntimeError("no storage"))

        self.assertEqual(self.runtime.storage_nbytes(fake), 16)
        self.assertFalse(self.runtime.is_swappable_tensor(fake, 1))

    def test_is_swappable_tensor_accepts_a_plain_device_template(self):
        """Floating point state above the threshold with exact storage is swappable."""
        runtime = self.runtime
        template = _FakeScheduleTensor.make(numel=8, element_size=4, storage_size=32)

        self.assertTrue(runtime.is_swappable_tensor(template, 8))
        self.assertTrue(runtime.is_swappable_tensor(template, 0))

    def test_is_swappable_tensor_checks_the_size_threshold(self):
        """State smaller than ``min_numel`` stays resident."""
        self.assertFalse(self.runtime.is_swappable_tensor(torch.ones(8), 9))

    def test_is_swappable_tensor_rejects_cpu_tensors(self):
        """CPU state has nothing to swap."""
        self.assertFalse(self.runtime.is_swappable_tensor(torch.ones(8), 0))

    def test_is_swappable_tensor_rejects_non_tensor_state(self):
        """Objects without the Torch tensor protocol are never swappable."""
        self.assertFalse(self.runtime.is_swappable_tensor(SimpleNamespace(), 0))
        self.assertFalse(self.runtime.is_swappable_tensor(None, 0))

    def test_is_swappable_tensor_rejects_integer_state(self):
        """Only floating point optimizer state is swapped."""
        fake = _FakeScheduleTensor.make(floating=False)
        self.runtime._storage_tensor = lambda _tensor: fake

        self.assertFalse(self.runtime.is_swappable_tensor(fake, 0))

    def test_is_swappable_tensor_rejects_sparse_state(self):
        """Sparse state cannot participate in the dense copy pipeline."""
        fake = _FakeScheduleTensor.make(sparse=True)
        self.runtime._storage_tensor = lambda _tensor: fake

        self.assertFalse(self.runtime.is_swappable_tensor(fake, 0))

    def test_is_swappable_tensor_rejects_non_contiguous_state(self):
        """Only contiguous state can be copied as one storage range."""
        fake = _FakeScheduleTensor.make(contiguous=False)
        self.runtime._storage_tensor = lambda _tensor: fake

        self.assertFalse(self.runtime.is_swappable_tensor(fake, 0))

    def test_is_swappable_tensor_rejects_shared_storage(self):
        """State whose storage is larger than its elements is never swapped."""
        fake = _FakeScheduleTensor.make(numel=4, element_size=4, storage_size=64)
        self.runtime._storage_tensor = lambda _tensor: fake

        self.assertFalse(self.runtime.is_swappable_tensor(fake, 0))

    def test_is_packable_template_follows_the_packed_switch(self):
        """Templates are only packable while packed staging is enabled."""
        runtime = self.runtime
        template = torch.ones(8)
        runtime.is_swappable_tensor = mock.Mock(return_value=True)

        runtime._packed_enabled = False
        self.assertFalse(runtime.is_packable_template(template, 1))
        runtime.is_swappable_tensor.assert_not_called()

        runtime._packed_enabled = True
        self.assertTrue(runtime.is_packable_template(template, 1))
        runtime.is_swappable_tensor.assert_called_once_with(template, 1)

    def test_populate_slot_metadata_copies_the_view_layout(self):
        """Slot metadata follows the view shape even when its storage is wider."""
        base = torch.zeros(16, dtype=torch.float32)
        template = base.narrow(0, 0, 3)
        slot = SwapSlot(name="exp_avg", tensor=None)

        self.runtime.populate_slot_metadata(slot, template)

        self.assertEqual(slot.shape, (3,))
        self.assertEqual(slot.dtype, torch.float32)
        self.assertEqual(slot.device, template.device)
        self.assertEqual(slot.numel, 3)
        self.assertEqual(slot.storage_nbytes, 12)
        # The bytes-on-device helper reports the whole storage instead.
        self.assertEqual(self.runtime.storage_nbytes(template), 64)

    def test_unit_cost_counts_only_swappable_state(self):
        """Partition cost ignores resident state that is never copied."""
        unit = _unit([
            SwapSlot(name="swapped", tensor=object(), storage_nbytes=64),
            SwapSlot(name="resident", tensor=object(), storage_nbytes=4096, swappable=False),
        ])

        self.assertEqual(self.runtime._unit_cost(unit), 64)

    def test_unit_cost_is_zero_without_slots(self):
        """Units with no state contribute no cost to the partition."""
        self.assertEqual(self.runtime._unit_cost(_unit([])), 0)

    def test_align_bytes_rounds_up_to_the_packed_alignment(self):
        """Packed layouts align every dtype region to the fixed boundary."""
        cases = {0: 0, 1: 512, 511: 512, 512: 512, 513: 1024, 1024: 1024, 1500: 1536}
        for value, expected in cases.items():
            with self.subTest(num_bytes=value):
                self.assertEqual(PipelineSwapRuntime._align_bytes(value), expected)


class TestSwapRuntimePackedPlanning(unittest.TestCase):
    """Packed batch plans, device validation and host packing."""

    def setUp(self):
        self.runtime = PipelineSwapRuntime(_DummyConfig(packed_swap=True))
        # Host packing asks for pinned buffers, which a CPU-only host cannot provide.
        self._allocators = _patch_allocators()
        self.addCleanup(lambda: [patcher.stop() for patcher in self._allocators])

    @staticmethod
    def _packed_slot(host_offset, numel=4, dtype=torch.float32, state="host"):
        """Build one host-packed slot with an explicit offset in the host buffer."""
        slot = SwapSlot(name="exp_avg", tensor=None, swappable=True, packed=True, state=state)
        slot.dtype = dtype
        slot.numel = numel
        slot.shape = (numel,)
        slot.storage_nbytes = numel * 4
        slot.host_offset = host_offset
        slot.cpu_tensor = torch.zeros(numel, dtype=dtype)
        return slot

    def test_build_packed_batch_plan_groups_contiguous_dtype_ranges(self):
        """One dtype maps to a single host range covering its contiguous slots."""
        first = self._packed_slot(0)
        second = self._packed_slot(4)
        bf16 = self._packed_slot(0, numel=4, dtype=torch.bfloat16)

        plan = self.runtime._build_packed_batch_plan([_unit([first, second, bf16])])

        self.assertEqual(set(plan.regions), {torch.float32, torch.bfloat16})
        fp32_region = plan.regions[torch.float32]
        self.assertEqual((fp32_region.host_offset, fp32_region.numel), (0, 8))
        self.assertEqual(fp32_region.slots, [first, second])
        bf16_region = plan.regions[torch.bfloat16]
        self.assertEqual((bf16_region.host_offset, bf16_region.numel), (0, 4))
        self.assertEqual(bf16_region.slots, [bf16])

    def test_build_packed_batch_plan_sorts_slots_by_host_offset(self):
        """Region slots follow the host layout even when units arrive out of order."""
        late = self._packed_slot(4)
        early = self._packed_slot(0)

        plan = self.runtime._build_packed_batch_plan([_unit([late]), _unit([early])])

        region = plan.regions[torch.float32]
        self.assertEqual(region.slots, [early, late])
        self.assertEqual(region.host_offset, 0)

    def test_build_packed_batch_plan_skips_unpacked_and_unswappable_slots(self):
        """Only packed swappable state participates in a transfer region."""
        packed = self._packed_slot(0)
        unpacked = SwapSlot(name="exp_avg_sq", tensor=torch.zeros(4), swappable=True, packed=False)
        tiny = self._packed_slot(0)
        tiny.swappable = False

        plan = self.runtime._build_packed_batch_plan([_unit([packed, unpacked, tiny])])

        self.assertEqual(plan.regions[torch.float32].slots, [packed])

    def test_build_packed_batch_plan_ignores_an_empty_batch(self):
        """A batch without packed state produces an empty plan."""
        self.assertEqual(self.runtime._build_packed_batch_plan([]).regions, {})

    def test_build_packed_batch_plan_rejects_a_non_contiguous_host_range(self):
        """A gap in the host layout means the batch plan cannot be packed."""
        with self.assertRaisesRegex(RuntimeError, "non-contiguous"):
            self.runtime._build_packed_batch_plan([_unit([self._packed_slot(0), self._packed_slot(8)])])

    def test_validate_packed_devices_accepts_a_single_device(self):
        """Host packing is allowed when every candidate state shares a device."""
        first = SwapSlot(name="exp_avg", tensor=None, device=torch.device("meta"))
        second = SwapSlot(name="exp_avg_sq", tensor=None, device=torch.device("meta"))

        self.runtime.validate_packed_devices([first, second])

    def test_validate_packed_devices_accepts_empty_candidates(self):
        """A step without packed state has no device constraint to enforce."""
        self.runtime.validate_packed_devices([])

    def test_validate_packed_devices_rejects_spread_and_unknown_devices(self):
        """Slots spanning devices or carrying no device metadata fail before packing."""
        cases = {
            "spread": [
                SwapSlot(name="exp_avg", tensor=None, device=torch.device("meta:0")),
                SwapSlot(name="exp_avg_sq", tensor=None, device=torch.device("meta:1")),
            ],
            "unknown": [SwapSlot(name="exp_avg", tensor=None, device=None)],
        }
        for case, slots in cases.items():
            with self.subTest(devices=case), self.assertRaisesRegex(RuntimeError, "single local device"):
                self.runtime.validate_packed_devices(slots)

    def test_prepare_packed_host_is_skipped_when_packing_is_disabled(self):
        """A per-tensor runtime never builds host buffers."""
        runtime = PipelineSwapRuntime(_DummyConfig())
        slot = self._packed_slot(0)

        runtime.prepare_packed_host([slot])

        self.assertEqual(runtime._host_buffers, {})
        self.assertIsNone(slot.tensor)

    def test_prepare_packed_host_reuses_buffers_for_an_unchanged_layout(self):
        """The host layout is rebuilt only when slot identity, dtype or numel changes."""
        runtime = self.runtime
        slot = SwapSlot(
            name="exp_avg",
            tensor=torch.arange(4, dtype=torch.float32),
            swappable=True,
            packed=True,
        )
        runtime.populate_slot_metadata(slot, slot.tensor)

        runtime.prepare_packed_host([slot])
        first_buffer = slot.cpu_tensor
        runtime.prepare_packed_host([slot])

        self.assertIs(slot.cpu_tensor, first_buffer)
        self.assertEqual(
            runtime._host_buffers[torch.float32].untyped_storage().data_ptr(),
            first_buffer.untyped_storage().data_ptr(),
        )

    def test_prepare_packed_host_packs_by_dtype_and_releases_device_state(self):
        """One host buffer per dtype holds every slot at its recorded offset."""
        runtime = _like_device_runtime()
        fp32 = SwapSlot(name="exp_avg", tensor=None, swappable=True, packed=True, state="pending")
        bf16 = SwapSlot(name="exp_avg_sq", tensor=None, swappable=True, packed=True, state="pending")
        fp32.cpu_tensor = torch.arange(4, dtype=torch.float32)
        bf16.cpu_tensor = torch.arange(2, dtype=torch.bfloat16)
        for slot, template in ((fp32, torch.arange(4, dtype=torch.float32)),
                               (bf16, torch.arange(2, dtype=torch.bfloat16))):
            runtime.populate_slot_metadata(slot, template)
        cursor = []
        runtime.release_device_storage = cursor.append

        runtime.prepare_packed_host([fp32, bf16])

        self.assertEqual(set(runtime._host_buffers), {torch.float32, torch.bfloat16})
        self.assertEqual((fp32.host_offset, bf16.host_offset), (0, 0))
        self.assertTrue(torch.equal(fp32.cpu_tensor, torch.arange(4, dtype=torch.float32)))
        self.assertTrue(torch.equal(bf16.cpu_tensor, torch.arange(2, dtype=torch.bfloat16)))
        # The host mirrors are already CPU-resident, so no device storage is freed.
        self.assertEqual(cursor, [])
        self.assertEqual([fp32.state, bf16.state], ["host", "host"])

    def test_prepare_packed_host_lays_out_two_slots_of_one_dtype(self):
        """A second same-dtype slot lands after the first, not on top of it."""
        runtime = self.runtime
        first = SwapSlot(name="exp_avg", tensor=torch.arange(4, dtype=torch.float32), swappable=True, packed=True)
        second = SwapSlot(name="exp_avg_sq", tensor=torch.arange(3, dtype=torch.float32) + 10,
                          swappable=True, packed=True)
        runtime.populate_slot_metadata(first, first.tensor)
        runtime.populate_slot_metadata(second, second.tensor)

        runtime.prepare_packed_host([first, second])

        self.assertEqual((first.host_offset, second.host_offset), (0, 4))
        self.assertEqual(runtime._host_buffers[torch.float32].numel(), 7)
        self.assertTrue(torch.equal(first.cpu_tensor, torch.arange(4, dtype=torch.float32)))
        self.assertTrue(torch.equal(second.cpu_tensor, torch.arange(3, dtype=torch.float32) + 10))

    def test_prepare_packed_host_zeroes_slots_without_a_data_source(self):
        """A slot created before its state existed is packed as zeros."""
        runtime = self.runtime
        slot = SwapSlot(name="exp_avg", tensor=None, swappable=True, packed=True, state="pending")
        slot.dtype = torch.float32
        slot.numel = 4
        slot.shape = (4,)
        slot.device = torch.device("cpu")
        slot.host_offset = 0

        runtime.prepare_packed_host([slot])

        self.assertEqual(torch.count_nonzero(slot.cpu_tensor).item(), 0)
        self.assertEqual(slot.state, "host")

    def test_prepare_packed_host_ignores_unpacked_and_unswappable_slots(self):
        """Only fully packed candidates enter the host layout."""
        runtime = self.runtime
        unpacked = SwapSlot(name="exp_avg", tensor=torch.ones(4), swappable=True, packed=False)
        tiny = SwapSlot(name="exp_avg_sq", tensor=torch.ones(4), swappable=False, packed=True)

        runtime.prepare_packed_host([unpacked, tiny])

        self.assertEqual(runtime._host_buffers, {})
        self.assertEqual(runtime._host_layout_signature, ())

    def test_prepare_packed_host_reports_a_mixed_device_layout(self):
        """Packing states from two devices fails before any slot is rebound."""
        runtime = self.runtime
        first = self._packed_slot(0)
        second = self._packed_slot(4)
        first.device = torch.device("meta:0")
        second.device = torch.device("meta:1")

        with mock.patch.object(torch, "empty", _unpinned_empty), \
                self.assertRaisesRegex(RuntimeError, "single local device"):
            runtime.prepare_packed_host([first, second])


class TestSwapRuntimePackedStaging(unittest.TestCase):
    """Staging arenas, dtype views and the packed copy helpers."""

    def setUp(self):
        self.runtime = PipelineSwapRuntime(_DummyConfig(packed_swap=True))

    @staticmethod
    def _host_packed_slot(source, host_offset):
        """Build a host-packed slot whose values live in ``source``."""
        slot = SwapSlot(name="exp_avg", tensor=source, swappable=True, packed=True, state="host")
        slot.dtype = source.dtype
        slot.numel = source.numel()
        slot.shape = tuple(source.shape)
        slot.storage_nbytes = source.numel() * source.element_size()
        slot.host_offset = host_offset
        slot.cpu_tensor = source
        return slot

    def _plan(self, region):
        """Register one single-batch plan holding ``region``."""
        self.runtime._packed_batch_plans = [_PackedBatchPlan({region.dtype: region})]

    def test_materialize_staging_arena_reuses_its_own_storage(self):
        """A same-size arena keeps one allocation and its dtype views."""
        runtime = self.runtime
        device = torch.device("cpu")
        arena = runtime._materialize_staging_arena(0, 512, device)
        layout = {(torch.float32, 0, 16)}
        arena.dtype_views = {torch.float32: arena.raw_buffer.narrow(0, 0, 16).view(torch.float32)}
        arena.layout_signature = layout
        buffer_id = id(arena.raw_buffer)

        same = runtime._materialize_staging_arena(0, 512, device)

        self.assertIs(same, arena)
        self.assertEqual(id(same.raw_buffer), buffer_id)
        self.assertEqual(same.layout_signature, layout)
        self.assertEqual(set(same.dtype_views), {torch.float32})

    def test_materialize_staging_arena_grows_and_drops_cached_views(self):
        """Growing an arena invalidates its cached dtype and device views."""
        runtime = self.runtime
        device = torch.device("cpu")
        arena = runtime._materialize_staging_arena(0, 512, device)
        arena.dtype_views = {torch.float32: arena.raw_buffer.narrow(0, 0, 512).view(torch.float32)}
        arena.layout_signature = {(torch.float32, 0, 512)}
        runtime._packed_device_views = {(0, id(arena.raw_buffer), 1, 2, torch.float32, 0, 4, (4,)): "view"}

        grown = runtime._materialize_staging_arena(0, 1024, device)

        self.assertIs(grown, arena)
        self.assertEqual(grown.dtype_views, {})
        self.assertIsNone(grown.layout_signature)
        self.assertEqual(grown.raw_buffer.shape, (1024,))
        self.assertEqual(grown.raw_buffer.untyped_storage().size(), 1024)
        self.assertEqual(runtime._packed_device_views, {})

    def test_materialize_staging_arena_replaces_an_arena_from_another_device(self):
        """One arena slot never mixes buffers from different devices."""
        runtime = self.runtime
        arena = runtime._materialize_staging_arena(1, 512, torch.device("meta"))

        replaced = runtime._materialize_staging_arena(1, 512, torch.device("cpu"))

        self.assertIsNot(replaced, arena)
        self.assertEqual(replaced.raw_buffer.device.type, "cpu")
        self.assertEqual(replaced.dtype_views, {})

    def test_materialize_staging_arena_keeps_the_other_staging_index(self):
        """Resizing one arena leaves its partner untouched."""
        runtime = self.runtime
        device = torch.device("cpu")
        first = runtime._materialize_staging_arena(0, 512, device)
        second = runtime._materialize_staging_arena(1, 512, device)

        runtime._materialize_staging_arena(1, 1024, device)

        self.assertIs(runtime._staging_arenas[0], first)
        self.assertEqual(first.raw_buffer.untyped_storage().size(), 512)
        self.assertEqual(second.raw_buffer.shape, (1024,))

    def test_drop_packed_views_keeps_other_staging_buffers(self):
        """Cache eviction is scoped to the arena whose layout changed."""
        runtime = self.runtime
        runtime._packed_device_views = {
            (0, 1, 2, 3, torch.float32, 0, 4, (4,)): "view0",
            (1, 1, 2, 3, torch.float32, 0, 4, (4,)): "view1",
        }

        runtime._drop_packed_views(0)

        self.assertEqual(list(runtime._packed_device_views.values()), ["view1"])

    def test_require_staging_arena_rejects_an_unmaterialized_index(self):
        """Packed copies fail loudly when their arena was never allocated."""
        with self.assertRaisesRegex(RuntimeError, "staging arena 1 is not materialized"):
            self.runtime._require_staging_arena(1)

    def test_require_staging_arena_returns_the_materialized_arena(self):
        """A materialized arena is returned as-is."""
        runtime = self.runtime
        arena = runtime._materialize_staging_arena(0, 512, torch.device("cpu"))

        self.assertIs(runtime._require_staging_arena(0), arena)

    def test_copy_packed_to_device_moves_the_plan_region(self):
        """H2D copies the plan's host range into the arena dtype view."""
        runtime = self.runtime
        runtime._host_buffers = {torch.float32: torch.arange(4, dtype=torch.float32)}
        slot = self._host_packed_slot(torch.arange(4, dtype=torch.float32), 0)
        self._plan(_PackedBatchRegion(torch.float32, 0, 4, [slot]))
        arena = runtime._materialize_staging_arena(0, 512, torch.device("cpu"))
        arena.dtype_views = {torch.float32: arena.raw_buffer.narrow(0, 0, 16).view(torch.float32)}

        runtime._copy_packed_to_device(0, 0)

        self.assertTrue(torch.equal(arena.dtype_views[torch.float32], torch.arange(4, dtype=torch.float32)))

    def test_copy_packed_to_device_respects_the_region_offset(self):
        """Only the region's own host range is transferred."""
        runtime = self.runtime
        host_buffer = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        runtime._host_buffers = {torch.float32: host_buffer}
        self._plan(_PackedBatchRegion(torch.float32, 4, 4, [self._host_packed_slot(host_buffer.narrow(0, 4, 4), 4)]))
        arena = runtime._materialize_staging_arena(0, 512, torch.device("cpu"))
        arena.dtype_views = {torch.float32: arena.raw_buffer.narrow(0, 0, 16).view(torch.float32)}

        runtime._copy_packed_to_device(0, 0)

        self.assertTrue(torch.equal(
            arena.dtype_views[torch.float32],
            torch.tensor([5.0, 6.0, 7.0, 8.0]),
        ))

    def test_copy_packed_to_host_writes_the_region_back(self):
        """D2H publishes the arena view back into the persistent host buffer."""
        runtime = self.runtime
        host_buffer = torch.zeros(4, dtype=torch.float32)
        runtime._host_buffers = {torch.float32: host_buffer}
        self._plan(_PackedBatchRegion(torch.float32, 0, 4, [self._host_packed_slot(host_buffer, 0)]))
        arena = runtime._materialize_staging_arena(0, 512, torch.device("cpu"))
        arena.dtype_views = {torch.float32: arena.raw_buffer.narrow(0, 0, 16).view(torch.float32)}
        arena.dtype_views[torch.float32].copy_(torch.full((4,), 9.0))

        runtime._copy_packed_to_host(0, 0)

        self.assertTrue(torch.equal(host_buffer, torch.full((4,), 9.0)))

    def test_packed_copy_helpers_require_an_arena(self):
        """Both packed copy directions fail before their staging arena exists."""
        runtime = self.runtime
        runtime._host_buffers = {torch.float32: torch.zeros(4)}
        self._plan(_PackedBatchRegion(torch.float32, 0, 4, [self._host_packed_slot(torch.zeros(4), 0)]))

        with self.assertRaisesRegex(RuntimeError, "not materialized"):
            runtime._copy_packed_to_device(0, 0)
        with self.assertRaisesRegex(RuntimeError, "not materialized"):
            runtime._copy_packed_to_host(0, 0)

    def test_arena_and_plan_dataclasses_keep_their_layout(self):
        """Packed layout helpers expose the fields the runtime reads."""
        slot = SwapSlot(name="exp_avg", tensor=None)
        region = _PackedBatchRegion(torch.float32, 4, 8, [slot])
        plan = _PackedBatchPlan({torch.float32: region})
        arena = _StagingArena(torch.zeros(8, dtype=torch.uint8))

        self.assertEqual((region.dtype, region.host_offset, region.numel), (torch.float32, 4, 8))
        self.assertEqual(region.slots, [slot])
        self.assertIs(plan.regions[torch.float32], region)
        self.assertEqual(arena.dtype_views, {})
        self.assertIsNone(arena.layout_signature)

    def test_packed_plans_do_not_share_their_region_mapping(self):
        """Each plan owns an independent region mapping."""
        first = _PackedBatchPlan()
        second = _PackedBatchPlan()
        first.regions[torch.float32] = _PackedBatchRegion(torch.float32, 0, 4, [])

        self.assertEqual(second.regions, {})


class TestSwapRuntimePackedStep(unittest.TestCase):
    """End-to-end packed step bookkeeping."""

    @classmethod
    def setUpClass(cls):
        cls._allocators = _patch_allocators()
        cls.addClassCleanup(lambda: [patcher.stop() for patcher in cls._allocators])
        cls._device_sync = mock.patch.object(torch, "cuda", SimpleNamespace(synchronize=lambda *a, **k: None))
        cls._device_sync.start()
        cls.addClassCleanup(cls._device_sync.stop)

    def setUp(self):
        # Packed transfers reach for the device handle of the state tensors, so
        # the copy chain is driven through the runtime's own transfer hooks.
        self.runtime = _like_device_runtime(swap_times=1, min_numel=1)
        self.runtime._get_copy_stream = lambda: None
        self.runtime.current_stream = lambda: None

    def _prepared_slot(self, values):
        """Build a slot that went through host packing."""
        slot = SwapSlot(name="exp_avg", tensor=None, swappable=True, packed=True, state="pending")
        slot.dtype = values.dtype
        slot.numel = values.numel()
        slot.shape = tuple(values.shape)
        slot.storage_nbytes = values.numel() * values.element_size()
        slot.device = values.device
        slot.cpu_tensor = values.clone()
        self.runtime.prepare_packed_host([slot])
        return slot

    def _transfer_packed(self, batch_index, staging_index):
        """Run the packed H2D for one batch through the shared copy stream."""
        runtime = self.runtime
        runtime._copy_packed_to_device(batch_index, staging_index)
        runtime._packed_ready_events[batch_index] = None
        runtime._packed_offload_events[batch_index] = None

    def test_packed_step_round_trips_state_through_the_staging_arena(self):
        """H2D binds the arena view and D2H republishes the host buffer."""
        runtime = self.runtime
        slot = self._prepared_slot(torch.arange(4, dtype=torch.float32))
        expected = torch.arange(4, dtype=torch.float32)
        batch = [_unit(slot)]

        runtime.begin_packed_step([batch])
        self._transfer_packed(0, 0)
        runtime.activate_packed_batch(0, 0)

        device_tensor = slot.tensor
        self.assertEqual(slot.state, "device")
        self.assertIsNot(device_tensor, slot.cpu_tensor)
        # The live tensor is a view of the batch's staging arena region.
        self.assertEqual(device_tensor.untyped_storage().data_ptr(),
                         runtime._staging_arenas[0].raw_buffer.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(device_tensor, expected))

        device_tensor.add_(1)
        runtime._copy_packed_to_host(0, 0)
        runtime.finish_packed_offload(0)

        self.assertEqual(slot.state, "host")
        self.assertIs(slot.tensor, slot.cpu_tensor)
        self.assertTrue(torch.equal(slot.cpu_tensor, expected + 1))
        self.assertIsNone(slot.event)

    def test_end_packed_step_releases_arenas_and_step_local_state(self):
        """Teardown shrinks both arenas and clears every per-step mapping."""
        runtime = self.runtime
        slot = self._prepared_slot(torch.arange(4, dtype=torch.float32))

        runtime.begin_packed_step([[_unit(slot)]])
        self._transfer_packed(0, 0)
        runtime.activate_packed_batch(0, 0)
        runtime.end_packed_step()

        self.assertEqual(runtime._staging_arenas[0].raw_buffer.untyped_storage().size(), 0)
        self.assertEqual(runtime._staging_arenas[1].raw_buffer.untyped_storage().size(), 0)
        self.assertEqual(runtime._packed_batch_plans, [])
        self.assertEqual(runtime._packed_ready_events, {})
        self.assertEqual(runtime._packed_offload_events, {})
        self.assertEqual(runtime._packed_device_views, {})
        self.assertIsNone(runtime._packed_tail_event)

    def test_end_packed_step_copies_an_active_slot_back_before_teardown(self):
        """A slot still bound to staging storage is published before releasing it."""
        runtime = self.runtime
        slot = self._prepared_slot(torch.arange(4, dtype=torch.float32))
        expected = torch.arange(4, dtype=torch.float32)

        runtime.begin_packed_step([[_unit(slot)]])
        self._transfer_packed(0, 0)
        runtime.activate_packed_batch(0, 0)
        slot.tensor.add_(3)
        runtime.end_packed_step()

        self.assertEqual(slot.state, "host")
        self.assertIs(slot.tensor, slot.cpu_tensor)
        self.assertTrue(torch.equal(slot.cpu_tensor, expected + 3))

    def test_begin_packed_step_requires_the_host_layout(self):
        """A packed step cannot run before the host buffers exist."""
        slot = _fake_device_slot()
        batch = [_unit(slot)]

        with self.assertRaises(KeyError):
            self.runtime.begin_packed_step([batch])
        self.assertEqual(slot.state, "pending")

    def test_begin_packed_step_rejects_state_without_device_metadata(self):
        """A region without device metadata cannot materialize a staging arena."""
        runtime = self.runtime
        runtime._host_buffers = {torch.float32: torch.zeros(4)}
        slot = _fake_device_slot()
        runtime._build_packed_batch_plan = mock.Mock(return_value=_PackedBatchPlan())
        batch = [_unit(slot)]

        with self.assertRaisesRegex(RuntimeError, "no device-resident state metadata"):
            runtime.begin_packed_step([batch])

    def test_wait_hooks_report_missing_events(self):
        """The packed wait hooks fail loudly when their event was never recorded."""
        runtime = self.runtime

        with self.assertRaisesRegex(RuntimeError, "batch 0 has no ready event"):
            runtime.wait_packed_prefetch(0, 0)
        with self.assertRaisesRegex(RuntimeError, "batch 0 has no offload event"):
            runtime.wait_packed_offload(0)

    def test_supports_packed_pipeline_requires_a_single_fully_packed_layout(self):
        """Every swappable slot must carry a host mirror and a host-buffer dtype."""
        runtime = self.runtime
        runtime._host_buffers = {torch.float32: torch.zeros(4)}
        packed_slot = mock.Mock(
            swappable=True,
            packed=True,
            cpu_tensor=torch.zeros(4),
            dtype=torch.float32,
            device=torch.device("meta"),
        )

        self.assertTrue(runtime.supports_packed_pipeline([[_unit(packed_slot)]]))
        self.assertFalse(runtime.supports_packed_pipeline([]))

    def test_supports_packed_pipeline_returns_false_without_swappable_state(self):
        """A step without swappable state never uses the packed pipeline."""
        runtime = self.runtime
        runtime._host_buffers = {torch.float32: torch.zeros(4)}
        slot = mock.Mock(swappable=False, packed=False)

        self.assertFalse(runtime.supports_packed_pipeline([[_unit(slot)]]))

    def test_supports_packed_pipeline_rejects_a_partially_packed_step(self):
        """A packed slot already bound to a host view cannot fall back to per-tensor."""
        runtime = self.runtime
        runtime._host_buffers = {torch.float32: torch.zeros(4)}
        slot = mock.Mock(
            swappable=True,
            packed=True,
            cpu_tensor=None,
            dtype=torch.float32,
            device=torch.device("meta"),
        )

        with self.assertRaisesRegex(RuntimeError, "cannot fall back to per-tensor swap"):
            runtime.supports_packed_pipeline([[_unit(slot)]])

    def test_supports_packed_pipeline_reports_a_device_mismatch(self):
        """Packed slots spanning two devices fail instead of falling back."""
        runtime = self.runtime
        runtime._host_buffers = {torch.float32: torch.zeros(4)}
        first = mock.Mock(
            swappable=True, packed=True, cpu_tensor=torch.zeros(4),
            dtype=torch.float32, device=torch.device("meta:0"),
        )
        second = mock.Mock(
            swappable=True, packed=True, cpu_tensor=torch.zeros(4),
            dtype=torch.float32, device=torch.device("meta:1"),
        )

        with self.assertRaisesRegex(RuntimeError, "cannot fall back to per-tensor swap"):
            runtime.supports_packed_pipeline([[_unit(first)], [_unit(second)]])


class TestSwapRuntimeStreamHooks(unittest.TestCase):
    """Device handle, stream, event and copy-stream hooks."""

    def setUp(self):
        self.runtime = PipelineSwapRuntime(_DummyConfig())

    def test_device_handle_returns_the_torch_device_module(self):
        """The runtime resolves its accelerator namespace through Torch."""
        handle = SimpleNamespace(current_stream=lambda: "compute")
        with mock.patch.object(torch, "npu", handle, create=True):
            self.assertIs(self.runtime.device_handle(), handle)

    def test_device_handle_reports_a_missing_backend(self):
        """A build without the accelerator namespace fails with a clear message."""
        with mock.patch("hyper_parallel.core.optimizer.swap_optimizer_base._DEVICE_TYPE", "missing_backend"), \
                self.assertRaisesRegex(RuntimeError, "torch.missing_backend"):
            self.runtime.device_handle()

    def test_current_stream_and_new_stream_delegate_to_the_device_handle(self):
        """Streams always come from the same device handle."""
        handle = SimpleNamespace(current_stream=lambda: "compute", Stream=lambda: "copy")
        with mock.patch.object(torch, "npu", handle, create=True):
            self.assertEqual(self.runtime.current_stream(), "compute")
            self.assertEqual(self.runtime.new_stream(), "copy")

    def test_stream_context_wraps_none_and_real_streams(self):
        """A missing stream degrades to a no-op context manager."""
        self.assertEqual(type(self.runtime.stream_context(None)).__name__, "nullcontext")

        handle = SimpleNamespace(stream=lambda stream: ("stream", stream))
        with mock.patch.object(torch, "npu", handle, create=True):
            self.assertEqual(self.runtime.stream_context("copy"), ("stream", "copy"))

    def test_record_event_binds_the_requested_stream(self):
        """Events are recorded on the compute stream unless one is given."""
        handle = SimpleNamespace(Event=mock.Mock)
        with mock.patch.object(torch, "npu", handle, create=True):
            unbound = self.runtime.record_event()
            bound = self.runtime.record_event("stream")

        unbound.record.assert_called_once_with()
        bound.record.assert_called_once_with("stream")

    def test_wait_event_handles_empty_events_and_streams(self):
        """No event is a no-op; no stream means a host-side synchronize."""
        event = mock.Mock()

        self.runtime.wait_event(None, "stream")
        event.wait.assert_not_called()
        self.runtime.wait_event(event, None)
        self.runtime.wait_event(event, "stream")

        event.synchronize.assert_called_once_with()
        event.wait.assert_called_once_with("stream")

    def test_get_copy_stream_is_created_once(self):
        """The copy stream is created lazily and then reused."""
        self.runtime.new_stream = mock.Mock(return_value="copy-stream")

        first = self.runtime._get_copy_stream()
        second = self.runtime._get_copy_stream()

        self.assertIs(first, second)
        self.assertEqual(first, "copy-stream")
        self.runtime.new_stream.assert_called_once_with()

    def test_record_current_stream_event_uses_the_compute_stream(self):
        """The event that gates a copy is tied to the compute stream."""
        self.runtime.current_stream = mock.Mock(return_value="compute")
        self.runtime.record_event = mock.Mock(return_value="event")

        self.assertEqual(self.runtime._record_current_stream_event(), "event")
        self.runtime.record_event.assert_called_once_with("compute")


class TestModuleHelpers(unittest.TestCase):
    """Module-level slot/event iterators and state-key validation."""

    def test_iter_unique_slots_deduplicates_across_units(self):
        """The same slot reached through two units is yielded once."""
        slot = SwapSlot(name="exp_avg", tensor=object())
        other = SwapSlot(name="exp_avg_sq", tensor=object())

        self.assertEqual(list(_iter_unique_slots([_unit([slot, slot]), _unit([other])])), [slot, other])

    def test_iter_unique_slot_objects_keeps_the_first_alias(self):
        """Aliases share one tensor object, so only the first slot is returned."""
        shared = object()
        first = SwapSlot(name="first", tensor=shared)
        duplicate = SwapSlot(name="duplicate", tensor=shared)

        self.assertEqual(list(_iter_unique_slot_objects([duplicate, first])), [duplicate])

    def test_iter_unique_events_filters_empty_and_duplicate_events(self):
        """Only non-empty, distinct copy events are waited on."""
        shared_event = object()
        distinct_event = object()
        slots = [
            SwapSlot(name="first", tensor=object(), event=shared_event),
            SwapSlot(name="second", tensor=object(), event=shared_event),
            SwapSlot(name="idle", tensor=object(), event=None),
            SwapSlot(name="third", tensor=object(), event=distinct_event),
        ]

        events = list(_iter_unique_events(slots))

        self.assertEqual(len(events), 2)
        self.assertIs(events[0], shared_event)
        self.assertIs(events[1], distinct_event)

    def test_iter_unique_events_yields_nothing_without_events(self):
        """Slots that never entered a copy window produce no waits."""
        self.assertEqual(list(_iter_unique_events([SwapSlot(name="idle", tensor=object())])), [])
        self.assertEqual(list(_iter_unique_events([])), [])

    def test_validate_state_keys_normalizes_and_keeps_none(self):
        """Configured keys keep their order; master_param is allowed through."""
        self.assertEqual(validate_state_keys(("exp_avg_sq", "exp_avg")), ("exp_avg_sq", "exp_avg"))
        self.assertEqual(validate_state_keys(("master_param", "exp_avg")), ("master_param", "exp_avg"))
        self.assertEqual(validate_state_keys([]), ())
        self.assertIsNone(validate_state_keys(None))

    def test_validate_state_keys_rejects_unknown_logical_keys(self):
        """Only Adam/AdamW logical slots are accepted."""
        with self.assertRaisesRegex(ValueError, "only supports Adam/AdamW logical slots"):
            validate_state_keys(("exp_avg", "momentum_buffer"))


class _AdapterTestCase(unittest.TestCase):
    """Shared construction helpers for the adapter tests."""

    def _adapter(self, optimizer, **config):
        """Build an adapter over ``optimizer`` with a per-tensor runtime."""
        values = {"swap_times": 2, "packed_swap": False, "min_numel": 0, "state_keys": None}
        values.update(config)
        runtime = PipelineSwapRuntime(SimpleNamespace(**values))
        return OptimizerSwapAdapter(optimizer, runtime.config, runtime), runtime

    def _adam(self, params):
        """Build an Adam adapter with an empty slot registry."""
        optimizer = torch.optim.Adam(list(params), lr=0.01)
        adapter, runtime = self._adapter(optimizer)
        return optimizer, adapter, runtime


class TestSwapAdapterSlots(_AdapterTestCase):
    """Adapter slot construction, ordering and state-key selection."""

    def test_make_slot_requires_a_tensor_or_template(self):
        """A slot cannot be described without metadata."""
        _, adapter, _ = self._adam([torch.nn.Parameter(torch.ones(4))])

        with self.assertRaisesRegex(ValueError, "without a tensor or template"):
            adapter._make_slot("exp_avg", None)

    def test_make_slot_from_a_tensor_records_swappability_and_state(self):
        """A tensor-backed slot is device-resident and follows runtime eligibility."""
        _, adapter, runtime = self._adam([torch.nn.Parameter(torch.ones(4))])
        runtime.is_swappable_tensor = mock.Mock(return_value=True)

        slot = adapter._make_slot("exp_avg", torch.ones(8))

        self.assertEqual(slot.name, "exp_avg")
        self.assertTrue(slot.swappable)
        self.assertEqual(slot.state, "device")
        self.assertIsNone(slot.logical_tensor)
        runtime.is_swappable_tensor.assert_called_once_with(mock.ANY, 0)

    def test_make_slot_marks_packed_state_for_packed_runtimes(self):
        """Packed runtimes flag the slot and bind a DTensor logical tensor."""
        local = torch.ones(4)
        wrapper = SimpleNamespace(to_local=lambda: local, device=torch.device("cpu"))
        adapter, runtime = self._adapter(
            torch.optim.Adam([torch.nn.Parameter(torch.ones(4))], lr=0.01),
            packed_swap=True,
        )
        # The wrapper nests a real tensor, so only the local shard decides eligibility.
        runtime.is_swappable_tensor = mock.Mock(return_value=True)

        slot = adapter._make_slot("exp_avg", wrapper)

        self.assertTrue(slot.packed)
        self.assertTrue(slot.swappable)
        self.assertIs(slot.logical_tensor, wrapper)
        self.assertEqual(slot.device, local.device)

    def test_make_slot_keeps_plain_tensors_out_of_the_logical_binding(self):
        """A state tensor without a local-shard wrapper is packed but never rebound."""
        adapter, runtime = self._adapter(
            torch.optim.Adam([torch.nn.Parameter(torch.ones(4))], lr=0.01),
            packed_swap=True,
        )
        runtime.is_swappable_tensor = mock.Mock(return_value=True)

        slot = adapter._make_slot("exp_avg", torch.ones(4))

        self.assertTrue(slot.swappable)
        self.assertTrue(slot.packed)
        self.assertIsNone(slot.logical_tensor)

    def test_make_slot_from_a_template_starts_pending(self):
        """A packed candidate has no live tensor until its host buffer exists."""
        adapter, runtime = self._adapter(
            torch.optim.Adam([torch.nn.Parameter(torch.ones(4))], lr=0.01),
            packed_swap=True,
        )
        runtime.is_packable_template = mock.Mock(return_value=True)

        slot = adapter._make_slot("exp_avg", None, template=torch.ones(8))

        self.assertTrue(slot.swappable)
        self.assertTrue(slot.packed)
        self.assertIsNone(slot.tensor)
        self.assertEqual(slot.state, "pending")

    def test_build_slots_reuses_a_registered_slot(self):
        """A registered slot is reused while it still owns the state tensor."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, runtime = self._adam([param])
        runtime.config.state_keys = ("exp_avg",)
        tensor = torch.ones(4)
        registered = adapter._make_slot("exp_avg", tensor)
        adapter._slots[(id(param), "exp_avg")] = registered
        adapter.optimizer.state[param]["exp_avg"] = tensor

        slots = adapter._build_slots(param, {"exp_avg": tensor})

        self.assertEqual(slots, [registered])

    def test_build_slots_reuses_a_slot_bound_to_its_cpu_mirror(self):
        """A host-resident slot matching the CPU mirror survives an offload."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, runtime = self._adam([param])
        runtime.config.state_keys = ("exp_avg",)
        mirror = torch.zeros(4)
        slot = SwapSlot(name="exp_avg", tensor=torch.zeros(4), cpu_tensor=mirror, swappable=True, state="host")
        adapter._slots[(id(param), "exp_avg")] = slot
        adapter.optimizer.state[param]["exp_avg"] = mirror

        slots = adapter._build_slots(param, {"exp_avg": mirror})

        self.assertEqual(slots, [slot])

    def test_build_slots_replaces_a_slot_whose_tensor_changed(self):
        """A state tensor replaced outside the adapter gets a fresh slot."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, runtime = self._adam([param])
        runtime.config.state_keys = ("exp_avg",)
        stale = SwapSlot(name="exp_avg", tensor=torch.zeros(4), swappable=True)
        adapter._slots[(id(param), "exp_avg")] = stale
        replacement = torch.ones(4)
        adapter.optimizer.state[param]["exp_avg"] = replacement

        slots = adapter._build_slots(param, {"exp_avg": replacement})

        self.assertEqual(len(slots), 1)
        self.assertIsNot(slots[0], stale)
        self.assertIs(slots[0].tensor, replacement)
        self.assertIs(adapter._slots[(id(param), "exp_avg")], slots[0])

    def test_build_slots_creates_a_slot_for_unregistered_state(self):
        """State present but unregistered is wrapped in a fresh slot."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, runtime = self._adam([param])
        runtime.config.state_keys = ("exp_avg",)
        tensor = torch.ones(4)
        adapter.optimizer.state[param]["exp_avg"] = tensor

        slots = adapter._build_slots(param, {"exp_avg": tensor})

        self.assertEqual(len(slots), 1)
        self.assertIs(slots[0].tensor, tensor)
        self.assertIs(adapter._slots[(id(param), "exp_avg")], slots[0])

    def test_register_present_slots_adds_configured_state(self):
        """Configured state that already exists is registered for tracking."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, _ = self._adam([param])

        adapter._register_present_slots(param, {"exp_avg": torch.ones(4), "step": torch.zeros(())})

        self.assertEqual(set(adapter._slots), {(id(param), "exp_avg")})

    def test_register_present_slots_keeps_the_earlier_registration(self):
        """Registering twice is a no-op once a slot is already tracked."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, _ = self._adam([param])
        existing = SwapSlot(name="exp_avg", tensor=torch.zeros(4), swappable=True)
        adapter._slots[(id(param), "exp_avg")] = existing

        adapter._register_present_slots(param, {"exp_avg": torch.ones(4)})

        self.assertIs(adapter._slots[(id(param), "exp_avg")], existing)

    def test_ordered_slots_follow_parameter_and_key_order(self):
        """Slot ordering is stable and free of duplicates."""
        params = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
        optimizer, adapter, _ = self._adam(params)
        for param in params:
            for key in ADAM_STATE_KEYS:
                adapter._slots[(id(param), key)] = SwapSlot(name=key, tensor=torch.ones(4))
        # A second group adds its own parameter after the first group's slots.
        third = torch.nn.Parameter(torch.ones(4))
        optimizer.add_param_group({"params": [third]})
        for key in ADAM_STATE_KEYS:
            adapter._slots[(id(third), key)] = SwapSlot(name=key, tensor=torch.ones(4))

        slots = adapter._ordered_slots()

        self.assertEqual([slot.name for slot in slots], list(ADAM_STATE_KEYS) * 3)

    def test_configured_state_keys_drops_master_param_from_the_defaults(self):
        """master_param is not an Adam state key, so default selection skips it."""
        _, adapter, _ = self._adam([torch.nn.Parameter(torch.ones(4))])

        self.assertEqual(adapter._configured_state_keys(), ALL_STATE_KEYS)

    def test_configured_state_keys_honours_a_selection(self):
        """An explicit selection keeps its order."""
        _, adapter, runtime = self._adam([torch.nn.Parameter(torch.ones(4))])
        runtime.config.state_keys = ("exp_avg_sq", "exp_avg")

        self.assertEqual(adapter._configured_state_keys(), ("exp_avg_sq", "exp_avg"))

    def test_configured_state_keys_rejects_master_param_when_explicit(self):
        """master_param is only a valid key for optimizers that own a master copy."""
        _, adapter, runtime = self._adam([torch.nn.Parameter(torch.ones(4))])
        runtime.config.state_keys = ("exp_avg_sq", "master_param")

        with self.assertRaisesRegex(ValueError, "master_param"):
            adapter._configured_state_keys()

    def test_state_keys_for_param_reports_a_missing_configured_key(self):
        """A configured key absent from the parameter state is a hard error."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, runtime = self._adam([param])
        runtime.config.state_keys = ("exp_avg",)
        adapter.optimizer.state[param]["exp_avg_sq"] = torch.zeros(4)

        with self.assertRaisesRegex(ValueError, "is not present for parameter"):
            adapter._state_keys_for_param(param)

    def test_state_keys_for_param_returns_present_configured_keys(self):
        """Only keys that exist in the parameter state are selected."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, _ = self._adam([param])
        adapter.optimizer.state[param].update({"exp_avg": torch.zeros(4), "step": torch.zeros(())})

        self.assertEqual(adapter._state_keys_for_param(param), ("exp_avg",))

    def test_state_keys_for_param_falls_back_to_defaults_when_unconfigured(self):
        """Without an explicit selection, present default keys are returned."""
        param = torch.nn.Parameter(torch.ones(4))
        _, adapter, _ = self._adam([param])
        adapter.optimizer.state[param].update({
            "exp_avg": torch.zeros(4),
            "exp_avg_sq": torch.zeros(4),
            "max_exp_avg_sq": torch.zeros(4),
        })

        self.assertEqual(
            adapter._state_keys_for_param(param),
            ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"),
        )

    def test_slot_tensor_prefers_an_active_slot_and_falls_back(self):
        """Only device-resident swappable slots shadow the optimizer state."""
        fallback = torch.zeros(4)
        active = torch.ones(4)
        slot = SwapSlot(name="exp_avg", tensor=active, swappable=True, state="device")
        unit = _unit(slot)

        self.assertIs(OptimizerSwapAdapter._slot_tensor(unit, "exp_avg", fallback), active)

        slot.state = "h2d"
        self.assertIs(OptimizerSwapAdapter._slot_tensor(unit, "exp_avg", fallback), fallback)
        self.assertIs(OptimizerSwapAdapter._slot_tensor(unit, "exp_avg_sq", fallback), fallback)

    def test_slot_tensor_ignores_slots_without_a_live_tensor(self):
        """A pending or unswappable slot always falls back to the state tensor."""
        fallback = torch.zeros(4)
        pending = SwapSlot(name="exp_avg", tensor=None, swappable=True, state="pending")
        tiny = SwapSlot(name="exp_avg", tensor=torch.ones(4), swappable=False, state="device")

        self.assertIs(OptimizerSwapAdapter._slot_tensor(_unit(pending), "exp_avg", fallback), fallback)
        self.assertIs(OptimizerSwapAdapter._slot_tensor(_unit(tiny), "exp_avg", fallback), fallback)


class TestSwapAdapterStepPreparation(_AdapterTestCase):
    """Unit collection, flag validation and state initialization."""

    def test_prepare_step_rejects_closure_or_extra_arguments(self):
        """The swap pipeline never runs a Torch closure."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, _ = self._adapter(torch.optim.Adam([param], lr=0.01))

        with self.assertRaisesRegex(ValueError, "does not support closure or extra arguments"):
            adapter.prepare_step(lambda: None)

    def test_prepare_step_skips_parameters_without_dense_gradients(self):
        """Parameters without a gradient contribute no update unit."""
        params = [torch.nn.Parameter(torch.ones(8)) for _ in range(2)]
        adapter, _ = self._adapter(torch.optim.Adam(params, lr=0.01))
        params[0].grad = torch.ones_like(params[0])

        units = adapter.prepare_step()["units"]

        self.assertEqual(len(units), 1)
        self.assertIs(units[0].param, params[0])

    def test_prepare_step_rejects_sparse_gradients(self):
        """Sparse gradients cannot be replayed through the functional Adam."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, _ = self._adapter(torch.optim.Adam([param], lr=0.01))
        param.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (8,))

        with self.assertRaisesRegex(ValueError, "only supports dense Adam/AdamW gradients"):
            adapter.prepare_step()

    def test_prepare_step_registers_slots_for_collected_units(self):
        """Unit slots are created while initializing lazy optimizer state."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, _ = self._adapter(torch.optim.Adam([param], lr=0.01))
        param.grad = torch.ones_like(param)

        units = adapter.prepare_step()["units"]

        self.assertEqual([unit.adapter_index for unit in units], [0])
        self.assertEqual({slot.name for slot in units[0].slots}, set(ADAM_STATE_KEYS))
        for slot in units[0].slots:
            self.assertIs(adapter._slots[(id(param), slot.name)], slot)

    def test_prepare_step_reports_the_group_index_of_each_unit(self):
        """Units carry the index of the group whose parameters they came from."""
        first = torch.nn.Parameter(torch.ones(8))
        second = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([{"params": [first]}, {"params": [second], "lr": 0.02}], lr=0.01)
        adapter, _ = self._adapter(optimizer)
        first.grad = torch.ones_like(first)
        second.grad = torch.ones_like(second)

        units = adapter.prepare_step()["units"]

        self.assertEqual([unit.adapter_index for unit in units], [0, 1])
        self.assertEqual([id(unit.param) for unit in units], [id(first), id(second)])

    def test_validate_rejects_unsupported_flag_combinations(self):
        """Flags that change kernel selection cannot be preserved through swap."""
        cases = [
            (torch.optim.Adam, {"foreach": True}, "foreach=True"),
            (torch.optim.Adam, {"fused": True}, "fused=True"),
            (torch.optim.Adam, {"differentiable": True}, "differentiable=True"),
            (torch.optim.Adam, {"capturable": True}, "capturable=True"),
            (torch.optim.AdamW, {"capturable": True}, "capturable=True"),
        ]
        for optimizer_type, kwargs, message in cases:
            with self.subTest(optimizer=optimizer_type.__name__, flag=message):
                param = torch.nn.Parameter(torch.ones(8))
                adapter, _ = self._adapter(optimizer_type([param], **kwargs))

                with self.assertRaisesRegex(ValueError, message):
                    adapter.validate()

    def test_validate_accepts_the_defaults(self):
        """Default Adam and AdamW groups pass validation."""
        param = torch.nn.Parameter(torch.ones(8))
        for optimizer_type in (torch.optim.Adam, torch.optim.AdamW):
            with self.subTest(optimizer=optimizer_type.__name__):
                adapter, _ = self._adapter(optimizer_type([param], lr=0.01))
                adapter.validate()

    def test_torch_adapters_select_by_optimizer_type(self):
        """Adam and AdamW pick different adapters even though AdamW subclasses Adam."""
        param = torch.nn.Parameter(torch.ones(8))
        adam = torch.optim.Adam([param], lr=0.01)
        adamw = torch.optim.AdamW([param], lr=0.01)

        self.assertTrue(TorchNativeAdamAdapter.matches(adam))
        self.assertFalse(TorchNativeAdamAdapter.matches(adamw))
        self.assertTrue(TorchNativeAdamWAdapter.matches(adamw))
        self.assertFalse(TorchNativeAdamWAdapter.matches(adam))

    def test_init_param_state_keeps_swappable_state_on_the_host(self):
        """Legacy swap initializes a zero CPU mirror and empty device placeholders."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, runtime = self._adapter(torch.optim.Adam([param], lr=0.01))
        runtime.is_swappable_tensor = mock.Mock(return_value=True)
        runtime.make_zero_cpu_tensor_like = mock.Mock(return_value=torch.zeros(8))
        runtime.release_device_storage = mock.Mock()

        adapter._init_param_state(param, torch.ones(8), {"amsgrad": False})

        state = adapter.optimizer.state[param]
        self.assertEqual(set(state), {"step", "exp_avg", "exp_avg_sq"})
        self.assertEqual(state["step"].device.type, "cpu")
        slot = adapter._slots[(id(param), "exp_avg")]
        self.assertEqual(slot.state, "host")
        self.assertTrue(torch.equal(slot.cpu_tensor, torch.zeros(8)))
        self.assertEqual(runtime.release_device_storage.call_count, 2)

    def test_init_param_state_zeroes_state_when_swap_is_not_eligible(self):
        """Resident state is materialized as a zero tensor on the parameter device."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, runtime = self._adapter(torch.optim.Adam([param], lr=0.01), min_numel=1024)

        adapter._init_param_state(param, torch.ones(8), {"amsgrad": False})

        state = adapter.optimizer.state[param]
        self.assertEqual(state["exp_avg"].device.type, "cpu")
        self.assertEqual(torch.count_nonzero(state["exp_avg"]).item(), 0)
        self.assertNotIn((id(param), "exp_avg"), adapter._slots)
        self.assertFalse(runtime.packed_enabled)

    def test_init_param_state_is_idempotent_for_existing_state(self):
        """State that already exists is never re-initialized."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, _ = self._adapter(torch.optim.Adam([param], lr=0.01))
        existing = torch.ones(8)
        adapter.optimizer.state[param]["exp_avg"] = existing

        adapter._init_param_state(param, torch.ones(8), {"amsgrad": False})

        self.assertIs(adapter.optimizer.state[param]["exp_avg"], existing)

    def test_init_param_state_skips_step_for_new_adamw(self):
        """hyper-parallel's AdamW owns its step counter, so none is created."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=0.01)
        adapter = object.__new__(OptimizerSwapAdapter)
        adapter.optimizer = optimizer
        adapter._slots = {}
        adapter.is_new_adamw = True
        adapter.runtime = PipelineSwapRuntime(_DummyConfig())
        adapter.config = adapter.runtime.config

        adapter._init_param_state(param, None, {"amsgrad": False})

        self.assertNotIn("step", optimizer.state[param])
        self.assertEqual(set(optimizer.state[param]), set(ADAM_STATE_KEYS))

    def test_init_param_state_adds_amsgrad_state_only_when_requested(self):
        """amsgrad adds the running maximum beside the two moments."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=0.01)
        adapter = object.__new__(OptimizerSwapAdapter)
        adapter.optimizer = optimizer
        adapter._slots = {}
        adapter.is_new_adamw = True
        adapter.runtime = PipelineSwapRuntime(_DummyConfig())
        adapter.config = adapter.runtime.config

        adapter._init_param_state(param, None, {"amsgrad": True})

        self.assertEqual(set(optimizer.state[param]), set(ALL_STATE_KEYS))

    def test_publish_packed_state_is_a_noop_for_per_tensor_runtimes(self):
        """Legacy runtimes leave the optimizer state mapping untouched."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        adapter, _ = self._adapter(optimizer)
        slot = SwapSlot(name="exp_avg", tensor=torch.ones(8), swappable=True, state="host")
        slot.cpu_tensor = torch.zeros(8)
        adapter._slots[(id(param), "exp_avg")] = slot

        adapter.publish_packed_state()

        self.assertNotIn("exp_avg", optimizer.state[param])

    def test_publish_packed_state_creates_missing_state_mappings(self):
        """A packed host view is published even when no mapping exists yet."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        adapter, _ = self._adapter(optimizer, packed_swap=True, min_numel=1)
        slot = SwapSlot(name="exp_avg", tensor=None, swappable=True, packed=True, state="host")
        slot.cpu_tensor = torch.zeros(8)
        adapter._slots[(id(param), "exp_avg")] = slot
        optimizer.state.pop(param, None)

        adapter.publish_packed_state()

        self.assertIs(optimizer.state[param]["exp_avg"], slot.cpu_tensor)

    def test_publish_packed_state_skips_unpacked_slots(self):
        """Only packed host views replace the optimizer state tensors."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        adapter, _ = self._adapter(optimizer, packed_swap=True, min_numel=1)
        slot = SwapSlot(name="exp_avg", tensor=torch.ones(8), swappable=True, packed=False, state="host")
        slot.cpu_tensor = torch.zeros(8)
        adapter._slots[(id(param), "exp_avg")] = slot
        published = torch.ones(8)
        optimizer.state[param]["exp_avg"] = published

        adapter.publish_packed_state()

        self.assertIs(optimizer.state[param]["exp_avg"], published)

    def test_strip_swappable_state_splits_adam_buffers(self):
        """Checkpoint loading removes swap-managed keys from the Torch state dict."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, _ = self._adapter(torch.optim.Adam([param], lr=0.01))
        state_dict = {
            "state": {
                id(param): {
                    "step": torch.zeros(()),
                    "exp_avg": torch.ones(8),
                    "exp_avg_sq": torch.full((8,), 2.0),
                },
            },
            "param_groups": [{"params": [id(param)]}],
        }

        stripped, removed = adapter.strip_swappable_state(state_dict)

        self.assertEqual(set(stripped["state"][id(param)]), {"step"})
        self.assertEqual(set(removed[id(param)]), set(ADAM_STATE_KEYS))
        # The original checkpoint is left untouched for the second load phase.
        self.assertEqual(set(state_dict["state"][id(param)]), {"step", *ADAM_STATE_KEYS})

    def test_initial_slots_discovers_pre_existing_state(self):
        """State materialized before wrapping is discovered for the first offload."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        optimizer.state[param]["exp_avg"] = torch.ones(8)
        optimizer.state[param]["exp_avg_sq"] = torch.ones(8)
        adapter, _ = self._adapter(optimizer)

        slots = adapter.initial_slots()

        self.assertEqual({slot.name for slot in slots}, set(ADAM_STATE_KEYS))
        self.assertEqual(
            {id(slot.tensor) for slot in slots},
            {id(optimizer.state[param]["exp_avg"]), id(optimizer.state[param]["exp_avg_sq"])},
        )

    def test_initial_slots_ignores_parameters_without_state(self):
        """Lazy state stays lazy until the first optimizer step."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, _ = self._adapter(torch.optim.Adam([param], lr=0.01))

        self.assertEqual(tuple(adapter.initial_slots()), ())


if __name__ == "__main__":
    unittest.main()
