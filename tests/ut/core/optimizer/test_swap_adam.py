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
"""Unit tests for the Adam/AdamW swap adapters."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

from hyper_parallel.core.optimizer.swap_adam import (
    ADAM_STATE_KEYS,
    AdamSwapAdapter,
    TorchNativeAdamAdapter,
    TorchNativeAdamWAdapter,
    TorchNewAdamWAdapter,
    build_adam_swap_adapter,
)
from hyper_parallel.core.optimizer.swap_optimizer_base import (
    PipelineSwapRuntime,
    SwapSlot,
    UpdateUnit,
    _slot_tensor,
)

# Adam state keys created by the lazy per-tensor initialization path.
LAZY_STATE_KEYS = ("exp_avg", "exp_avg_sq")
# Every logical key the adapter knows about, including the optional amsgrad moment.
ALL_STATE_KEYS = LAZY_STATE_KEYS + ("max_exp_avg_sq",)


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


def _patch_allocators():
    """Swap ``pin_memory`` allocations for pageable host memory on CPU-only hosts."""

    def _unpinned(node):
        def _allocate(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return node(*args, **kwargs)

        return _allocate

    patchers = [
        mock.patch.object(torch, "empty", _unpinned(torch.empty)),
        mock.patch.object(torch, "empty_like", _unpinned(torch.empty_like)),
        mock.patch.object(torch, "zeros", _unpinned(torch.zeros)),
        mock.patch.object(torch, "zeros_like", _unpinned(torch.zeros_like)),
    ]
    for patcher in patchers:
        patcher.start()
        yield patcher
    for patcher in patchers:
        patcher.stop()

class _AdapterTestCase(unittest.TestCase):
    """Shared construction helpers for the adapter tests."""

    def _adapter(self, optimizer, **config):
        """Build an adapter over ``optimizer`` with a per-tensor runtime."""
        values = {"swap_times": 2, "packed_swap": False, "min_numel": 0, "state_keys": None}
        values.update(config)
        runtime = PipelineSwapRuntime(SimpleNamespace(**values))
        return AdamSwapAdapter(optimizer, runtime.config, runtime), runtime

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
            for key in LAZY_STATE_KEYS:
                adapter._slots[(id(param), key)] = SwapSlot(name=key, tensor=torch.ones(4))
        # A second group adds its own parameter after the first group's slots.
        third = torch.nn.Parameter(torch.ones(4))
        optimizer.add_param_group({"params": [third]})
        for key in LAZY_STATE_KEYS:
            adapter._slots[(id(third), key)] = SwapSlot(name=key, tensor=torch.ones(4))

        slots = adapter._ordered_slots()

        self.assertEqual([slot.name for slot in slots], list(LAZY_STATE_KEYS) * 3)

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

        self.assertIs(_slot_tensor(unit, "exp_avg", fallback), active)

        slot.state = "h2d"
        self.assertIs(_slot_tensor(unit, "exp_avg", fallback), fallback)
        self.assertIs(_slot_tensor(unit, "exp_avg_sq", fallback), fallback)

    def test_slot_tensor_ignores_slots_without_a_live_tensor(self):
        """A pending or unswappable slot always falls back to the state tensor."""
        fallback = torch.zeros(4)
        pending = SwapSlot(name="exp_avg", tensor=None, swappable=True, state="pending")
        tiny = SwapSlot(name="exp_avg", tensor=torch.ones(4), swappable=False, state="device")

        self.assertIs(_slot_tensor(_unit(pending), "exp_avg", fallback), fallback)
        self.assertIs(_slot_tensor(_unit(tiny), "exp_avg", fallback), fallback)


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
        self.assertEqual({slot.name for slot in units[0].slots}, set(LAZY_STATE_KEYS))
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
        adapter = object.__new__(AdamSwapAdapter)
        adapter.optimizer = optimizer
        adapter._slots = {}
        adapter.is_new_adamw = True
        adapter.runtime = PipelineSwapRuntime(_DummyConfig())
        adapter.config = adapter.runtime.config

        adapter._init_param_state(param, None, {"amsgrad": False})

        self.assertNotIn("step", optimizer.state[param])
        self.assertEqual(set(optimizer.state[param]), set(LAZY_STATE_KEYS))

    def test_init_param_state_adds_amsgrad_state_only_when_requested(self):
        """amsgrad adds the running maximum beside the two moments."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.AdamW([param], lr=0.01)
        adapter = object.__new__(AdamSwapAdapter)
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
        self.assertEqual(set(removed[id(param)]), set(LAZY_STATE_KEYS))
        # The original checkpoint is left untouched for the second load phase.
        self.assertEqual(set(state_dict["state"][id(param)]), {"step", *LAZY_STATE_KEYS})

    def test_strip_swappable_state_shares_checkpoint_tensor_storage(self):
        """Stripping must not duplicate optimizer tensors on host memory."""
        param = torch.nn.Parameter(torch.ones(8))
        adapter, _ = self._adapter(torch.optim.Adam([param], lr=0.01))
        exp_avg = torch.ones(8)
        state_dict = {
            "state": {id(param): {"step": torch.zeros(()), "exp_avg": exp_avg}},
            "param_groups": [{"params": [id(param)], "lr": 0.01}],
        }

        stripped, removed = adapter.strip_swappable_state(state_dict)

        # The removed buffer is the checkpoint's own storage, not a copy.
        self.assertIs(removed[id(param)]["exp_avg"], exp_avg)
        # Non-swappable entries stay aliased too: no tensor is cloned at all.
        self.assertIs(stripped["state"][id(param)]["step"], state_dict["state"][id(param)]["step"])
        # Container skeleton is fresh, so popping cannot touch the caller's dict.
        self.assertIsNot(stripped, state_dict)
        self.assertIsNot(stripped["state"], state_dict["state"])
        self.assertIsNot(stripped["param_groups"], state_dict["param_groups"])
        self.assertIsNot(stripped["param_groups"][0], state_dict["param_groups"][0])
        self.assertEqual(set(state_dict["state"][id(param)]), {"step", "exp_avg"})

    def test_initial_slots_discovers_pre_existing_state(self):
        """State materialized before wrapping is discovered for the first offload."""
        param = torch.nn.Parameter(torch.ones(8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        optimizer.state[param]["exp_avg"] = torch.ones(8)
        optimizer.state[param]["exp_avg_sq"] = torch.ones(8)
        adapter, _ = self._adapter(optimizer)

        slots = adapter.initial_slots()

        self.assertEqual({slot.name for slot in slots}, set(LAZY_STATE_KEYS))
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
