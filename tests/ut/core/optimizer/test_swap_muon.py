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
"""Unit tests for the Muon momentum-only swap adapter."""

import unittest
from unittest import mock

import torch

from hyper_parallel.core.optimizer import Muon, SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.swap_muon import (
    MUON_STATE_KEYS,
    MuonSwapAdapter,
    MuonSwapUnit,
    build_muon_swap_adapter,
    swap_muon,
)
from hyper_parallel.core.optimizer.swap_optimizer_base import (
    PipelineSwapRuntime,
    SwapOptimizer as CoreSwapOptimizer,
    SwapSlot,
    UpdateUnit,
)


def _muon(params, **kwargs):
    """Build a bare CPU Muon optimizer over ``params``."""
    return Muon(list(params), lr=0.01, **kwargs)


def _swap_config(**overrides):
    """Build a per-tensor swap config, the only mode Muon supports."""
    values = {"swap_times": 2, "min_numel": 1, "packed_swap": False}
    values.update(overrides)
    return SwapOptimizerConfig(**values)


def _make_assignment(params, is_shard):
    """Build a stand-in HSDP assignment with the fields the adapter reads."""
    assignment = mock.Mock()
    assignment.is_shard = is_shard
    assignment.owned_params = list(params)
    return assignment


def _install_schedule(optimizer, assignments=None, no_comm=None):
    """Install Muon's private schedule metadata layout on ``optimizer``.

    Muon builds ``_hsdp_assignment_batches[group]`` as ``{"no_comm": [...],
    "batch_groups": [{"sub_batches": [...]}]}``; the adapter reads that shape
    directly, so tests drive it the same way instead of stubbing adapter code.
    """
    optimizer._hsdp_assignment_batches = {
        0: {
            "no_comm": list(no_comm or []),
            "batch_groups": [{"sub_batches": list(assignments or [])}],
        }
    }


def _adapter(optimizer, **config):
    """Build a Muon adapter with a per-tensor runtime."""
    from types import SimpleNamespace

    values = {"swap_times": 2, "packed_swap": False, "min_numel": 1, "state_keys": None}
    values.update(config)
    runtime = PipelineSwapRuntime(SimpleNamespace(**values))
    return MuonSwapAdapter(optimizer, runtime.config, runtime), runtime


class TestMuonSwapDispatch(unittest.TestCase):
    """Public entry point dispatch and configuration validation."""

    def test_muon_is_dispatched_to_the_muon_family(self):
        """``swap_optimizer`` recognizes a Muon leaf and returns the core wrapper."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        wrapped = swap_optimizer(_muon([param]), _swap_config())

        self.assertIsInstance(wrapped, CoreSwapOptimizer)
        self.assertIsInstance(wrapped.adapter, MuonSwapAdapter)
        self.assertTrue(getattr(wrapped, "_is_swap_optimizer", False))

    def test_packed_swap_is_rejected_for_muon(self):
        """Muon swaps tensor by tensor, so packed staging must fail fast."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        with self.assertRaisesRegex(ValueError, "packed_swap=False"):
            swap_optimizer(_muon([param]), SwapOptimizerConfig(packed_swap=True))

    def test_swap_muon_rejects_a_non_muon_optimizer(self):
        """The Muon factory only accepts a Muon leaf."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        with self.assertRaisesRegex(ValueError, "only supports"):
            swap_muon(torch.optim.SGD([param], lr=0.01), _swap_config())

    def test_build_adapter_rejects_a_non_muon_optimizer(self):
        """Adapter construction fails for a non-Muon optimizer."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = torch.optim.Adam([param], lr=0.01)
        runtime = PipelineSwapRuntime(_swap_config())

        with self.assertRaisesRegex(ValueError, "only supports"):
            build_muon_swap_adapter(optimizer, _swap_config(), runtime)

    def test_adam_optimizers_still_dispatch_to_adam(self):
        """Adding the Muon branch does not change Adam/AdamW dispatch."""
        from hyper_parallel.core.optimizer.swap_adam import AdamSwapAdapter

        param = torch.nn.Parameter(torch.ones(8))
        wrapped = swap_optimizer(torch.optim.AdamW([param], lr=0.01), SwapOptimizerConfig(min_numel=1))

        self.assertIsInstance(wrapped.adapter, AdamSwapAdapter)


class TestMuonSwapAdapter(unittest.TestCase):
    """Adapter-level schedule construction and validation."""

    def test_default_state_keys_only_contain_momentum(self):
        """Muon's only swappable persistent state is the momentum buffer."""
        self.assertEqual(MuonSwapAdapter.default_state_keys(), MUON_STATE_KEYS)
        self.assertEqual(MUON_STATE_KEYS, ("momentum_buffer",))

    def test_unsupported_state_key_is_rejected_for_muon(self):
        """An Adam-only key is not a valid Muon swap selection."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        adapter, runtime = _adapter(_muon([param]))
        runtime.config.state_keys = ("exp_avg",)

        with self.assertRaisesRegex(ValueError, "not available for"):
            adapter.validate()

    def test_supported_state_key_selection_is_accepted(self):
        """Selecting Muon's own state key explicitly stays supported."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        adapter, runtime = _adapter(_muon([param]))
        runtime.config.state_keys = ("momentum_buffer",)

        self.assertIsNone(adapter.validate())

    def test_validate_rejects_non_contiguous_momentum_state(self):
        """Momentum state that is not contiguous cannot be swapped."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([param])
        adapter, _ = _adapter(optimizer)
        buffer = torch.zeros(16, 16).t()
        self.assertFalse(buffer.is_contiguous())
        optimizer.state[param]["momentum_buffer"] = buffer

        with self.assertRaisesRegex(ValueError, "contiguous momentum state"):
            adapter.validate()

    def test_validate_accepts_pristine_state(self):
        """An optimizer with no materialized state validates cleanly."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        adapter, _ = _adapter(_muon([param]))

        self.assertIsNone(adapter.validate())

    def test_prepare_step_rejects_extra_arguments(self):
        """Muon swap steps accept neither a closure nor kwargs."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        adapter, _ = _adapter(_muon([param]))

        with self.assertRaisesRegex(ValueError, "does not support closure"):
            adapter.prepare_step(lambda: None)

    def test_prepare_step_materializes_momentum_state_for_every_grad_param(self):
        """First-step momentum is created before the schedule runs.

        CPU tensors are never swap-eligible, so on a CPU-only host the runtime
        keeps them resident; the host-mirror/device-release lifecycle is covered
        by the accelerator tests in ``tests/torch/swap_optimizer``.
        """
        params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(2)]
        optimizer = _muon(params)
        adapter, _ = _adapter(optimizer)
        for param in params:
            param.grad = torch.ones_like(param)

        context = adapter.prepare_step()

        self.assertTrue(all("momentum_buffer" in optimizer.state[param] for param in params))
        self.assertTrue(all(
            torch.count_nonzero(optimizer.state[param]["momentum_buffer"]) == 0 for param in params
        ))
        self.assertEqual(len(context["units"]), 1)
        self.assertEqual(context["units"][0].kind, "no_comm")

    def test_swappable_momentum_slot_starts_host_resident(self):
        """An eligible momentum slot is registered with a host mirror and no device storage."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([param])
        adapter, runtime = _adapter(optimizer)
        param.grad = torch.ones_like(param)
        runtime.is_swappable_tensor = mock.Mock(return_value=True)
        runtime.release_device_storage = mock.Mock()

        adapter.prepare_step()

        slot = adapter._slots[(id(param), "momentum_buffer")]
        self.assertEqual(slot.state, "host")
        self.assertIsNotNone(slot.cpu_tensor)
        self.assertEqual(runtime.release_device_storage.call_count, 1)
        self.assertTrue(torch.count_nonzero(slot.cpu_tensor) == 0)

    def test_prepare_step_skips_params_without_gradients(self):
        """A parameter without a gradient keeps Muon's skip semantics."""
        with_grad = torch.nn.Parameter(torch.ones(8, 8))
        without_grad = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([with_grad, without_grad])
        adapter, _ = _adapter(optimizer)
        with_grad.grad = torch.ones_like(with_grad)

        context = adapter.prepare_step()

        units = context["units"]
        self.assertEqual([id(p) for unit in units for p in unit.params], [id(with_grad)])
        self.assertEqual(len(tuple(adapter.all_slots())), 1)

    def test_prepare_step_advances_every_group_step(self):
        """Each group's step counter advances exactly once per step."""
        params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(2)]
        optimizer = _muon(params)
        adapter, _ = _adapter(optimizer)
        for param in params:
            param.grad = torch.ones_like(param)

        adapter.prepare_step()

        self.assertEqual([group["step"] for group in optimizer.param_groups], [1])

    def test_units_split_by_shape_group(self):
        """Different shapes form separate atomic units, matching the NS batching."""
        square = torch.nn.Parameter(torch.ones(8, 8))
        wide = torch.nn.Parameter(torch.ones(4, 12))
        optimizer = _muon([square, wide])
        adapter, _ = _adapter(optimizer)
        square.grad = torch.ones_like(square)
        wide.grad = torch.ones_like(wide)

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual(len(units), 2)
        self.assertTrue(all(unit.kind == "no_comm" for unit in units))
        self.assertEqual([len(unit.params) for unit in units], [1, 1])

    def test_empty_no_comm_schedule_still_yields_no_units(self):
        """A group with no gradient-bearing params produces no unit at all."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        adapter, _ = _adapter(_muon([param]))

        context = adapter.prepare_step()

        self.assertEqual(context["units"], [])

    def test_hsdp_assignment_produces_one_unit_per_shard_batch(self):
        """A sharded assignment keeps its communication batch as the unit boundary."""
        params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(2)]
        optimizer = _muon(params)
        adapter, _ = _adapter(optimizer)
        for param in params:
            param.grad = torch.ones_like(param)

        assignment = _make_assignment(params, is_shard=True)
        _install_schedule(optimizer, assignments=[assignment], no_comm=[])

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "hsdp")
        self.assertIs(units[0].hsdp_assign, assignment)

    def test_hsdp_assignment_without_local_owner_still_occupies_the_schedule(self):
        """A rank with nothing owned keeps the assignment slot, so collectives align."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([param])
        adapter, _ = _adapter(optimizer)
        param.grad = torch.ones_like(param)

        assignment = _make_assignment([], is_shard=True)
        _install_schedule(optimizer, assignments=[assignment], no_comm=[])

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].params, [])
        self.assertEqual(units[0].slots, [])
        # the empty unit still owns the assignment's broadcast flush
        self.assertTrue(units[0].flushes_broadcast)

    def test_unsharded_assignment_uses_the_inner_ns_batch_boundaries(self):
        """An unsharded assignment splits into the same batches NS would use."""
        params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(3)]
        optimizer = _muon(params)
        adapter, _ = _adapter(optimizer)
        for param in params:
            param.grad = torch.ones_like(param)

        assignment = _make_assignment(params, is_shard=False)
        _install_schedule(optimizer, assignments=[assignment], no_comm=[])
        # a single-rank shard topology keeps the NS split at shard_size=1
        optimizer._get_shard_info = mock.Mock(return_value=((1,), (0,), (None,), 1))
        optimizer._split_into_memory_safe_batches = mock.Mock(
            return_value=[[params[0]], [params[1], params[2]]]
        )

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual([[p.shape for p in unit.params] for unit in units], [[(8, 8)], [(8, 8), (8, 8)]])
        # exactly one broadcast flush, at the assignment boundary
        self.assertEqual([unit.flushes_broadcast for unit in units], [False, True])

    def test_repeated_prepare_steps_reuse_the_same_slot(self):
        """A second step reuses the registered slot instead of rebuilding it."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([param])
        adapter, _ = _adapter(optimizer)
        param.grad = torch.ones_like(param)
        adapter.prepare_step()
        first = adapter._slots[(id(param), "momentum_buffer")]

        adapter.prepare_step()

        self.assertIs(adapter._slots[(id(param), "momentum_buffer")], first)
        self.assertEqual([group["step"] for group in optimizer.param_groups], [2])


class TestMuonSwapUnitContract(unittest.TestCase):
    """Muon units satisfy the generic runtime contract without Adam fields."""

    def test_muon_unit_exposes_slots_without_param_or_grad(self):
        """The runtime only needs ``slots``; a Muon unit carries no Adam fields."""
        slot = SwapSlot(name="momentum_buffer", tensor=torch.ones(4))
        unit = MuonSwapUnit(group_index=0, kind="no_comm", params=[], slots=[slot])

        self.assertEqual(unit.slots, [slot])
        self.assertFalse(hasattr(unit, "param"))
        self.assertFalse(hasattr(unit, "grad"))

    def test_runtime_partitions_muon_units(self):
        """``PipelineSwapRuntime.partition`` accepts Muon units unchanged."""
        runtime = PipelineSwapRuntime(_swap_config(swap_times=2, min_numel=0))
        units = [
            MuonSwapUnit(
                group_index=0,
                kind="no_comm",
                params=[],
                slots=[SwapSlot(
                    name="momentum_buffer",
                    tensor=torch.ones(4),
                    swappable=True,
                    storage_nbytes=1024,
                )],
            )
            for _ in range(4)
        ]

        batches = runtime.partition(units)

        self.assertEqual(sum(len(batch) for batch in batches), 4)
        self.assertEqual([len(batch) for batch in batches], [2, 2])

    def test_generic_update_unit_is_still_constructible(self):
        """The legacy generic unit keeps its shape for optimizers that need it."""
        slot = SwapSlot(name="momentum_buffer", tensor=torch.ones(4))
        unit = UpdateUnit(adapter_index=0, param=object(), grad=object(), slots=[slot])

        self.assertEqual(unit.adapter_index, 0)
        self.assertEqual(unit.slots, [slot])


if __name__ == "__main__":
    unittest.main()
