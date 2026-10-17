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
from types import SimpleNamespace
from unittest import mock

import torch

from hyper_parallel import DTensor, DeviceMesh, Replicate, SkipDTensorDispatch
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer import Muon, SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.swap_adam import AdamSwapAdapter
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
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.optimizer.sharding_category import (
    HSDPGroupAssignment,
    ParamRecord,
    build_owner_by_size,
    select_owned_records,
)


def _muon(params, **kwargs):
    """Build a bare CPU Muon optimizer over ``params``."""
    return Muon(list(params), lr=0.01, **kwargs)


def _swap_config(**overrides):
    """Build a Muon swap config with per-tensor defaults."""
    values = {"swap_times": 2, "min_numel": 1, "packed_swap": False}
    values.update(overrides)
    return SwapOptimizerConfig(**values)


def _make_assignment(params, is_shard, owners=None, rank=0):
    """Build real assignment metadata while leaving process groups uninitialized."""
    records = [ParamRecord(index, param) for index, param in enumerate(params)]
    owner_by_index = dict(enumerate(owners if owners is not None else [(0,)] * len(records)))
    return HSDPGroupAssignment(
        owned_records=[record for record in records if owner_by_index[record.index] == (rank,)],
        all_records=records,
        owner_by_index=owner_by_index,
        replicate_group_ranks=(rank,),
        replicate_sizes=(3,),
        is_shard=is_shard,
    )


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
    """Build a Muon adapter with a configurable swap runtime."""
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

    def test_packed_swap_is_dispatched_for_muon(self):
        """Muon accepts packed momentum staging through the shared runtime."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        wrapped = swap_optimizer(_muon([param]), SwapOptimizerConfig(packed_swap=True))

        self.assertIsInstance(wrapped.adapter, MuonSwapAdapter)
        self.assertTrue(wrapped.runtime.packed_enabled)

    def test_muon_without_explicit_config_uses_packed_default(self):
        """Muon shares the packed default used by every swap optimizer family."""
        param = torch.nn.Parameter(torch.ones(8, 8))

        wrapped = swap_optimizer(_muon([param]))

        self.assertTrue(wrapped.config.packed_swap)

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

    def test_validate_rejects_custom_momentum_update_fn(self):
        """A custom momentum hook cannot be silently split across swap units."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon(
            [param],
            momentum_update_fn=lambda grads, *_: grads,
        )

        with self.assertRaisesRegex(ValueError, "does not support momentum_update_fn"):
            swap_muon(optimizer, _swap_config())

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

    def test_no_comm_post_update_keeps_original_order_and_boundary(self):
        """Callbacks run after all shape units and retain parameter-group order."""
        params = [
            torch.nn.Parameter(torch.ones(2, 2)),
            torch.nn.Parameter(torch.ones(3, 2)),
            torch.nn.Parameter(torch.ones(2, 2)),
        ]
        updated = set()
        observed = []
        for index, param in enumerate(params):
            param.model_name = f"p{index}"
            param.grad = torch.full_like(param, index + 1)

        def _post_update(param, tensor, context):
            del tensor, context
            observed.append((param.model_name, tuple(sorted(updated))))

        wrapped = swap_muon(_muon(params, post_update_fn=_post_update), _swap_config(swap_times=2))

        def _record_update(group, ns_inputs):
            del group
            updated.update(param.model_name for param in ns_inputs)

        with mock.patch.object(wrapped.adapter, "_apply_unshard_update", side_effect=_record_update):
            wrapped.step()

        expected_updated = ("p0", "p1", "p2")
        self.assertEqual(observed, [
            ("p0", expected_updated),
            ("p1", expected_updated),
            ("p2", expected_updated),
        ])

    def test_no_comm_without_gradients_retains_the_logical_unit(self):
        """Local gradient absence does not delete a unit or create momentum."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        adapter, _ = _adapter(_muon([param]))

        context = adapter.prepare_step()

        self.assertEqual(len(context["units"]), 1)
        self.assertEqual(context["units"][0].params, [])
        self.assertEqual(context["units"][0].slots, [])
        self.assertEqual(context["units"][0].logical_nbytes, 256)
        self.assertEqual(dict(adapter.optimizer.state), {})

    @mock.patch.object(DeviceMesh, "get_group", return_value=None)
    @mock.patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
    def test_dtensor_logical_bytes_use_local_tensor_metadata(self, get_rank, get_group):
        """DTensor byte sizing must not dispatch the unsupported element_size op."""
        del get_rank, get_group
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp",), _init_backend=False)
        param = torch.nn.Parameter(DTensor.from_local(
            torch.ones(4, 4, dtype=torch.bfloat16), mesh, (Replicate(),),
        ))
        adapter, _ = _adapter(_muon([param]))

        context = adapter.prepare_step()

        self.assertEqual(context["units"][0].logical_nbytes, 32)

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
        optimizer._split_into_memory_safe_batches = mock.Mock(
            return_value=[[params[0]], [params[1], params[2]]]
        )

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual([[p.shape for p in unit.params] for unit in units], [[(8, 8)], [(8, 8), (8, 8)]])
        self.assertEqual([unit.post_update_params for unit in units], [(), tuple(params)])
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


class TestMuonPackedSwap(unittest.TestCase):
    """Packed host layout and live momentum-state rebinding."""

    def setUp(self) -> None:
        """Replace pinned allocation and accelerator events with CPU equivalents."""
        allocate = torch.empty

        def _cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return allocate(*args, **kwargs)

        patcher = mock.patch.object(torch, "empty", side_effect=_cpu_empty)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _prepare(params, gradients=None, swap_times=2):
        """Build logical batches while forcing CPU tensors through packed bookkeeping."""
        optimizer = _muon(params)
        adapter, runtime = _adapter(optimizer, packed_swap=True, swap_times=swap_times)
        runtime.is_swappable_tensor = mock.Mock(return_value=True)
        runtime.release_device_storage = mock.Mock()
        runtime.current_stream = mock.Mock(return_value=None)
        runtime.new_stream = mock.Mock(return_value=None)
        runtime.record_event = mock.Mock(side_effect=lambda *args: mock.Mock())
        gradients = params if gradients is None else gradients
        for param in gradients:
            param.grad = torch.ones_like(param)
        context = adapter.prepare_step()
        units = adapter.iter_update_units(context)
        batches = runtime.partition(units, adapter.partition_costs(units, context))
        return optimizer, adapter, runtime, context, batches

    def test_pipeline_numerics_and_checkpoint_survive_arena_reuse(self):
        """Real CPU copies exercise >2 batches, lazy state and changing active sets."""
        shapes = ((3, 4), (5, 4), (3, 4), (6, 4), (7, 4))
        for swap_times in (1, 2, 16):
            with self.subTest(swap_times=swap_times):
                params = [torch.nn.Parameter(torch.ones(shape)) for shape in shapes]
                optimizer, adapter, runtime, _, _ = self._prepare(params, [], swap_times)
                bare_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
                bare = _muon(bare_params)
                for step in range(4):
                    for index, (param, reference) in enumerate(zip(params, bare_params)):
                        grad = torch.full_like(param, step + index + 1)
                        param.grad = None if index == step else grad
                        reference.grad = None if param.grad is None else grad.clone()
                    with torch.no_grad():
                        context = adapter.prepare_step()
                        units = context["units"]
                        batches = runtime.partition(units, adapter.partition_costs(units, context))
                        adapter.prepare_pipeline(batches, context)
                        runtime.run_pipeline(batches, context, adapter.step_batch)
                        adapter.finish_step(context)
                        bare.step()
                    for param, reference in zip(params, bare_params):
                        torch.testing.assert_close(param, reference, rtol=0, atol=0)
                        if reference in bare.state:
                            slot = adapter._slots[(id(param), "momentum_buffer")]
                            self.assertIs(optimizer.state[param]["momentum_buffer"], slot.cpu_tensor)
                            torch.testing.assert_close(
                                slot.cpu_tensor, bare.state[reference]["momentum_buffer"], rtol=0, atol=0,
                            )
                    self.assertTrue(all(arena.raw_buffer.untyped_storage().size() == 0
                                        for arena in runtime._staging_arenas if arena is not None))
                    if step == 1:
                        adapter.load_checkpoint_state_dict(adapter.checkpoint_state_dict())

    def test_exception_does_not_copy_back_an_already_reused_arena(self):
        """A failing second update preserves batch 0 after arena 0 loads batch 2."""
        params = [torch.nn.Parameter(torch.ones(rows, 4)) for rows in (3, 4, 5, 6)]
        optimizer, adapter, runtime, context, batches = self._prepare(params, swap_times=16)
        adapter.prepare_pipeline(batches, context)
        called = []

        def _fail_second_batch(batch, context):
            del context
            called.append(batch)
            for unit in batch:
                for slot in unit.slots:
                    slot.tensor.fill_(len(called))
            if len(called) == 2:
                raise RuntimeError("injected update failure")

        with self.assertRaisesRegex(RuntimeError, "injected update failure"):
            runtime.run_pipeline(batches, context, _fail_second_batch)

        for index, param in enumerate(params):
            slot = adapter._slots[(id(param), "momentum_buffer")]
            self.assertEqual(slot.state, "host")
            self.assertIs(optimizer.state[param]["momentum_buffer"], slot.cpu_tensor)
            torch.testing.assert_close(slot.cpu_tensor, torch.full_like(param, index + 1 if index < 2 else 0))
        self.assertEqual(runtime._packed_batch_plans, [])

    def test_empty_batches_keep_arena_parity_and_inactive_state(self):
        """Empty local batches retain their logical positions and never lose state."""
        params = [torch.nn.Parameter(torch.ones(rows, 4)) for rows in (3, 4, 5, 6)]
        optimizer, adapter, runtime, context, batches = self._prepare(params, swap_times=16)
        for index in (0, 2):
            batches[index][0].params = []
            batches[index][0].slots = []
        adapter.prepare_pipeline(batches, context)
        with mock.patch.object(runtime, "activate_packed_batch", wraps=runtime.activate_packed_batch) as activate:
            with torch.no_grad():
                runtime.run_pipeline(batches, context, adapter.step_batch)
        self.assertEqual([call.args for call in activate.call_args_list], [(0, 0), (1, 1), (2, 0), (3, 1)])
        for index, param in enumerate(params):
            slot = adapter._slots[(id(param), "momentum_buffer")]
            self.assertIs(optimizer.state[param]["momentum_buffer"], slot.cpu_tensor)
            torch.testing.assert_close(slot.cpu_tensor, torch.full_like(param, 0 if index in (0, 2) else 1))

    def test_no_local_state_executes_units_without_allocating_arenas(self):
        """A zero-byte rank still reaches every callback and logical batch."""
        params = [torch.nn.Parameter(torch.ones(rows, 4)) for rows in (3, 4, 5)]
        _, adapter, runtime, context, batches = self._prepare(params, [], swap_times=16)
        adapter.prepare_pipeline(batches, context)
        execute = mock.Mock()

        runtime.run_pipeline(batches, context, execute)

        self.assertEqual([call.args[0] for call in execute.call_args_list], batches)
        self.assertEqual(runtime._staging_arenas, [None, None])
        self.assertEqual(runtime._host_buffers, {})

    def test_rank_local_layouts_preserve_global_batch_signatures(self):
        """Uneven and absent ownership yields different local layouts, not partitions."""
        signatures = []
        slot_counts = []
        for rank in range(3):
            wrapped, _, _ = TestMuonLogicalSchedule._rank_view(rank, packed_swap=True)
            runtime = wrapped.runtime
            runtime.is_swappable_tensor = mock.Mock(return_value=True)
            runtime.release_device_storage = mock.Mock()
            context = wrapped.adapter.prepare_step()
            units = context["units"]
            batches = runtime.partition(units, wrapped.adapter.partition_costs(units, context))
            signatures.append([[(unit.schedule_key, unit.logical_nbytes) for unit in batch] for batch in batches])
            wrapped.adapter.prepare_pipeline(batches, context)
            plans = [runtime._build_packed_batch_plan(batch) for batch in batches]
            slot_counts.append(sum(len(region.slots) for plan in plans for region in plan.regions.values()))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])
        self.assertEqual(slot_counts, [4, 1, 0])

    def test_layout_follows_logical_batches_and_separates_dtypes(self):
        """Local offsets follow batch/unit order independently for each dtype."""
        fp32 = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32))
        fp64 = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.float64))
        optimizer, adapter, runtime, context, batches = self._prepare([fp32, fp64])
        del optimizer
        reversed_batches = list(reversed(batches))

        adapter.prepare_pipeline(reversed_batches, context)

        fp32_slot = adapter._slots[(id(fp32), "momentum_buffer")]
        fp64_slot = adapter._slots[(id(fp64), "momentum_buffer")]
        self.assertEqual(fp32_slot.host_offset, 0)
        self.assertEqual(fp64_slot.host_offset, 0)
        self.assertEqual(set(runtime._host_buffers), {torch.float32, torch.float64})
        self.assertEqual(
            [entry[2] for entry in runtime._host_layout_signature],
            [tuple(reversed_batches[0][0].slots[0].shape), tuple(reversed_batches[1][0].slots[0].shape)],
        )

    def test_packed_binding_publishes_staging_and_persistent_host_views(self):
        """Muon state follows every packed slot binding used by the runtime."""
        param = torch.nn.Parameter(torch.ones(4, 4))
        optimizer, adapter, runtime, context, batches = self._prepare([param])
        adapter.prepare_pipeline(batches, context)
        slot = adapter._slots[(id(param), "momentum_buffer")]
        self.assertIs(optimizer.state[param]["momentum_buffer"], slot.cpu_tensor)

        staging = torch.full_like(slot.cpu_tensor, 3)
        runtime._bind_packed_slot(slot, staging)
        self.assertIs(optimizer.state[param]["momentum_buffer"], staging)

        runtime._bind_packed_slot(slot, slot.cpu_tensor)
        self.assertIs(optimizer.state[param]["momentum_buffer"], slot.cpu_tensor)

    def test_lazy_momentum_refreshes_the_layout_before_the_same_step(self):
        """A newly gradient-bearing parameter enters the packed layout immediately."""
        first = torch.nn.Parameter(torch.ones(2, 3))
        second = torch.nn.Parameter(torch.ones(3, 4))
        optimizer, adapter, runtime, context, batches = self._prepare([first, second], [first])
        adapter.prepare_pipeline(batches, context)
        first_signature = runtime._host_layout_signature

        second.grad = torch.ones_like(second)
        context = adapter.prepare_step()
        units = adapter.iter_update_units(context)
        batches = runtime.partition(units, adapter.partition_costs(units, context))
        adapter.prepare_pipeline(batches, context)

        self.assertEqual(len(first_signature), 1)
        self.assertEqual(len(runtime._host_layout_signature), 2)
        second_slot = adapter._slots[(id(second), "momentum_buffer")]
        self.assertIs(optimizer.state[second]["momentum_buffer"], second_slot.cpu_tensor)

    def test_checkpoint_materialization_restores_the_packed_host_view(self):
        """A checkpoint-only live tensor never escapes the broadcast window."""
        param = torch.nn.Parameter(torch.ones(4, 4))
        optimizer, adapter, runtime, context, batches = self._prepare([param])
        adapter.prepare_pipeline(batches, context)
        slot = adapter._slots[(id(param), "momentum_buffer")]
        optimizer._param_to_broadcast_info = {param: ((), ())}

        with mock.patch.object(runtime, "current_stream", return_value=None), \
                mock.patch.object(runtime, "new_stream", return_value=None), \
                mock.patch.object(runtime, "record_event", return_value=None):
            adapter.prepare_checkpoint()
            self.assertEqual(slot.state, "device")
            self.assertIs(optimizer.state[param]["momentum_buffer"], slot.tensor)

            adapter.finish_checkpoint()

        self.assertEqual(slot.state, "host")
        self.assertIs(slot.tensor, slot.cpu_tensor)
        self.assertIs(optimizer.state[param]["momentum_buffer"], slot.cpu_tensor)


class TestMuonSwapUnitContract(unittest.TestCase):
    """Muon units satisfy the generic runtime contract without Adam fields."""

    def test_muon_unit_exposes_slots_without_param_or_grad(self):
        """The runtime only needs ``slots``; a Muon unit carries no Adam fields."""
        slot = SwapSlot(name="momentum_buffer", tensor=torch.ones(4))
        unit = MuonSwapUnit(
            params=[], slots=[slot], schedule_key=(0, "no_comm", -1, 0), logical_nbytes=16,
        )

        self.assertEqual(unit.slots, [slot])
        self.assertEqual(unit.group_index, unit.schedule_key[0])
        self.assertEqual(unit.kind, unit.schedule_key[1])
        self.assertFalse(hasattr(unit, "param"))
        self.assertFalse(hasattr(unit, "grad"))

    def test_runtime_partitions_muon_units(self):
        """``PipelineSwapRuntime.partition`` accepts Muon units unchanged."""
        runtime = PipelineSwapRuntime(_swap_config(swap_times=2, min_numel=0))
        units = [
            MuonSwapUnit(
                params=[],
                schedule_key=(0, "no_comm", -1, index),
                logical_nbytes=1024,
                slots=[SwapSlot(
                    name="momentum_buffer",
                    tensor=torch.ones(4),
                    swappable=True,
                    storage_nbytes=1024,
                )],
            )
            for index in range(4)
        ]

        batches = runtime.partition(units)

        self.assertEqual(sum(len(batch) for batch in batches), 4)
        self.assertEqual([len(batch) for batch in batches], [2, 2])

    def test_unknown_schedule_kind_is_rejected_before_execution(self):
        """Only execution kinds represented by the schedule contract are accepted."""
        param = torch.nn.Parameter(torch.ones(4, 4))
        adapter, _ = _adapter(_muon([param]))
        unit = MuonSwapUnit(
            params=[], slots=[], schedule_key=(0, "unknown", -1, 0), logical_nbytes=0,
        )

        with self.assertRaisesRegex(ValueError, "Unknown Muon swap unit kind"):
            adapter._run_unit(unit, mock.Mock())

    def test_generic_update_unit_is_still_constructible(self):
        """The legacy generic unit keeps its shape for optimizers that need it."""
        slot = SwapSlot(name="momentum_buffer", tensor=torch.ones(4))
        unit = UpdateUnit(adapter_index=0, param=object(), grad=object(), slots=[slot])

        self.assertEqual(unit.adapter_index, 0)
        self.assertEqual(unit.slots, [slot])


class TestMuonLogicalSchedule(unittest.TestCase):
    """Rank-independent planning over real HSDP ownership metadata."""

    @staticmethod
    def _rank_view(rank, is_shard=False, swap_times=2, packed_swap=False):
        """Use unequal owner workloads plus a gradient-dependent no-comm prefix."""
        params = [
            torch.nn.Parameter(torch.ones(8, 8)),
            torch.nn.Parameter(torch.ones(8, 8)),
            torch.nn.Parameter(torch.ones(4, 8, dtype=torch.float64)),
            torch.nn.Parameter(torch.ones(4, 12)),
        ]
        no_comm = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = _muon([no_comm] + params)
        for param in params:
            param.grad = torch.ones_like(param)
        if rank == 0:
            no_comm.grad = torch.ones_like(no_comm)
        assignment = _make_assignment(params, is_shard, owners=[(0,), (1,), (0,), (0,)], rank=rank)
        _install_schedule(optimizer, assignments=[assignment], no_comm=[no_comm])
        wrapped = swap_muon(optimizer, _swap_config(
            swap_times=swap_times, min_numel=rank + 1, packed_swap=packed_swap,
        ))
        return wrapped, assignment, no_comm

    @staticmethod
    def _snapshot(wrapped):
        """Observe the complete adapter-to-runtime partition contract."""
        context = wrapped.adapter.prepare_step()
        units = context["units"]
        costs = wrapped.adapter.partition_costs(units, context)
        batches = wrapped.runtime.partition(units, unit_costs=costs)
        signature = (
            [(unit.schedule_key, unit.logical_nbytes, unit.flushes_broadcast) for unit in units],
            [[unit.schedule_key for unit in batch] for batch in batches],
        )
        return signature, units

    def test_rank_views_keep_identical_units_costs_and_partitions(self):
        """Owner imbalance, mixed dtype and empty owners never alter the plan."""
        for is_shard in (False, True):
            for swap_times in (1, 2, 16):
                with self.subTest(is_shard=is_shard, swap_times=swap_times):
                    snapshots = []
                    local_counts = []
                    for rank in range(3):
                        wrapped, assignment, _ = self._rank_view(rank, is_shard, swap_times)
                        signature, units = self._snapshot(wrapped)
                        snapshots.append(signature)
                        local_counts.append(sum(len(unit.slots) for unit in units))
                        self.assertEqual(sum(unit.logical_nbytes for unit in units), 984)
                        self.assertEqual(len(units), 2 if is_shard else 4)
                        self.assertEqual(sum(unit.flushes_broadcast for unit in units), 1)
                        self.assertTrue(units[-1].flushes_broadcast)
                        self.assertEqual(
                            {id(param) for unit in units[1:] for param in unit.params},
                            {id(param) for param in assignment.owned_params},
                        )
                    self.assertEqual(snapshots[0], snapshots[1])
                    self.assertEqual(snapshots[0], snapshots[2])
                    self.assertEqual(local_counts, [4, 1, 0])

    def test_gradient_changes_and_lazy_state_keep_plan_stable(self):
        """Slot discovery and missing gradients only affect the local projection."""
        wrapped, assignment, no_comm = self._rank_view(0)
        before, _ = self._snapshot(wrapped)
        assignment.owned_params[0].grad = None
        no_comm.grad = None
        after, units = self._snapshot(wrapped)
        self.assertEqual(before, after)
        self.assertEqual(len(units[0].slots), 0)
        self.assertEqual(len(units[1].slots), 0)
        self.assertEqual(sum(len(unit.params) for unit in units), 2)

    def test_unshard_memory_batches_use_full_metadata_without_allocating_state(self):
        """Large logical matrices split without allocating their momentum buffers."""
        params = [torch.nn.Parameter(torch.empty(20000, 20000, device="meta")) for _ in range(4)]
        signatures = []
        for rank in range(3):
            optimizer = _muon(params)
            assignment = _make_assignment(params, False, owners=[(0,), (1,), (0,), (0,)], rank=rank)
            _install_schedule(optimizer, assignments=[assignment])
            wrapped = swap_muon(optimizer, _swap_config())
            signature, units = self._snapshot(wrapped)
            signatures.append(signature)
            self.assertEqual(len(units), 3)
            self.assertEqual([unit.logical_nbytes for unit in units], [3200000000, 1600000000, 1600000000])
            self.assertEqual(dict(optimizer.state), {})
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])

    def test_multiple_groups_keep_original_assignment_interleave(self):
        """Stable keys preserve no-comm prefixes and batch-index/group-index order."""
        params = [torch.nn.Parameter(torch.ones(2, 2)) for _ in range(5)]
        optimizer = _muon([{"params": params[:3]}, {"params": params[3:]}])
        optimizer._hsdp_assignment_batches = {
            0: {"no_comm": params[:1], "batch_groups": [{"sub_batches": [
                _make_assignment(params[1:2], True), _make_assignment(params[2:3], True)
            ]}]},
            1: {"no_comm": params[3:4], "batch_groups": [{"sub_batches": [
                _make_assignment(params[4:], True)
            ]}]},
        }
        wrapped = swap_muon(optimizer, _swap_config())
        _, units = self._snapshot(wrapped)
        self.assertEqual([unit.schedule_key for unit in units], [
            (0, "no_comm", -1, 0), (1, "no_comm", -1, 0),
            (0, "hsdp", 0, 0), (1, "hsdp", 0, 0), (0, "hsdp", 1, 0),
        ])

    def test_empty_owner_executes_final_flush_after_other_ranks_local_batches(self):
        """A rank with fewer NS batches still flushes at the global final unit."""
        for rank in range(3):
            with self.subTest(rank=rank):
                wrapped, assignment, _ = self._rank_view(rank)
                context = wrapped.adapter.prepare_step()
                context["broadcaster"] = mock.Mock()
                position = []
                context["broadcaster"].flush_group.side_effect = (
                    lambda _, position=position: position.append(current_key)
                )
                units = context["units"]
                with torch.no_grad():
                    for unit in units:
                        current_key = unit.schedule_key
                        wrapped.adapter.step_batch([unit], context)
                wrapped.adapter.finish_step(context)
                self.assertEqual(position, [(0, "hsdp", 0, 2)])
                context["broadcaster"].flush_group.assert_called_once_with(assignment)
                context["broadcaster"].wait_all.assert_called_once_with()

    def test_sharded_executor_keeps_the_original_assignment_even_with_no_local_state(self):
        """The shard entry is called once and owns its collective membership."""
        for rank in range(3):
            with self.subTest(rank=rank):
                wrapped, assignment, _ = self._rank_view(rank, is_shard=True)
                with mock.patch.object(wrapped.optimizer, "_process_shard_params") as execute:
                    wrapped.step()
                execute.assert_called_once()
                self.assertIs(execute.call_args.args[2][0], assignment)

    def test_wrapper_uses_logical_costs_for_actual_pipeline_batches(self):
        """CPU slots cost zero locally, but the wrapper still partitions by logical bytes."""
        params = [torch.nn.Parameter(torch.ones(*shape)) for shape in ((2, 100), (2, 2), (3, 2), (4, 2))]
        wrapped = swap_muon(_muon(params), _swap_config())
        for param in params:
            param.grad = torch.ones_like(param)
        with mock.patch.object(wrapped.runtime, "run_pipeline", wraps=wrapped.runtime.run_pipeline) as pipeline:
            wrapped.step()
        self.assertEqual([len(batch) for batch in pipeline.call_args.args[0]], [1, 3])

    def test_hsdp_callback_changes_are_included_in_broadcast(self):
        """Flush sees the owner callback result, including the last logical unit."""
        param = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = _muon([param], post_update_fn=lambda _, tensor, __: tensor.fill_(7))
        assignment = _make_assignment([param], False)
        _install_schedule(optimizer, assignments=[assignment])
        wrapped = swap_muon(optimizer, _swap_config())
        param.grad = torch.ones_like(param)
        context = wrapped.adapter.prepare_step()
        context["broadcaster"] = mock.Mock()
        seen = []
        context["broadcaster"].flush_group.side_effect = lambda _: seen.append(torch.clone(param))
        with torch.no_grad():
            wrapped.adapter.step_batch(context["units"], context)
        torch.testing.assert_close(seen[0], torch.full_like(param, 7))

    def test_unshard_steps_and_checkpoint_resume_match_bare_muon(self):
        """Planning changes retain numerical behavior across multiple steps and reload."""
        for swap_times in (1, 2, 16):
            with self.subTest(swap_times=swap_times):
                bare_params = [torch.nn.Parameter(torch.ones(*shape)) for shape in ((4, 4), (2, 4), (3, 2))]
                swap_params = [torch.nn.Parameter(param.detach().clone()) for param in bare_params]
                bare = _muon(bare_params, momentum=(0.8, 0.9))
                wrapped = swap_muon(_muon(swap_params, momentum=(0.8, 0.9)), _swap_config(swap_times=swap_times))
                _install_schedule(bare, assignments=[_make_assignment(bare_params, False)])
                _install_schedule(wrapped.optimizer, assignments=[_make_assignment(swap_params, False)])
                for step in range(4):
                    for index, (param, swapped) in enumerate(zip(bare_params, swap_params)):
                        grad = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param) * 0.01 + step + 1
                        param.grad = None if step == index else grad
                        swapped.grad = None if param.grad is None else grad.clone()
                    bare.step()
                    wrapped.step()
                    for param, swapped in zip(bare_params, swap_params):
                        torch.testing.assert_close(param, swapped, rtol=0, atol=0)
                        if param in bare.state:
                            torch.testing.assert_close(
                                bare.state[param]["momentum_buffer"], wrapped.state[swapped]["momentum_buffer"],
                                rtol=0, atol=0,
                            )
                    if step == 1:
                        wrapped.load_state_dict(wrapped.state_dict())


class TestMuonFrozenUnshardSchedule(unittest.TestCase):
    """Frozen records retain assignment positions without requiring an owner."""

    @staticmethod
    def _assignment(params, rank_coord=(0,), replicate_sizes=(3,)):
        """Use production ownership rules, including frozen-record filtering."""
        records = [ParamRecord(index, param) for index, param in enumerate(params)]
        owners = build_owner_by_size(records, replicate_sizes)
        return HSDPGroupAssignment(
            owned_records=select_owned_records(records, owners, rank_coord),
            all_records=records, owner_by_index=owners,
            replicate_group_ranks=rank_coord, replicate_sizes=replicate_sizes,
            is_shard=False,
        )

    def test_mixed_frozen_records_match_bare_muon_and_keep_global_plan(self):
        """All rank views plan identically and only owned trainable params update."""
        for swap_times in (1, 2, 16):
            signatures = []
            for rank in range(3):
                with self.subTest(swap_times=swap_times, rank=rank):
                    bare_params = [
                        torch.nn.Parameter(torch.ones(4, 4)),
                        torch.nn.Parameter(torch.ones(3, 3), requires_grad=False),
                        torch.nn.Parameter(torch.ones(2, 4)),
                    ]
                    swap_params = [
                        torch.nn.Parameter(torch.detach(param).clone(), requires_grad=param.requires_grad)
                        for param in bare_params
                    ]
                    bare, swapped = _muon(bare_params), _muon(swap_params)
                    bare_assignment = self._assignment(bare_params, (rank,))
                    assignment = self._assignment(swap_params, (rank,))
                    self.assertNotIn(1, assignment.owner_by_index)
                    _install_schedule(bare, assignments=[bare_assignment])
                    _install_schedule(swapped, assignments=[assignment])
                    wrapped = swap_muon(swapped, _swap_config(swap_times=swap_times))
                    for step in range(3):
                        for param in bare_params + swap_params:
                            if param.requires_grad:
                                param.grad = torch.full_like(param, step + 1)
                        bare.step()
                        with mock.patch.object(wrapped.runtime, "run_pipeline",
                                               wraps=wrapped.runtime.run_pipeline) as pipeline:
                            wrapped.step()
                        batches = pipeline.call_args.args[0]
                        signatures.append([[(unit.schedule_key, unit.logical_nbytes) for unit in batch]
                                           for batch in batches])
                        self.assertEqual(sum(unit.logical_nbytes for batch in batches for unit in batch), 96)
                        for expected, actual in zip(bare_params, swap_params):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        for expected, actual in zip(bare_assignment.owned_params, assignment.owned_params):
                            torch.testing.assert_close(
                                wrapped.state[actual]["momentum_buffer"],
                                bare.state[expected]["momentum_buffer"], rtol=0, atol=0,
                            )
                    torch.testing.assert_close(swap_params[1], torch.ones(3, 3), rtol=0, atol=0)
                    self.assertEqual(set(wrapped.state), set(assignment.owned_params))
                    self.assertEqual({param_id for param_id, _ in wrapped.adapter._slots},
                                     {id(param) for param in assignment.owned_params})
            self.assertTrue(all(signature == signatures[0] for signature in signatures))

    def test_all_frozen_assignment_keeps_empty_unit_and_single_flush(self):
        """A zero-byte assignment still reaches its logical execution and flush."""
        for rank in range(3):
            with self.subTest(rank=rank):
                param = torch.nn.Parameter(torch.ones(4, 4), requires_grad=False)
                optimizer = _muon([param])
                assignment = self._assignment([param], (rank,))
                _install_schedule(optimizer, assignments=[assignment])
                wrapped = swap_muon(optimizer, _swap_config())
                with mock.patch("hyper_parallel.core.optimizer.swap_muon.AsyncReplicateBroadcaster") as broadcaster, \
                        mock.patch.object(wrapped.adapter, "step_batch", wraps=wrapped.adapter.step_batch) as execute:
                    wrapped.step()
                execute.assert_called_once()
                units = execute.call_args.args[0]
                self.assertEqual(len(units), 1)
                self.assertEqual(units[0].schedule_key, (0, "hsdp", 0, 0))
                self.assertEqual(units[0].logical_nbytes, 0)
                self.assertEqual(units[0].params, [])
                self.assertEqual(units[0].slots, [])
                self.assertTrue(units[0].flushes_broadcast)
                broadcaster.return_value.flush_group.assert_called_once_with(assignment)
                broadcaster.return_value.wait_all.assert_called_once_with()
                self.assertFalse(wrapped.state)
                self.assertEqual(tuple(wrapped.adapter.all_slots()), ())

    def test_empty_owner_coordinate_is_valid(self):
        """No replicate dimensions use () as an owner, which must not be skipped."""
        param = torch.nn.Parameter(torch.ones(4, 4))
        optimizer = _muon([param])
        assignment = self._assignment([param], rank_coord=(), replicate_sizes=())
        self.assertEqual(assignment.owner_by_index, {0: ()})
        _install_schedule(optimizer, assignments=[assignment])
        wrapped = swap_muon(optimizer, _swap_config())
        param.grad = torch.ones_like(param)

        wrapped.step()

        self.assertFalse(torch.equal(param, torch.ones(4, 4)))
        torch.testing.assert_close(wrapped.state[param]["momentum_buffer"], torch.ones(4, 4))
        self.assertEqual(set(wrapped.adapter._slots), {(id(param), "momentum_buffer")})


class TestMuonSwapCheckpointHooks(unittest.TestCase):
    """Checkpoint materialize/offload lifecycle shared with the HSDP broadcast."""

    @staticmethod
    def _host_resident_adapter(params, broadcast_params):
        """Build an adapter whose momentum slots are swappable and host-resident."""
        optimizer = _muon(params)
        adapter, runtime = _adapter(optimizer)
        for param in params:
            param.grad = torch.ones_like(param)
        runtime.is_swappable_tensor = mock.Mock(return_value=True)
        runtime.release_device_storage = mock.Mock()
        runtime.current_stream = mock.Mock(return_value=None)
        runtime.new_stream = mock.Mock(return_value=None)
        adapter.prepare_step()
        for param in params:
            slot = adapter._slots[(id(param), "momentum_buffer")]
            slot.cpu_tensor = torch.full_like(slot.cpu_tensor, 0.5)
            slot.state = "host"
        optimizer._param_to_broadcast_info = {param: ((), ()) for param in broadcast_params}
        return optimizer, adapter

    def test_prepare_checkpoint_materializes_only_replicated_slots(self):
        """Only the state the HSDP broadcast reads returns to device storage."""
        replicated = torch.nn.Parameter(torch.ones(8, 8))
        local_only = torch.nn.Parameter(torch.ones(8, 8))
        _, adapter = self._host_resident_adapter([replicated, local_only], [replicated])
        adapter.runtime.materialize_slots = mock.Mock(wraps=adapter.runtime.materialize_slots)

        adapter.prepare_checkpoint()

        replicated_slot = adapter._slots[(id(replicated), "momentum_buffer")]
        local_slot = adapter._slots[(id(local_only), "momentum_buffer")]
        self.assertEqual(adapter._checkpoint_slots(), (replicated_slot,))
        self.assertEqual(
            adapter.runtime.materialize_slots.call_args.args[0], (replicated_slot,)
        )
        self.assertEqual(replicated_slot.state, "device")
        self.assertEqual(local_slot.state, "host")
        self.assertTrue(adapter._checkpoint_active)

    def test_finish_checkpoint_returns_broadcast_state_to_host(self):
        """The broadcast result is copied back and the device storage released."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer, adapter = self._host_resident_adapter([param], [param])
        adapter.prepare_checkpoint()
        slot = adapter._slots[(id(param), "momentum_buffer")]
        with torch.no_grad():
            slot.tensor.fill_(0.25)

        adapter.finish_checkpoint()

        self.assertFalse(adapter._checkpoint_active)
        self.assertEqual(slot.state, "host")
        self.assertTrue(torch.allclose(slot.cpu_tensor, torch.full_like(slot.cpu_tensor, 0.25)))
        self.assertIs(optimizer.state[param]["momentum_buffer"], slot.tensor)

    def test_checkpoint_hooks_are_idempotent(self):
        """Nested prepare/finish pairs keep exactly one active lifecycle."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        _, adapter = self._host_resident_adapter([param], [param])
        adapter.runtime.synchronize_cpu_mirrors = mock.Mock()
        adapter.runtime.materialize_slots = mock.Mock()

        adapter.prepare_checkpoint()
        adapter.prepare_checkpoint()
        self.assertEqual(adapter.runtime.synchronize_cpu_mirrors.call_count, 1)
        self.assertEqual(adapter.runtime.materialize_slots.call_count, 1)

        adapter.finish_checkpoint()
        adapter.finish_checkpoint()
        self.assertEqual(adapter.runtime.synchronize_cpu_mirrors.call_count, 2)
        self.assertEqual(adapter.runtime.materialize_slots.call_count, 1)
        self.assertFalse(adapter._checkpoint_active)

    def test_prepare_checkpoint_without_broadcast_info_materializes_nothing(self):
        """A non-HSDP optimizer keeps every slot host-resident."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        _, adapter = self._host_resident_adapter([param], [])

        adapter.prepare_checkpoint()

        self.assertEqual(adapter._checkpoint_slots(), ())
        self.assertEqual(adapter._slots[(id(param), "momentum_buffer")].state, "host")

    def test_prepare_checkpoint_failure_restores_host_state(self):
        """A failed H2D wait closes the checkpoint window and offloads its slot."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        _, adapter = self._host_resident_adapter([param], [param])
        adapter.runtime.wait_prefetch = mock.Mock(side_effect=RuntimeError("h2d failed"))

        with self.assertRaisesRegex(RuntimeError, "h2d failed"):
            adapter.prepare_checkpoint()

        slot = adapter._slots[(id(param), "momentum_buffer")]
        self.assertEqual(slot.state, "host")
        self.assertFalse(adapter._checkpoint_active)
        self.assertFalse(adapter._checkpoint_broadcasted)


class TestMuonSwapHSDPIntegration(unittest.TestCase):
    """HSDP ownership, collective order and checkpoint wiring for swap Muon."""

    def test_non_owner_state_is_dropped_when_the_wrapper_is_created(self):
        """Pre-materialized non-owner state never becomes a swap slot."""
        owner = torch.nn.Parameter(torch.ones(8, 8))
        non_owner = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([owner, non_owner])
        optimizer.state[owner]["momentum_buffer"] = torch.ones_like(owner)
        optimizer.state[non_owner]["momentum_buffer"] = torch.ones_like(non_owner)
        _install_schedule(
            optimizer,
            assignments=[_make_assignment([owner], is_shard=True)],
            no_comm=[],
        )

        wrapped = swap_muon(optimizer, _swap_config(min_numel=1))

        self.assertIn(owner, wrapped.state)
        self.assertNotIn(non_owner, wrapped.state)
        self.assertEqual({key[0] for key in wrapped.adapter._slots}, {id(owner)})

    def test_empty_schedule_keeps_preexisting_state_for_plain_muon(self):
        """An empty non-HSDP schedule means every parameter is locally owned."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([param])
        optimizer.state[param]["momentum_buffer"] = torch.ones_like(param)
        optimizer._hsdp_assignment_batches = {}

        wrapped = swap_muon(optimizer, _swap_config())

        self.assertIn(param, wrapped.state)
        self.assertIn((id(param), "momentum_buffer"), wrapped.adapter._slots)

    def test_step_creates_state_only_for_locally_updated_params(self):
        """A rank without local ownership allocates no momentum buffer."""
        owner = torch.nn.Parameter(torch.ones(8, 8))
        non_owner = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([owner, non_owner])
        adapter, _ = _adapter(optimizer)
        owner.grad = torch.ones_like(owner)
        non_owner.grad = torch.ones_like(non_owner)
        _install_schedule(
            optimizer,
            assignments=[_make_assignment([owner], is_shard=True)],
            no_comm=[],
        )

        adapter.prepare_step()

        self.assertIn("momentum_buffer", optimizer.state[owner])
        self.assertNotIn(non_owner, optimizer.state)

    def test_load_checkpoint_restores_only_locally_owned_state(self):
        """A full HSDP checkpoint does not recreate non-owner momentum slots."""
        owner = torch.nn.Parameter(torch.ones(8, 8))
        non_owner = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([owner, non_owner])
        _install_schedule(
            optimizer,
            assignments=[_make_assignment([owner], is_shard=True)],
            no_comm=[],
        )
        wrapped = swap_muon(optimizer, _swap_config(min_numel=1))
        wrapped.runtime.is_swappable_tensor = mock.Mock(return_value=True)
        checkpoint = {
            "state": {
                0: {"momentum_buffer": torch.full_like(owner, 1.0)},
                1: {"momentum_buffer": torch.full_like(non_owner, 2.0)},
            },
            "param_groups": optimizer.state_dict()["param_groups"],
        }

        wrapped.load_state_dict(checkpoint)

        self.assertIn(owner, wrapped.state)
        self.assertNotIn(non_owner, wrapped.state)
        self.assertEqual({param_id for param_id, _ in wrapped.adapter._slots}, {id(owner)})
        owner_slot = wrapped.adapter._slots[(id(owner), "momentum_buffer")]
        self.assertTrue(torch.equal(owner_slot.cpu_tensor, checkpoint["state"][0]["momentum_buffer"]))

    def test_units_follow_assignment_order_with_one_flush_each(self):
        """Every assignment keeps its schedule slot and broadcasts exactly once."""
        params = [torch.nn.Parameter(torch.ones(8, 8)) for _ in range(4)]
        optimizer = _muon(params)
        adapter, _ = _adapter(optimizer)
        for param in params:
            param.grad = torch.ones_like(param)
        first = _make_assignment(params[:2], is_shard=True)
        second = _make_assignment(params[2:], is_shard=True)
        _install_schedule(optimizer, assignments=[first, second], no_comm=[])

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual([unit.hsdp_assign for unit in units], [first, second])
        self.assertEqual([unit.flushes_broadcast for unit in units], [True, True])

    def test_no_comm_units_precede_hsdp_units(self):
        """Locally orthonormalized params run before the HSDP schedule."""
        no_comm = torch.nn.Parameter(torch.ones(8, 8))
        owned = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([no_comm, owned])
        adapter, _ = _adapter(optimizer)
        no_comm.grad = torch.ones_like(no_comm)
        owned.grad = torch.ones_like(owned)
        assignment = _make_assignment([owned], is_shard=True)
        _install_schedule(optimizer, assignments=[assignment], no_comm=[no_comm])

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual([unit.kind for unit in units], ["no_comm", "hsdp"])
        self.assertIs(units[1].hsdp_assign, assignment)

    def test_empty_ownership_keeps_the_schedule_position_without_slots(self):
        """A rank that owns nothing still walks the collective schedule."""
        optimizer = _muon([torch.nn.Parameter(torch.ones(8, 8))])
        adapter, _ = _adapter(optimizer)
        _install_schedule(
            optimizer, assignments=[_make_assignment([], is_shard=True)], no_comm=[]
        )

        units = adapter.iter_update_units(adapter.prepare_step())

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].params, [])
        self.assertEqual(units[0].slots, [])
        self.assertTrue(units[0].flushes_broadcast)
        self.assertEqual(tuple(adapter.all_slots()), ())

    def test_wrapper_broadcasts_with_materialized_state(self):
        """The broadcast reaches the wrapped Muon with live device storage."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([param])
        wrapped = swap_muon(optimizer, _swap_config(min_numel=1))
        param.grad = torch.ones_like(param)
        wrapped.step()
        seen = []
        optimizer._broadcast_state_fused_for_ckpt = mock.Mock(
            side_effect=lambda: seen.append(
                wrapped.adapter._slots[(id(param), "momentum_buffer")].state
            )
        )
        optimizer.cleanup_synced_state = mock.Mock()

        wrapped._broadcast_state_fused_for_ckpt()

        self.assertEqual(seen, ["device"])
        self.assertTrue(wrapped.adapter._checkpoint_active)

        wrapped.cleanup_synced_state()

        self.assertFalse(wrapped.adapter._checkpoint_active)
        optimizer.cleanup_synced_state.assert_called_once_with()

    def test_broadcast_failure_restores_the_swapped_lifecycle(self):
        """A failing broadcast leaves no active checkpoint window behind."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = _muon([param])
        wrapped = swap_muon(optimizer, _swap_config(min_numel=1))
        optimizer._broadcast_state_fused_for_ckpt = mock.Mock(side_effect=RuntimeError("boom"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            wrapped._broadcast_state_fused_for_ckpt()

        self.assertFalse(wrapped.adapter._checkpoint_active)

    def test_checkpoint_hooks_are_skipped_without_a_state_broadcast(self):
        """A wrapped optimizer that cannot broadcast state keeps swap hooks inert."""
        param = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = torch.optim.SGD([param], lr=0.01)
        wrapped = CoreSwapOptimizer(
            optimizer,
            _swap_config(min_numel=1),
            adapter_factory=lambda opt, cfg, runtime: mock.Mock(
                validate=mock.Mock(),
                initial_slots=mock.Mock(return_value=()),
                publish_packed_state=mock.Mock(),
                prepare_checkpoint=mock.Mock(),
                finish_checkpoint=mock.Mock(),
            ),
        )

        wrapped._prepare_swap_checkpoint()
        wrapped._broadcast_state_fused_for_ckpt()

        wrapped.adapter.prepare_checkpoint.assert_not_called()
        wrapped.adapter.finish_checkpoint.assert_not_called()


class TestMuonSwapParameterReset(unittest.TestCase):
    """Parameter replacement must update both the schedule and persistent state."""

    @staticmethod
    def _mixed_optimizer(use_swap):
        """Build the production main-parameter wrapper around two optimizer leaves."""
        model = torch.nn.Linear(4, 4).to(torch.bfloat16)
        with torch.no_grad():
            model.weight.fill_(1)
            model.bias.fill_(1)
        muon = _muon([model.weight])
        adam = torch.optim.AdamW([model.bias], lr=0.01, foreach=False)
        if use_swap:
            muon = swap_optimizer(muon, _swap_config())
            adam = swap_optimizer(adam, _swap_config())
        chain = ChainedOptimizer(model, {"muon": muon, "adamw": adam}, flatten=True)
        return model, Float16OptimizerWithFloat16Params(chain, model)

    def test_mixed_precision_chain_updates_main_params_and_resumes_checkpoint(self):
        """A wrapped Muon updates FP32 main params and retains FQN checkpoint state."""
        bare_model, bare = self._mixed_optimizer(False)
        model, wrapped = self._mixed_optimizer(True)
        muon = wrapped.optimizers_dict["muon"]
        main = model.weight.main_param
        self.assertIs(muon.optimizer.unshard_params_by_group[0][0], main)
        self.assertIs(muon.optimizer._hsdp_grouping[0][0][0], main)
        self.assertIs(muon.optimizer._hsdp_assignment_batches[0]["no_comm"][0], main)
        self.assertFalse(muon.adapter._slots)
        initial = main.detach().clone()
        for step in range(3):
            for current_model, optimizer in ((bare_model, bare), (model, wrapped)):
                for param in current_model.parameters():
                    param.grad = torch.full_like(param, step + 1)
                optimizer.step()
                optimizer.zero_grad()
            for expected, actual in zip(bare_model.parameters(), model.parameters()):
                torch.testing.assert_close(actual.main_param, expected.main_param, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if step == 0:
                checkpoint = wrapped.state_dict()
                self.assertIn("state.weight.momentum_buffer", checkpoint)
                self.assertIn("state.bias.exp_avg", checkpoint)
                wrapped.load_state_dict(checkpoint)
        self.assertFalse(torch.equal(main, initial))
        self.assertEqual(set(muon.adapter._slots), {(id(main), "momentum_buffer")})
        self.assertNotIn(model.weight, muon.state)
        self.assertIsNot(muon.state, wrapped.optimizers_dict["adamw"].state)

    def test_offloaded_momentum_survives_dtype_and_parameter_replacement(self):
        """The authoritative host mirror survives a released old device shell."""
        old = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        optimizer = _muon([old])
        expected = torch.arange(16, dtype=torch.bfloat16).reshape_as(old)
        optimizer.state[old]["momentum_buffer"] = expected.clone()
        wrapped = swap_muon(optimizer, _swap_config())
        old_slot = wrapped.adapter._slots[(id(old), "momentum_buffer")]
        old_slot.cpu_tensor = expected.clone()
        old_slot.swappable = True
        old_slot.state = "host"
        old_slot.tensor.untyped_storage().resize_(0)
        new = torch.nn.Parameter(torch.detach(old).float())
        wrapped.param_groups[0]["params"] = [new]
        wrapped.state[new] = wrapped.state.pop(old)
        self.assertEqual(tuple(wrapped.adapter.all_slots()), ())
        with mock.patch.object(wrapped.runtime, "current_stream", return_value=None):
            wrapped.reset_optimizer_parameters({new: old})
        self.assertEqual(set(wrapped.adapter._slots), {(id(new), "momentum_buffer")})
        self.assertIs(optimizer.unshard_params_by_group[0][0], new)
        torch.testing.assert_close(wrapped.state[new]["momentum_buffer"], expected.float(), rtol=0, atol=0)
        self.assertEqual(wrapped.adapter._slots[(id(new), "momentum_buffer")].dtype, torch.float32)
        new.grad = torch.ones_like(new)
        wrapped.step()
        torch.testing.assert_close(
            wrapped.state[new]["momentum_buffer"], expected.float() * 0.95 + 1, rtol=0, atol=0,
        )

    def test_reset_drains_removed_slots_and_keeps_unchanged_state(self):
        """Slots removed from param_groups are still synchronized before disposal."""
        old, retained, removed = [torch.nn.Parameter(torch.ones(4, 4)) for _ in range(3)]
        optimizer = _muon([old, retained, removed])
        for index, param in enumerate((old, retained, removed)):
            optimizer.state[param]["momentum_buffer"] = torch.full_like(param, index + 1)
        wrapped = swap_muon(optimizer, _swap_config())
        old_slots = tuple(wrapped.adapter.all_slots())
        for slot in old_slots:
            slot.cpu_tensor = slot.tensor.clone()
            slot.tensor.untyped_storage().resize_(0)
            slot.state = "host"
            slot.swappable = True
            slot.event = mock.Mock()
        events = [slot.event for slot in old_slots]
        new = torch.nn.Parameter(torch.ones_like(old))
        wrapped.param_groups[0]["params"] = [new, retained]
        with mock.patch.object(wrapped.runtime, "current_stream", return_value=None):
            wrapped.reset_optimizer_parameters({new: old})
        for event in events:
            event.synchronize.assert_called_once_with()
        self.assertEqual(set(wrapped.state), {new, retained})
        self.assertEqual(set(wrapped.adapter._slots), {
            (id(new), "momentum_buffer"), (id(retained), "momentum_buffer"),
        })
        torch.testing.assert_close(wrapped.state[new]["momentum_buffer"], torch.ones_like(new))
        self.assertIs(wrapped.adapter._slots[(id(retained), "momentum_buffer")], old_slots[1])
        self.assertTrue(all(slot.event is None for slot in old_slots))

    @mock.patch.object(DeviceMesh, "get_group", return_value=None)
    @mock.patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
    def test_dtensor_replacement_preserves_logical_state_layout(self, get_rank, get_group):
        """New momentum shells carry the FP32 main parameter's DTensor metadata."""
        del get_rank, get_group
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dp",), _init_backend=False)
        old = torch.nn.Parameter(DTensor.from_local(torch.ones(4, 4, dtype=torch.bfloat16), mesh, (Replicate(),)))
        optimizer = _muon([old])
        optimizer.state[old]["momentum_buffer"] = DTensor.from_local(
            torch.ones(4, 4, dtype=torch.bfloat16), mesh, (Replicate(),),
        )
        wrapped = swap_muon(optimizer, _swap_config())
        new = torch.nn.Parameter(DTensor.from_local(old.to_local().detach().float(), mesh, (Replicate(),)))
        wrapped.param_groups[0]["params"] = [new]
        with SkipDTensorDispatch():
            wrapped.reset_optimizer_parameters({new: old})
        momentum = wrapped.state[new]["momentum_buffer"]
        self.assertIsInstance(momentum, DTensor)
        self.assertIs(momentum.device_mesh, mesh)
        self.assertEqual(momentum.placements, new.placements)
        self.assertEqual(momentum.dtype, torch.float32)
        torch.testing.assert_close(momentum.to_local(), torch.ones(4, 4))

    def test_reset_filters_state_using_rebuilt_hsdp_ownership(self):
        """Non-owner state is removed only after the new assignment is available."""
        old_params = [torch.nn.Parameter(torch.ones(4, 4)) for _ in range(2)]
        optimizer = _muon(old_params)
        for param in old_params:
            optimizer.state[param]["momentum_buffer"] = torch.ones_like(param)
        wrapped = swap_muon(optimizer, _swap_config())
        new_params = [torch.nn.Parameter(param.detach().clone()) for param in old_params]
        wrapped.param_groups[0]["params"] = new_params
        assignment = _make_assignment(new_params, False, owners=[(0,), (1,)])
        with mock.patch.object(optimizer, "reset_optimizer_parameters", side_effect=lambda: _install_schedule(
                optimizer, assignments=[assignment])) as reset:
            wrapped.reset_optimizer_parameters(dict(zip(new_params, old_params)))
        reset.assert_called_once_with()
        self.assertEqual(set(wrapped.state), {new_params[0]})
        self.assertEqual(set(wrapped.adapter._slots), {(id(new_params[0]), "momentum_buffer")})
        context = wrapped.adapter.prepare_step()
        self.assertIs(context["units"][0].hsdp_assign, assignment)

    def test_reset_rejects_active_step_checkpoint_and_transfer(self):
        """Calling reset inside an active lifecycle fails before rebuilding metadata."""
        param = torch.nn.Parameter(torch.ones(4, 4))
        wrapped = swap_muon(_muon([param]), _swap_config())
        param.grad = torch.ones_like(param)
        original = wrapped.optimizer.reset_optimizer_parameters

        def reset_during_step(batch: list, context: dict) -> None:
            """Exercise a reset attempt while the actual wrapper step is active."""
            del batch, context
            with self.assertRaisesRegex(RuntimeError, "during a step, checkpoint or transfer"):
                wrapped.reset_optimizer_parameters()

        with mock.patch.object(wrapped.adapter, "step_batch", side_effect=reset_during_step), \
                mock.patch.object(wrapped.optimizer, "reset_optimizer_parameters", wraps=original) as reset:
            wrapped.step()
            reset.assert_not_called()
        wrapped._prepare_swap_checkpoint()
        try:
            with self.assertRaisesRegex(RuntimeError, "during a step, checkpoint or transfer"):
                wrapped.reset_optimizer_parameters()
        finally:
            wrapped.cleanup_synced_state()
        slot = next(iter(wrapped.adapter.all_slots()))
        for state in ("h2d", "d2h"):
            slot.state = state
            with self.assertRaisesRegex(RuntimeError, "during a step, checkpoint or transfer"):
                wrapped.reset_optimizer_parameters()
        slot.state = "device"
        wrapped.reset_optimizer_parameters()

    def test_failed_reset_retains_mirrors_and_can_retry_with_mapping(self):
        """An omitted mapping cannot silently treat a released shell as fresh state."""
        old = torch.nn.Parameter(torch.ones(4, 4))
        optimizer = _muon([old])
        optimizer.state[old]["momentum_buffer"] = torch.ones_like(old)
        wrapped = swap_muon(optimizer, _swap_config())
        old_slot = wrapped.adapter._slots[(id(old), "momentum_buffer")]
        new = torch.nn.Parameter(torch.detach(old).clone())
        wrapped.param_groups[0]["params"] = [new]
        wrapped.state[new] = wrapped.state.pop(old)
        with self.assertRaisesRegex(ValueError, "explicit new_to_old_param"):
            wrapped.reset_optimizer_parameters()
        self.assertIs(wrapped.adapter._slots[(id(old), "momentum_buffer")], old_slot)
        with self.assertRaisesRegex(RuntimeError, "reset has not completed"):
            wrapped.step()
        with self.assertRaisesRegex(RuntimeError, "reset has not completed"):
            wrapped.state_dict()
        wrapped.reset_optimizer_parameters({new: old})
        new.grad = torch.ones_like(new)
        wrapped.step()
        torch.testing.assert_close(wrapped.state[new]["momentum_buffer"], torch.full_like(new, 1.95))

    def test_reset_preserves_each_leafs_existing_state(self):
        """The shared new-to-old map cannot move momentum into an Adam leaf."""
        model = torch.nn.Linear(4, 4).to(torch.bfloat16)
        muon = _muon([model.weight])
        muon.state[model.weight]["momentum_buffer"] = torch.full_like(model.weight, 2)
        adam = torch.optim.AdamW([model.bias], lr=0.01, foreach=False)
        adam.state[model.bias] = {
            "step": torch.tensor(3.), "exp_avg": torch.full_like(model.bias, 4),
            "exp_avg_sq": torch.full_like(model.bias, 5),
        }
        muon = swap_optimizer(muon, _swap_config())
        adam = swap_optimizer(adam, _swap_config())
        chain = ChainedOptimizer(model, {"adamw": adam, "muon": muon}, flatten=True)
        Float16OptimizerWithFloat16Params(chain, model)
        torch.testing.assert_close(muon.state[model.weight.main_param]["momentum_buffer"], torch.full((4, 4), 2.))
        torch.testing.assert_close(adam.state[model.bias.main_param]["exp_avg"], torch.full((4,), 4.))
        self.assertEqual(set(muon.state), {model.weight.main_param})
        self.assertEqual(set(adam.state), {model.bias.main_param})
        self.assertEqual(set(adam.adapter._slots), {
            (id(model.bias.main_param), "exp_avg"), (id(model.bias.main_param), "exp_avg_sq"),
        })
        self.assertIsNot(muon.runtime, adam.runtime)

    def test_reset_failure_preserves_old_slots_until_retry(self):
        """A failed synchronization or shape conversion cannot discard momentum."""
        old = torch.nn.Parameter(torch.ones(4, 4))
        optimizer = _muon([old])
        optimizer.state[old]["momentum_buffer"] = torch.full_like(old, 2)
        wrapped = swap_muon(optimizer, _swap_config())
        old_slot = wrapped.adapter._slots[(id(old), "momentum_buffer")]
        new = torch.nn.Parameter(torch.ones(3, 4))
        wrapped.param_groups[0]["params"] = [new]
        with mock.patch.object(wrapped.runtime, "synchronize_cpu_mirrors", side_effect=RuntimeError("copy failed")), \
                mock.patch.object(optimizer, "reset_optimizer_parameters") as reset:
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                wrapped.reset_optimizer_parameters({new: old})
            reset.assert_not_called()
        with self.assertRaisesRegex(ValueError, "local shape"):
            wrapped.reset_optimizer_parameters({new: old})
        self.assertIs(wrapped.adapter._slots[(id(old), "momentum_buffer")], old_slot)
        torch.testing.assert_close(wrapped.state[old]["momentum_buffer"], torch.full_like(old, 2))
        replacement = torch.nn.Parameter(torch.ones(2, 8))
        wrapped.param_groups[0]["params"] = [replacement]
        wrapped.reset_optimizer_parameters({replacement: old})
        torch.testing.assert_close(wrapped.state[replacement]["momentum_buffer"], torch.full((2, 8), 2.))

    def test_load_context_restores_existing_state_on_fqn_error(self):
        """DCP validation failure must not leave temporary schema state installed."""
        model = torch.nn.Linear(4, 4, bias=False)
        wrapped = swap_muon(_muon(model.parameters()), _swap_config())
        chain = ChainedOptimizer(model, {"muon": wrapped}, flatten=True)
        model.weight.grad = torch.ones_like(model.weight)
        chain.step()
        chain.zero_grad()
        checkpoint = chain.state_dict()
        previous_state = wrapped.state
        with mock.patch("hyper_parallel.core.optimizer.optimizer.set_optimizer_state_dict",
                        side_effect=ValueError("invalid FQN")):
            with self.assertRaisesRegex(ValueError, "invalid FQN"):
                chain.load_state_dict(checkpoint)
        self.assertIs(wrapped.state, previous_state)
        wrapped.reset_optimizer_parameters()

    def test_fresh_and_empty_owner_chain_load_never_steps(self):
        """Loading flat FQN state on an empty owner cannot start HSDP collectives."""
        for flatten in (False, True):
            model = torch.nn.Linear(4, 4, bias=False)
            source = swap_muon(_muon(model.parameters()), _swap_config())
            chain = ChainedOptimizer(model, {"muon": source}, flatten=flatten)
            model.weight.grad = torch.ones_like(model.weight)
            chain.step()
            chain.zero_grad()
            checkpoint = chain.state_dict()
            for owns_param in (False, True):
                with self.subTest(flatten=flatten, owns_param=owns_param):
                    target = swap_muon(_muon(model.parameters()), _swap_config())
                    _install_schedule(target.optimizer, assignments=[_make_assignment(
                        [model.weight], False, owners=[(0,) if owns_param else (1,)],
                    )])
                    target_chain = ChainedOptimizer(model, {"muon": target}, flatten=flatten)
                    with mock.patch.object(target, "step", side_effect=AssertionError("load called step")):
                        target_chain.load_state_dict(checkpoint)
                    self.assertEqual(set(target.state), {model.weight} if owns_param else set())
                    if owns_param:
                        torch.testing.assert_close(
                            target.state[model.weight]["momentum_buffer"], torch.ones_like(model.weight),
                        )


class TestChainedOptimizerSwapCheckpoint(unittest.TestCase):
    """Checkpoint lifecycle and tensor ownership for chained swap leaves."""

    @staticmethod
    def _wrapped_muon():
        param = torch.nn.Parameter(torch.ones(8, 8))
        return swap_muon(_muon([param]), _swap_config()), param

    @mock.patch("hyper_parallel.core.optimizer.optimizer.get_optimizer_state_dict")
    def test_flattened_swap_state_is_copied_before_cleanup(self, get_state):
        """Flattened checkpoint values do not alias storage released by cleanup."""
        wrapped, param = self._wrapped_muon()
        source = torch.arange(param.numel(), dtype=torch.float32).reshape_as(param)
        get_state.return_value = {
            "state": {"weight": {"momentum_buffer": source}},
            "param_groups": [{"params": ["weight"]}],
        }
        wrapped._prepare_swap_checkpoint = mock.Mock()
        wrapped._broadcast_state_fused_for_ckpt = mock.Mock()
        wrapped.cleanup_synced_state = mock.Mock()
        chain = ChainedOptimizer(torch.nn.Linear(8, 8, bias=False), {"muon": wrapped}, flatten=True)

        state_dict = chain.state_dict()

        saved = state_dict["state"]["weight"]["momentum_buffer"]
        self.assertTrue(torch.equal(saved, source))
        self.assertIsNot(saved, source)
        self.assertEqual(saved.device.type, "cpu")
        self.assertTrue(get_state.call_args.kwargs["options"].flatten_optimizer_state_dict)
        wrapped._prepare_swap_checkpoint.assert_called_once_with()
        wrapped._broadcast_state_fused_for_ckpt.assert_called_once_with()
        wrapped.cleanup_synced_state.assert_called_once_with()

    @mock.patch("hyper_parallel.core.optimizer.optimizer.get_optimizer_state_dict")
    def test_chained_checkpoint_cleanup_runs_when_export_fails(self, get_state):
        """A failing state export still closes every swap checkpoint window."""
        wrapped, _ = self._wrapped_muon()
        get_state.side_effect = RuntimeError("export failed")
        wrapped._prepare_swap_checkpoint = mock.Mock()
        wrapped._broadcast_state_fused_for_ckpt = mock.Mock()
        wrapped.cleanup_synced_state = mock.Mock()
        chain = ChainedOptimizer(torch.nn.Linear(8, 8, bias=False), {"muon": wrapped})

        with self.assertRaisesRegex(RuntimeError, "export failed"):
            chain.state_dict()

        wrapped.cleanup_synced_state.assert_called_once_with()

    def test_real_export_broadcasts_and_cleans_up_once(self):
        """Nested wrapper export reuses the checkpoint window owned by the chain."""
        model = torch.nn.Linear(8, 8, bias=False)
        optimizer = _muon(model.parameters())
        optimizer._broadcast_state_fused_for_ckpt = mock.Mock()
        optimizer.cleanup_synced_state = mock.Mock()
        wrapped = swap_muon(optimizer, _swap_config())
        chain = ChainedOptimizer(model, {"muon": wrapped})

        chain.state_dict()

        optimizer._broadcast_state_fused_for_ckpt.assert_called_once_with()
        optimizer.cleanup_synced_state.assert_called_once_with()
        self.assertFalse(wrapped.adapter._checkpoint_active)
        self.assertFalse(wrapped.adapter._checkpoint_broadcasted)


if __name__ == "__main__":
    unittest.main()
