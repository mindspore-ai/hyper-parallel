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
"""Recovery state integrity without distributed initialization or accelerators."""

import random
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from hyper_parallel import Replicate, Shard
from hyper_parallel.high_availability.state import ModelOptimizerState, TrainState
from hyper_parallel.high_availability.adapters import ModelRecoveryAdapter
from hyper_parallel.core.activation_memory.api import checkpoint as hyper_checkpoint
from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state
from hyper_parallel.distributed.activation_checkpoint import make_selective_checkpoint_context_fn
from hyper_parallel.high_availability.groups import GroupRegistry
from hyper_parallel.high_availability.topology import ReplicaTopology, model_replica_candidates
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.swap_optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.sharding_category import (
    CommDomainKey, HSDPCommGroup, ParamLayoutSpec, ParamRecord,
)


class StatefulValue:
    """Minimal consumed-data cursor or scheduler provider."""

    def __init__(self, value: int = 0) -> None:
        """Set an initial consumed position."""
        self.value = value

    def state_dict(self) -> dict[str, int]:
        """Export an independent cursor."""
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore a cursor."""
        self.value = state["value"]


class TestDeviceState(unittest.TestCase):
    """Device repair manifests contain no parameter or optimizer moment values."""

    def test_swap_repair_uses_receiver_slot_policy(self):
        """Recover both host and device slots from identical wire metadata."""
        for optimizer_type, key in ((AdamW, "exp_avg"), (Muon, "momentum_buffer")):
            for packed in (False, True):
                with self.subTest(optimizer=optimizer_type.__name__, packed=packed):
                    config = SwapOptimizerConfig(min_numel=1, packed_swap=packed)
                    source_model = torch.nn.Linear(4, 4, bias=False)
                    target_model = torch.nn.Linear(4, 4, bias=False)
                    source = swap_optimizer(optimizer_type(source_model.parameters()), config)
                    target = swap_optimizer(optimizer_type(target_model.parameters()), config)
                    # CPU tests emulate accelerator eligibility while exercising the real slot bindings.
                    with (
                        patch.object(source.runtime, "is_swappable_tensor", return_value=True),
                        patch.object(target.runtime, "is_swappable_tensor") as target_can_swap,
                        torch.no_grad(),
                    ):
                        source.state[source_model.weight][key] = source.adapter.allocate_recovery_state(
                            source_model.weight, key, torch.float32,
                        )
                        source.state[source_model.weight]["step"] = torch.tensor(7.0)
                        source_state = ModelOptimizerState(source_model, source)
                        target_state = ModelOptimizerState(target_model, target)
                        if packed:
                            # A tracked host mirror can supply a key absent from the leaf state.
                            source.state[source_model.weight].pop(key)
                        metadata = source_state.metadata()
                        entries = metadata["optimizers"][0][0]["states"]["weight"]
                        self.assertEqual(entries[key], {"tensor": True, "shape": (4, 4), "dtype": torch.float32})
                        self.assertEqual(entries["step"], {"tensor": True, "shape": (), "dtype": torch.float32})
                        source_mirror = source.adapter.recovery_state(source_model.weight)[key]
                        for can_swap, expected in ((True, 3.0), (False, 4.0), (True, 5.0)):
                            target_can_swap.return_value = can_swap
                            source_mirror.fill_(expected)
                            target_state.prepare_receive(metadata)
                            destination = dict(target_state.tensor_items())
                            for name, value in source_state.tensor_items():
                                destination[name].copy_(value)
                            slot, = target.adapter.all_slots()
                            restored = destination[f"state:0:weight:{key}"]
                            self.assertEqual(slot.swappable, can_swap)
                            self.assertEqual(slot.state, "host" if can_swap else "device")
                            self.assertIs(restored, slot.cpu_tensor if can_swap else slot.tensor)
                            torch.testing.assert_close(restored, torch.full((4, 4), expected))
                            self.assertEqual(target.state[target_model.weight]["step"].item(), 7.0)

    def test_swap_recovery_rejects_pending_mirror_event(self):
        """Reject donor metadata while a mirror still has a transfer event."""
        model = torch.nn.Linear(4, 4, bias=False)
        optimizer = swap_optimizer(AdamW(model.parameters()), SwapOptimizerConfig(min_numel=1))
        with patch.object(optimizer.runtime, "is_swappable_tensor", return_value=True):
            optimizer.state[model.weight]["exp_avg"] = optimizer.adapter.allocate_recovery_state(
                model.weight, "exp_avg", torch.float32,
            )
        slot, = optimizer.adapter.all_slots()
        slot.event = Mock()
        model_state = ModelOptimizerState(model, optimizer)
        with self.assertRaisesRegex(RuntimeError, "has not finished"):
            model_state.metadata()

    def test_manifest_and_receive_slots_preserve_live_parameter_identity(self):
        """Restore model and optimizer state without replacing persistent parameter objects."""
        model = torch.nn.Linear(4, 4)
        source = Muon([model.weight], ns_steps=2)
        for parameter in source.param_groups[0]["params"]:
            parameter.grad = torch.ones_like(parameter)
        source.step()
        source_state = ModelOptimizerState(model, source)
        metadata = source_state.metadata()
        self.assertNotIn("parameters", metadata)
        self.assertNotIn("buffers", metadata)
        moment = metadata["optimizers"][0][0]["states"]["weight"]["momentum_buffer"]
        self.assertEqual(moment, {"tensor": True, "shape": (4, 4), "dtype": torch.float32})
        items = dict(source_state.tensor_items())
        self.assertEqual(items["parameter:weight"].data_ptr(), model.weight.data_ptr())
        self.assertEqual(items["state:0:weight:momentum_buffer"].data_ptr(),
                         source.state[model.weight]["momentum_buffer"].data_ptr())
        target = torch.nn.Linear(4, 4)
        optimizer = Muon([target.weight], ns_steps=2)
        receiver = ModelOptimizerState(target, optimizer)
        identity = id(target.weight)
        receiver.prepare_receive(metadata)
        self.assertEqual(identity, id(target.weight))
        self.assertEqual(tuple(optimizer.state[target.weight]["momentum_buffer"].shape), (4, 4))

    def test_cp_fsdp_shards_require_a_separate_complete_replica(self) -> None:
        """Feature: CP combined with FSDP does not imply parameter redundancy.

        Description: Infer replicas from an eight-rank replicate/shard/CP mesh.
        Expectation: CP-sharded coordinates stay fixed; only the replicated axis varies.
        """
        mesh = SimpleNamespace(mesh_shape=(2, 2, 2), rank_list=tuple(range(8)))
        parameter = SimpleNamespace(device_mesh=mesh, placements=(Replicate(), Shard(0), Shard(1)))
        state = SimpleNamespace(parameters={"weight": parameter})
        profiles = []
        for rank in range(8):
            peers = model_replica_candidates(state, rank, 8)
            self.assertEqual(peers, (rank % 4, rank % 4 + 4))
            profiles.append({"candidates": peers, "schema": "cp-fsdp", "optimizer_ownership": "adamw"})
        self.assertEqual(ReplicaTopology.from_profiles(profiles).replica_groups,
                         ((0, 4), (1, 5), (2, 6), (3, 7)))
        mesh.mesh_shape = (1, 2, 2)
        mesh.rank_list = tuple(range(4))
        profiles = [{"candidates": model_replica_candidates(state, rank, 4),
                     "schema": "cp-fsdp", "optimizer_ownership": "adamw"} for rank in range(4)]
        with self.assertRaisesRegex(ValueError, "no equivalent device replica"):
            ReplicaTopology.from_profiles(profiles)

    def test_optimizer_family_cannot_change_across_repair(self):
        model = torch.nn.Linear(4, 4, bias=False)
        metadata = ModelOptimizerState(model, AdamW(model.parameters())).metadata()
        with self.assertRaisesRegex(RuntimeError, "schema/layout"):
            ModelOptimizerState(model, Muon(model.parameters())).prepare_receive(metadata)

    def test_receive_rejects_a_different_optimizer_state_owner(self):
        model = torch.nn.Linear(4, 4, bias=False)
        state = ModelOptimizerState(model, AdamW(model.parameters()))
        metadata = state.metadata()
        metadata["optimizer_ownership"] = "another-state-partition"
        with self.assertRaisesRegex(RuntimeError, "optimizer-state ownership mismatch"):
            state.prepare_receive(metadata)

    def test_rank_buffers_stay_on_their_original_device(self):
        model = torch.nn.Linear(4, 4)
        model.register_buffer("counter", torch.tensor([8]))
        state = ModelOptimizerState(model, AdamW(model.parameters()))
        saved = state.rank_state()
        model.counter.fill_(9)
        state.restore_rank_state(saved["buffers"], saved["extra"])
        self.assertEqual(model.counter.item(), 8)
        self.assertEqual(saved["buffers"]["counter"].device, model.counter.device)

    def test_reboot_defers_shape_collectives_until_groups_are_rebuilt(self):
        """Rebuild shard metadata after bootstrap and reuse the existing optimizer plan."""
        registry = GroupRegistry()
        registry.reboot = True
        model = torch.nn.Linear(4, 4, bias=False)
        with patch.object(Muon, "_build_param_shard_metadata") as build_metadata:
            with registry:
                optimizer = Muon(model.parameters())
            build_metadata.assert_not_called()
            batches = optimizer._hsdp_batches
            assignments = optimizer._hsdp_assignment_batches
            broadcast_info = optimizer._param_to_broadcast_info
            compute_coords = optimizer._shard_compute_coord
            ModelOptimizerState(model, optimizer).rebuild_runtime_state()
            build_metadata.assert_called_once()
            self.assertIs(optimizer._hsdp_batches, batches)
            self.assertIs(optimizer._hsdp_assignment_batches, assignments)
            self.assertIs(optimizer._param_to_broadcast_info, broadcast_info)
            self.assertIs(optimizer._shard_compute_coord, compute_coords)


class TestMuonReplicaOwnership(unittest.TestCase):
    """Model replica membership alone does not imply redundant Muon momentum."""

    @staticmethod
    @contextmanager
    def _replicated_muon(coordinate, replica_count, domain_size=2):
        """Build Muon with replicated parameters and an explicit state ownership plan."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False))
        mesh = SimpleNamespace(mesh_shape=(domain_size,), mesh_dim_names=("replicate",),
                               rank_list=tuple(range(domain_size)))
        parameters = list(model.parameters())
        for parameter in parameters:
            parameter.device_mesh = mesh
            parameter.placements = (Replicate(),)
        domain = CommDomainKey((domain_size,), tuple(range(domain_size)), (0,), ())
        layout = ParamLayoutSpec(2, (), (0,))

        def _group(optimizer):
            group = HSDPCommGroup(domain, layout, (object(),), (),
                                 [ParamRecord(index, param) for index, param in enumerate(parameters)])
            optimizer._hsdp_grouping = {0: ([], [group])}

        with ExitStack() as stack:
            stack.enter_context(patch.object(Muon, "_group_dtensor_by_mesh", _group))
            stack.enter_context(patch.object(Muon, "_build_param_shard_metadata"))
            stack.enter_context(patch.object(Muon, "_get_or_create_sub_pg", return_value=object()))
            stack.enter_context(patch("hyper_parallel.core.optimizer.optimizer.get_device_count", return_value=8))
            stack.enter_context(patch("hyper_parallel.core.optimizer.optimizer.dist.get_world_size",
                                      return_value=domain_size))
            stack.enter_context(patch("hyper_parallel.core.optimizer.optimizer.dist.get_process_group_ranks",
                                      return_value=list(range(domain_size))))
            stack.enter_context(patch("hyper_parallel.core.optimizer.optimizer.get_multi_dim_logical_info",
                                      return_value=((coordinate,), (domain_size,))))
            with GroupRegistry():
                optimizer = Muon(parameters, ns_steps=2, hsdp_replica_count=replica_count)
            yield model, optimizer

    def test_explicit_single_owner_group_keeps_all_local_momentum(self):
        for coordinate in (0, 1):
            with self.subTest(coordinate=coordinate), self._replicated_muon(coordinate, 1) as (model, optimizer):
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                ModelOptimizerState(model, optimizer)
                self.assertEqual(optimizer.hsdp_replica_count, 1)
                for parameter in model.parameters():
                    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"],
                                               torch.ones_like(parameter), rtol=0, atol=0)

    def test_implicit_state_dedup_leaves_no_complete_replica(self):
        """Reject model replicas whose optimizer state is completely deduplicated."""
        profiles = []
        for coordinate in (0, 1):
            with self.subTest(coordinate=coordinate), self._replicated_muon(coordinate, None) as (model, optimizer):
                self.assertIsNone(optimizer.hsdp_replica_count)
                assignment = optimizer._hsdp_assignment_batches[0]["batch_groups"][0]["sub_batches"][0]
                self.assertEqual(len(assignment.owned_params), 1)
                self.assertEqual(len(assignment.all_params), 2)
                state = ModelOptimizerState(model, optimizer)
                profiles.append({"schema": state.schema, "candidates": [0, 1],
                                 "optimizer_ownership": state.optimizer_ownership})
        with self.assertRaisesRegex(ValueError, "no complete device replica"):
            ReplicaTopology.from_profiles(profiles)

    def test_existing_muon_subgroups_leave_complete_replicas_across_groups(self):
        """Find complete state replicas across Muon's existing deduplication subgroups."""
        profiles = []
        for coordinate in range(4):
            with self.subTest(coordinate=coordinate), self._replicated_muon(coordinate, 2, 4) as (model, optimizer):
                self.assertEqual(optimizer.hsdp_replica_count, 2)
                assignment = optimizer._hsdp_assignment_batches[0]["batch_groups"][0]["sub_batches"][0]
                self.assertEqual(len(assignment.owned_params), 1)
                state = ModelOptimizerState(model, optimizer)
                profiles.append({"schema": state.schema, "candidates": list(range(4)),
                                 "optimizer_ownership": state.optimizer_ownership})
        self.assertEqual(ReplicaTopology.from_profiles(profiles).replica_groups, ((0, 2), (1, 3)))

    def test_replacement_derives_muon_owners_without_store_or_collectives(self):
        """Feature: Local full-replica inference after Muon's second partition.

        Description: Reconstruct each rank with a two-way state-owner subgroup over four model replicas.
        Expectation: Original owner fingerprints and replacement replica domains agree without global exchange.
        """
        for coordinate in range(4):
            with self.subTest(rank=coordinate), self._replicated_muon(coordinate, 2, 4) as (model, optimizer):
                state = ModelOptimizerState(model, optimizer)
                with patch("hyper_parallel.high_availability.topology.dist") as distributed:
                    topology = ReplicaTopology.discover(state, coordinate, 4, reboot=True)
                    self.assertEqual(topology.members(coordinate), (coordinate % 2, coordinate % 2 + 2))
                    distributed.all_gather_object.assert_not_called()

    def test_replacement_rejects_muon_without_a_complete_optimizer_replica(self):
        """Feature: Early rejection of nonredundant optimizer state.

        Description: Reconstruct Muon that deduplicates state across the entire model replica domain.
        Expectation: Local inference rejects recovery without contacting other ranks.
        """
        with self._replicated_muon(1, None) as (model, optimizer):
            with self.assertRaisesRegex(ValueError, "no complete device replica"):
                ReplicaTopology.discover(ModelOptimizerState(model, optimizer), 1, 2, reboot=True)


class TestTrainState(unittest.TestCase):
    """RNG and cursor restore must use the logical rank's exact state."""

    def test_grouped_snapshot_restores_rng_after_loader_and_replica_state(self):
        """Feature: Separate rank-owned and replica-equivalent snapshot groups.

        Description: Restore state with random-consuming hooks and then finish iterator reconstruction.
        Expectation: Both state groups restore before RNG/Generator resume at the committed position.
        """
        events = []
        generator = torch.Generator().manual_seed(42)

        class _ConsumingState(StatefulValue):
            """Exercise random-consuming provider restoration before the final RNG restore."""
            def load_state_dict(self, state: dict[str, int]) -> None:
                """Restore the value while exercising random-consuming initialization.

                Args:
                    state: Value saved at the committed step.
                """
                super().load_state_dict(state)
                events.append(self.value)
                random.random()
                np.random.rand()
                torch.rand(1)
                torch.rand(1, generator=generator)

        loader, scheduler = _ConsumingState(8), _ConsumingState(80)
        codec = TrainState(loader, {"scheduler": scheduler}, generators={"custom": generator})
        saved = codec.capture()
        self.assertEqual(set(saved), {"rank_state", "replica_state"})
        self.assertEqual(set(saved["rank_state"]), {"loader", "rng", "generators"})
        self.assertEqual(set(saved["rank_state"]["rng"]), {"python", "numpy", "torch", "device"})
        expected = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
        loader.value, scheduler.value = 9, 90
        codec.restore(saved)
        self.assertEqual(events, [8, 80])
        iter(())
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(3, generator=generator), expected[3], rtol=0, atol=0)

    def test_rng_failure_propagates_from_rollback_and_can_be_retried(self):
        """Feature: RNG is applied synchronously during training-state rollback.

        Description: Fail device RNG application, then repeat restore with the same committed snapshot.
        Expectation: Restore raises immediately on failure and reapplies the committed RNG on retry.
        """
        generator = torch.Generator().manual_seed(42)
        device = SimpleNamespace(get_rng_state=generator.get_state,
                                 set_rng_state=MagicMock(side_effect=RuntimeError("RNG restore failed")))
        codec = TrainState(StatefulValue(8), {}, device=device)
        saved = codec.capture()
        expected = torch.rand(3, generator=generator)
        with self.assertRaisesRegex(RuntimeError, "RNG restore failed"):
            codec.restore(saved)
        device.set_rng_state.side_effect = generator.set_state
        codec.restore(saved)
        self.assertEqual(device.set_rng_state.call_count, 2)
        torch.testing.assert_close(torch.rand(3, generator=generator), expected, rtol=0, atol=0)

    def test_normal_iterator_keeps_its_random_consumption(self):
        """Feature: Native RNG consumption when no recovery is pending.

        Description: Create two ordinary iterators whose constructors advance an explicit Generator.
        Expectation: Each constructor advances the stream without any recovery RNG read or write.
        """
        generator = torch.Generator().manual_seed(42)
        reference = torch.Generator().manual_seed(42)
        codec = TrainState(StatefulValue(8), {}, generators={"custom": generator})

        def _factory():
            torch.rand(3, generator=generator)
            return iter(())

        for _ in range(2):
            _factory()
            torch.rand(3, generator=reference)
            torch.testing.assert_close(torch.rand(3, generator=generator),
                                       torch.rand(3, generator=reference), rtol=0, atol=0)

    def test_rng_cursor_and_scheduler_restore(self):
        """Feature: Restore cursor, scheduler and RNG in one rollback.

        Description: Load committed state and immediately inspect the restored progress and random sequences.
        Expectation: Cursor, scheduler and subsequent random sequences match the commit.
        """
        loader, scheduler = StatefulValue(8), StatefulValue(80)
        codec = TrainState(loader, {"scheduler": scheduler})
        saved = codec.capture()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        loader.value, scheduler.value = 9, 90
        codec.restore(saved)
        self.assertEqual(loader.value, 8)
        self.assertEqual(scheduler.value, 80)
        iter(())
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_reentry_preserves_native_iterator_factory_and_rng(self):
        """Feature: Native iterator consumption after immediate RNG rollback.

        Description: Consume RNG before construction and forbid unrelated loader/provider state IO in the factory.
        Expectation: Native iterator and state mutations survive; post-rollback RNG consumption remains visible.
        """
        loader, scheduler = StatefulValue(8), StatefulValue(80)
        generator = torch.Generator().manual_seed(123)
        codec = TrainState(loader, {"scheduler": scheduler}, generators={"custom": generator})
        saved = codec.capture()
        for _ in range(2):
            random.random()
            np.random.rand()
            torch.rand(1)
            torch.rand(1, generator=generator)
        expected = (random.random(), np.random.rand(), torch.rand(3), torch.rand(3, generator=generator))
        codec.restore(saved)
        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, generator=generator)
        iterator = iter([8])

        def _factory():
            loader.value = 9
            scheduler.value = 90
            random.random()
            np.random.rand()
            torch.rand(1)
            torch.rand(1, generator=generator)
            return iterator

        with ExitStack() as stack:
            guards = [stack.enter_context(patch.object(value, method, side_effect=AssertionError("unrelated state IO")))
                      for value in (loader, scheduler) for method in ("state_dict", "load_state_dict")]
            self.assertIs(_factory(), iterator)
            for guard in guards:
                guard.assert_not_called()
        self.assertEqual(next(iterator), 8)
        self.assertEqual(loader.value, 9)
        self.assertEqual(scheduler.value, 90)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(3, generator=generator), expected[3], rtol=0, atol=0)

    def test_unrestorable_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state_dict"):
            TrainState(iter([1, 2]), {})

    def test_replica_provider_mapping_binds_objects_and_validates_schema(self):
        """Feature: One mapping of replica-equivalent state provider objects.

        Description: Register two objects without a second name list and restore an incomplete snapshot.
        Expectation: Both objects are captured, invalid providers and missing snapshot entries are rejected.
        """
        scheduler, progress = StatefulValue(80), StatefulValue(8)
        state = TrainState(StatefulValue(8), replica_providers={"scheduler": scheduler, "progress": progress})
        self.assertIs(state.replica_providers["scheduler"], scheduler)
        self.assertIs(state.replica_providers["progress"], progress)
        snapshot = state.capture()
        self.assertEqual(snapshot["replica_state"], {"scheduler": {"value": 80}, "progress": {"value": 8}})
        with self.assertRaisesRegex(ValueError, "state_dict/load_state_dict"):
            TrainState(StatefulValue(), replica_providers={"scheduler": object()})
        del snapshot["replica_state"]["progress"]
        with self.assertRaisesRegex(RuntimeError, "provider schema mismatch"):
            state.restore(snapshot)

    def test_custom_generator_and_refused_cursor(self):
        """Feature: Explicit Generator restore and rejected loader rollback.

        Description: Finalize a valid restore and then load into a loader that refuses state.
        Expectation: Generator resumes at the commit and the rejected cursor raises an error.
        """
        generator = torch.Generator().manual_seed(123)
        loader = StatefulValue(8)
        codec = TrainState(loader, {}, generators={"dropout": generator})
        saved = codec.capture()
        expected = torch.rand(4, generator=generator)
        codec.restore(saved)
        iter(())
        torch.testing.assert_close(torch.rand(4, generator=generator), expected, rtol=0, atol=0)
        loader.load_state_dict = lambda state: False
        with self.assertRaisesRegex(RuntimeError, "refused"):
            codec.restore(saved)

    def test_reentry_iterator_consumes_restored_rng_normally(self):
        """Feature: Iterator reconstruction after immediate RNG rollback.

        Description: Recreate a loader whose iterator consumes Python, NumPy and Torch random values.
        Expectation: Training draws follow the iterator constructor consumption of the committed sequence.
        """
        class _RandomIteratorLoader(StatefulValue):
            def __iter__(self) -> Iterator[int]:
                """Model the seed draw performed when a DataLoader creates an iterator."""
                random.random()
                np.random.rand()
                torch.rand(1)
                return iter([self.value])

        codec = TrainState(_RandomIteratorLoader(8), {})
        state = codec.capture()
        random.random()
        np.random.rand()
        torch.rand(1)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        codec.restore(state)
        iterator = iter(codec.loader)
        self.assertEqual(next(iterator), 8)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


class TestRecomputeRecovery(unittest.TestCase):
    """Exercise real checkpoint autograd and existing recovery state codecs after interrupted replay."""

    def _run(self, mode, phase, inject):
        """Execute the native training path with local state capture and simulated recovery."""
        torch.manual_seed(381)
        fault = SimpleNamespace(step=0, fired=False)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.25),
                                    torch.nn.Tanh(), torch.nn.Linear(8, 4))
        optimizer = AdamW(model.parameters(), lr=0.01)
        state = ModelOptimizerState(model, optimizer)
        loader = StatefulValue()
        train_state = TrainState(loader, {})
        adapter = ModelRecoveryAdapter(model)
        checkpoint_args = ({"context_fn": make_selective_checkpoint_context_fn()}
                           if mode == "selective" else {})

        def _raise_once(current):
            if inject and fault.step == 2 and phase == current and not fault.fired:
                fault.fired = True
                raise RuntimeError("injected checkpoint failure")

        def _forward(value):
            recomputing = (torch.is_grad_enabled() if mode == "reentrant"
                           else get_recompute_state().is_recomputing)
            result = model[1](model[0](value))
            _raise_once("recompute" if recomputing else "forward")
            return model[3](model[2](result))

        def _gradient_hook(gradient):
            _raise_once("backward")
            return gradient

        hook = model[0].weight.register_hook(_gradient_hook)

        def _iteration():
            loader.value += 1
            inputs = torch.randn(2, 4, requires_grad=True)
            if mode == "reentrant":
                output = torch_checkpoint(_forward, inputs, use_reentrant=True)
            else:
                output = hyper_checkpoint(_forward, inputs, **checkpoint_args)
            _raise_once("after_forward")
            output.square().sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        saved = {}
        try:
            for step in range(1, 4):
                fault.step = step
                try:
                    _iteration()
                except RuntimeError as error:
                    if str(error) != "injected checkpoint failure":
                        raise
                    self.assertTrue(fault.fired)
                    self.assertIsNone(get_recompute_state())
                    adapter.abort_iteration()
                    adapter.reset_after_device_restart(Mock())
                    state.prepare_receive(saved["metadata"])
                    with torch.no_grad():
                        for (name, target), (saved_name, source) in zip(state.tensor_items(), saved["tensors"]):
                            self.assertEqual(name, saved_name)
                            target.copy_(source)
                    rank_state = saved["rank_state"]
                    state.restore_rank_state(rank_state["buffers"], rank_state["extra"])
                    train_state.restore(saved["training"])
                    _iteration()
                if step == 1:
                    # Test-only committed reference values; production recovery uses a live donor.
                    saved = {
                        "metadata": state.metadata(),
                        "tensors": [(name, value.detach().clone()) for name, value in state.tensor_items()],
                        "rank_state": state.rank_state(),
                        "training": train_state.capture(),
                    }
        finally:
            hook.remove()
        self.assertIsNone(get_recompute_state())
        return state, loader.value, torch.get_rng_state(), fault.fired

    def test_recompute_recovery_matches_uninterrupted_training_in_each_phase(self) -> None:
        """Feature: Recovery reexecutes checkpointed iterations without stale contexts or RNG drift.

        Description: Interrupt forward, saved-graph lifetime, replay and backward for full/SAC/reentrant paths.
        Expectation: Model, optimizer, data position and RNG exactly match uninterrupted execution.
        """
        for mode in ("full", "selective", "reentrant"):
            for phase in ("forward", "after_forward", "recompute", "backward"):
                with self.subTest(mode=mode, phase=phase):
                    expected, expected_cursor, expected_rng, _ = self._run(mode, phase, False)
                    actual, cursor, rng, fired = self._run(mode, phase, True)
                    self.assertTrue(fired)
                    self.assertEqual(cursor, expected_cursor)
                    self.assertEqual(actual.metadata(), expected.metadata())
                    torch.testing.assert_close(rng, expected_rng, rtol=0, atol=0)
                    for (name, value), (expected_name, expected_value) in zip(
                            actual.tensor_items(), expected.tensor_items()):
                        self.assertEqual(name, expected_name)
                        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
