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
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hyper_parallel import Replicate
from hyper_parallel.core.fault_tolerance.state import ModelOptimizerState, TrainState
from hyper_parallel.core.fault_tolerance.groups import GroupRegistry
from hyper_parallel.core.fault_tolerance.topology import ReplicaTopology
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
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

    def test_manifest_and_receive_slots_preserve_live_parameter_identity(self):
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
        registry = GroupRegistry()
        registry.reboot = True
        model = torch.nn.Linear(4, 4, bias=False)
        with patch.object(Muon, "_build_param_shard_metadata") as build_metadata:
            with registry:
                optimizer = Muon(model.parameters())
            build_metadata.assert_not_called()
            ModelOptimizerState(model, optimizer).rebuild_runtime_state()
            build_metadata.assert_called_once()


class TestMuonReplicaOwnership(unittest.TestCase):
    """Model replica membership alone does not imply redundant Muon momentum."""

    @staticmethod
    @contextmanager
    def _replicated_muon(coordinate, replica_count, domain_size=2):
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
                with patch("hyper_parallel.core.fault_tolerance.topology.dist") as distributed:
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
        codec.make_iterator(factory=lambda: iter(()))
        codec.finish_restore()
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(3, generator=generator), expected[3], rtol=0, atol=0)

    def test_rng_is_applied_once_after_successful_iterator_reconstruction(self):
        """Feature: Deferred committed RNG across failed iterator reconstruction.

        Description: Stage device RNG, fail construction, retry and create another normal iterator.
        Expectation: Pending state survives construction failure and clears only after RNG is successfully applied.
        """
        generator = torch.Generator().manual_seed(42)
        device = SimpleNamespace(get_rng_state=generator.get_state,
                                 set_rng_state=MagicMock(side_effect=generator.set_state))
        codec = TrainState(StatefulValue(8), {}, device=device)
        self.assertFalse(codec.rng_restore_pending)
        saved = codec.capture()
        expected = torch.rand(3, generator=generator)
        codec.restore(saved)
        self.assertTrue(codec.rng_restore_pending)
        device.set_rng_state.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "iterator failed"):
            codec.make_iterator(factory=MagicMock(side_effect=RuntimeError("iterator failed")))
        self.assertTrue(codec.rng_restore_pending)
        device.set_rng_state.assert_not_called()
        codec.make_iterator(factory=lambda: iter(()))
        device.set_rng_state.assert_not_called()
        codec.finish_restore()
        self.assertFalse(codec.rng_restore_pending)
        device.set_rng_state.assert_called_once()
        torch.testing.assert_close(torch.rand(3, generator=generator), expected, rtol=0, atol=0)
        codec.make_iterator(factory=lambda: iter(()))
        codec.finish_restore()
        device.set_rng_state.assert_called_once()
        self.assertFalse(torch.equal(torch.rand(3, generator=generator), expected))

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
            codec.make_iterator(factory=_factory)
            codec.finish_restore()
            torch.rand(3, generator=reference)
            torch.testing.assert_close(torch.rand(3, generator=generator),
                                       torch.rand(3, generator=reference), rtol=0, atol=0)

    def test_rng_cursor_and_scheduler_restore(self):
        """Feature: Restore cursor and scheduler before finalizing RNG.

        Description: Load committed state and finish recovery through iterator construction.
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
        codec.make_iterator(factory=lambda: iter(()))
        codec.finish_restore()
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_reentry_preserves_native_iterator_factory_and_rng(self):
        """Feature: Committed RNG after recovery callbacks and native iterator construction.

        Description: Consume RNG before construction and forbid unrelated loader/provider state IO in the factory.
        Expectation: The native iterator and state mutations survive, while RNG resumes at the committed position.
        """
        loader, scheduler = StatefulValue(8), StatefulValue(80)
        generator = torch.Generator().manual_seed(123)
        codec = TrainState(loader, {"scheduler": scheduler}, generators={"custom": generator})
        saved = codec.capture()
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
            self.assertIs(codec.make_iterator(factory=_factory), iterator)
            for guard in guards:
                guard.assert_not_called()
        codec.finish_restore()
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
        codec.make_iterator(factory=lambda: iter(()))
        codec.finish_restore()
        torch.testing.assert_close(torch.rand(4, generator=generator), expected, rtol=0, atol=0)
        loader.load_state_dict = lambda state: False
        with self.assertRaisesRegex(RuntimeError, "refused"):
            codec.restore(saved)

    def test_reentry_iterator_does_not_consume_global_rng_twice(self):
        """Feature: Iterator reconstruction after RNG staging.

        Description: Recreate a loader whose iterator consumes Python, NumPy and Torch random values.
        Expectation: The first training draws match the committed sequence.
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
        expected = (random.random(), np.random.rand(), torch.rand(3))
        codec.restore(state)
        iterator = codec.make_iterator()
        codec.finish_restore()
        self.assertEqual(next(iterator), 8)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
