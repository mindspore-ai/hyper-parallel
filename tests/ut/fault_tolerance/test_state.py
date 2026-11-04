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
from unittest.mock import patch

import numpy as np
import torch

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
        mesh = SimpleNamespace(mesh_shape=(domain_size,), mesh_dim_names=("replicate",))
        parameters = list(model.parameters())
        for parameter in parameters:
            parameter.device_mesh = mesh
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


class TestTrainState(unittest.TestCase):
    """RNG and cursor restore must use the logical rank's exact state."""

    def test_rng_cursor_and_scheduler_restore(self):
        loader, scheduler = StatefulValue(8), StatefulValue(80)
        codec = TrainState(loader, {"scheduler": scheduler})
        saved = codec.capture()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        loader.value, scheduler.value = 9, 90
        codec.restore(saved)
        self.assertEqual(loader.value, 8)
        self.assertEqual(scheduler.value, 80)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_unrestorable_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "state_dict"):
            TrainState(iter([1, 2]), {})

    def test_only_registered_providers_can_use_donor_state(self):
        with self.assertRaisesRegex(ValueError, "registered training state providers"):
            TrainState(StatefulValue(), {}, replica_providers=("scheduler",))
        state = TrainState(StatefulValue(8), {"scheduler": StatefulValue(80), "custom": StatefulValue(1)},
                           replica_providers=("scheduler",))
        self.assertEqual(state.replica_providers, frozenset({"scheduler"}))
        self.assertEqual(state.capture()["providers"], {"scheduler": {"value": 80}, "custom": {"value": 1}})

    def test_custom_generator_and_refused_cursor(self):
        generator = torch.Generator().manual_seed(123)
        loader = StatefulValue(8)
        codec = TrainState(loader, {}, generators={"dropout": generator})
        saved = codec.capture()
        expected = torch.rand(4, generator=generator)
        codec.restore(saved)
        torch.testing.assert_close(torch.rand(4, generator=generator), expected, rtol=0, atol=0)
        loader.load_state_dict = lambda state: False
        with self.assertRaisesRegex(RuntimeError, "refused"):
            codec.restore(saved)

    def test_reentry_iterator_does_not_consume_global_rng_twice(self):
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
        iterator = codec.make_iterator(preserve_rng=True)
        self.assertEqual(next(iterator), 8)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
