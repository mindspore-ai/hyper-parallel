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
"""Actual replica inference and direct tensor transfer with CPU-only simulated IO."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.fault_tolerance.config import RecoveryConfig
from hyper_parallel.core.fault_tolerance.groups import GroupRegistry
from hyper_parallel.core.fault_tolerance.state import ModelOptimizerState
from hyper_parallel.core.fault_tolerance.topology import (
    ReplicaTopology, device_replica_ranks, model_replica_candidates,
)
from hyper_parallel.core.fault_tolerance.transport import DeviceMetadataBank, ReplicaTransport, decode, encode
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS


class TestRecoveryConfig(unittest.TestCase):
    """Controller hosting never requires user-supplied replica pairs."""

    def test_simple_configuration(self):
        self.assertEqual(RecoveryConfig(True, "192.168.1.1").controller_port, 8000)
        self.assertIsNone(RecoveryConfig(True).controller_port)
        with self.assertRaisesRegex(ValueError, "controller_ip"):
            RecoveryConfig(True, controller_port=8000)


class TestReplicaTopology(unittest.TestCase):
    """TP/EP shard axes never become equivalent merely because ranks can pair up."""

    def test_dp_tp_ep_coordinates_only_vary_on_replicated_axes(self):
        replicate = SimpleNamespace(is_replicate=lambda: True)
        shard = SimpleNamespace(is_replicate=lambda: False)
        parameter = SimpleNamespace(device_mesh=SimpleNamespace(mesh_shape=(2, 2, 2), rank_list=tuple(range(8))),
                                    placements=(replicate, shard, shard))
        self.assertEqual(device_replica_ranks(parameter, 1), {1, 5})
        parameter.placements = (shard, shard, shard)
        self.assertEqual(device_replica_ranks(parameter, 1), {1})

    def test_replicated_layout_needs_no_sharding_wrapper(self):
        replicate = SimpleNamespace(is_replicate=lambda: True)
        parameter = SimpleNamespace(device_mesh=SimpleNamespace(mesh_shape=(4,), rank_list=(0, 1, 2, 3)),
                                    placements=(replicate,))
        state = SimpleNamespace(parameters={"weight": parameter})
        self.assertEqual(model_replica_candidates(state, 1, 4), (0, 1, 2, 3))

    def test_plain_parameters_do_not_gain_implicit_replica_peers(self):
        state = SimpleNamespace(parameters={"weight": torch.nn.Parameter(torch.zeros(2, 2))})
        self.assertEqual(model_replica_candidates(state, 1, 4), (1,))

    def test_only_symmetric_equivalent_layouts_can_register(self):
        profiles = [{"schema": str(rank % 4), "candidates": [rank % 4, rank % 4 + 4],
                     "optimizer_ownership": "all-local-state"} for rank in range(8)]
        topology = ReplicaTopology.from_profiles(profiles)
        self.assertEqual(topology.members(1), (1, 5))
        profiles[5]["schema"] = "different-optimizer-owner"
        with self.assertRaisesRegex(ValueError, "ownership"):
            ReplicaTopology.from_profiles(profiles)

    def test_unique_shards_are_rejected_instead_of_creating_cpu_backups(self):
        profiles = [{"schema": "same-size", "candidates": [rank]} for rank in range(8)]
        with self.assertRaisesRegex(ValueError, "no equivalent device replica"):
            ReplicaTopology.from_profiles(profiles)

    def test_muon_owners_split_model_replicas_into_complete_state_replicas(self):
        profiles = [{"schema": "same-model-and-optimizer", "candidates": [0, 1, 2, 3],
                     "optimizer_ownership": str(rank % 2)} for rank in range(4)]
        topology = ReplicaTopology.from_profiles(profiles)
        self.assertEqual(topology.replica_groups, ((0, 2), (1, 3)))
        profiles = [{"schema": "same-model-and-optimizer", "candidates": [0, 1],
                     "optimizer_ownership": str(rank)} for rank in range(2)]
        with self.assertRaisesRegex(ValueError, "no complete device replica.*optimizer state"):
            ReplicaTopology.from_profiles(profiles)


class TestMetadataGroupReuse(unittest.TestCase):
    """Reuse exact cached groups without changing training group ownership."""

    def setUp(self) -> None:
        """Isolate the shared group cache and mock HCCL handle creation."""
        cache = patch.dict(EXISTING_COMM_GROUPS, {}, clear=True)
        cache.start()
        self.addCleanup(cache.stop)
        patcher = patch("hyper_parallel.core.fault_tolerance.topology.dist")
        self.dist = patcher.start()
        self.addCleanup(patcher.stop)
        registry_dist = patch("hyper_parallel.core.fault_tolerance.groups.dist", self.dist)
        registry_dist.start()
        self.addCleanup(registry_dist.stop)
        self.dist.group.WORLD = object()
        self.dist.get_rank.return_value = 0
        self.dist.get_world_size.return_value = 8
        self.dist.get_backend.return_value = "hccl"
        self.dist.get_process_group_ranks.return_value = [0, 2, 4, 6]
        self.registry = GroupRegistry()
        self.topology = ReplicaTopology(((0, 2, 4, 6),))
        self.group = object()

    def test_exact_cached_members_are_reused_and_registered_once(self):
        """Feature: Reuse an exact HCCL group from the shared framework cache.

        Description: Resolve a cached group whose internal rank order differs from sorted replica order.
        Expectation: The same handle is reused and registered once without new group creation.
        """
        EXISTING_COMM_GROUPS["(0, 2, 4, 6)"] = self.group
        self.dist.get_process_group_ranks.return_value = [6, 4, 2, 0]
        for _ in range(2):
            self.assertIs(self.topology.get_metadata_group(0, self.registry), self.group)
        self.dist.new_group.assert_not_called()
        self.assertEqual(len(self.registry.groups), 1)
        self.assertIs(self.registry.groups[0].group, self.group)

    def test_full_world_replicas_reuse_the_default_group(self):
        """Feature: Default group reuse for a full-world replica domain.

        Description: Resolve full replicas without a cached default group entry.
        Expectation: The default handle is returned without a duplicate rebuild entry or new group.
        """
        topology = ReplicaTopology((tuple(range(8)),))
        self.dist.get_process_group_ranks.return_value = list(range(8))
        self.assertIs(topology.get_metadata_group(0, self.registry), self.dist.group.WORLD)
        self.dist.new_group.assert_not_called()
        self.assertEqual(self.registry.groups, [])

    def test_replacement_creates_only_its_missing_lazy_metadata_group(self):
        """Feature: Local reconstruction of a missing metadata group.

        Description: Bootstrap a replacement with no exact cached group and then rebuild its communication.
        Expectation: The hashed group-local handle is cached and registered without bootstrap collectives.
        """
        self.registry.reboot = True
        self.dist.new_group.return_value = self.group
        with patch.dict("os.environ", {"TORCH_DIST_INIT_BARRIER": "0"}):
            self.assertIs(self.topology.get_metadata_group(0, self.registry), self.group)
        self.dist.new_group.assert_called_once_with(ranks=[0, 2, 4, 6], backend="hccl",
                                                     use_local_synchronization=True, group_desc="recovery_metadata")
        self.dist.barrier.assert_not_called()
        self.dist.all_gather.assert_not_called()
        self.assertIs(EXISTING_COMM_GROUPS["(0, 2, 4, 6)"], self.group)
        self.registry.rebuild()
        self.assertIs(self.dist.reinit_process_group.call_args.kwargs["group"], self.group)

    def test_cached_backend_and_actual_members_must_match(self):
        """Feature: Validation of framework cached group handles.

        Description: Present a Gloo group or a group whose actual members differ from its cache key.
        Expectation: Recovery rejects the incompatible handle without silently reusing it.
        """
        EXISTING_COMM_GROUPS["(0, 2, 4, 6)"] = self.group
        for backend, ranks in (("gloo", [0, 2, 4, 6]), ("hccl", [0, 2])):
            with self.subTest(backend=backend, ranks=ranks):
                self.dist.get_backend.return_value = backend
                self.dist.get_process_group_ranks.return_value = ranks
                with self.assertRaisesRegex(ValueError, "cached metadata group"):
                    self.topology.get_metadata_group(0, self.registry)
        self.dist.new_group.assert_not_called()

    def test_bootstrap_rejects_initialization_barriers_or_eager_device_binding(self):
        """Feature: Replacement group construction avoids peer synchronization.

        Description: Enable an initialization barrier or bind the default group's device.
        Expectation: Recovery rejects both before attempting a potentially blocking group creation.
        """
        with patch.dict("os.environ", {"TORCH_DIST_INIT_BARRIER": "1"}):
            with self.assertRaisesRegex(ValueError, "TORCH_DIST_INIT_BARRIER"):
                self.topology.get_metadata_group(0, self.registry)
        self.dist.group.WORLD = SimpleNamespace(bound_device_id=torch.device("cpu"))
        with patch.dict("os.environ", {"TORCH_DIST_INIT_BARRIER": "0"}):
            with self.assertRaisesRegex(ValueError, "bound device_id"):
                self.topology.get_metadata_group(0, self.registry)
        self.dist.new_group.assert_not_called()


class TestDeviceMetadataBank(unittest.TestCase):
    """The persistent bank has rank metadata, never model or optimizer values."""

    def test_owner_metadata_and_bounded_versions(self):
        """Feature: Exact-step rank metadata with bounded retained versions.

        Description: Commit three snapshots with empty replica-provider mappings.
        Expectation: Owner metadata remains readable and the oldest version is unavailable.
        """
        bank = DeviceMetadataBank(1, torch.device("cpu"))
        for step in (7, 8, 9):
            bank.put(step, {"state": {"rank_state": {"rng_owner": 1, "cursor": step},
                                       "replica_state": {}}}, {})
        self.assertEqual(set(bank.get(1, 8)), {"owner", "step", "payload", "buffers"})
        metadata = decode(bank.get(1, 8)["payload"].numpy().tobytes())
        self.assertEqual(metadata["state"]["rank_state"]["rng_owner"], 1)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            bank.get(1, 7)

    def test_donor_providers_stay_local_while_target_rank_state_is_mirrored(self):
        """Feature: Separation of replica state from rank-owned mirrored state.

        Description: Save a mapping of scheduler/progress snapshots and install a peer's rank payload.
        Expectation: Shared state stays local until repair and merges with exact peer RNG/cursor/extra.
        """
        bank = DeviceMetadataBank(0, torch.device("cpu"))
        for step in (7, 8, 9):
            metadata = {"state": {"rank_state": {"loader": {"owner": 0, "position": step},
                                                  "rng": {"python": "rank0-rng"}},
                                  "replica_state": {"scheduler": {"step": step}, "progress": {"step": step}}},
                        "extra": {"owner": 0}}
            bank.put(step, metadata, {})
            metadata["state"]["replica_state"]["scheduler"]["step"] = -1
        local = decode(bank.get(0, 8)["payload"].numpy().tobytes())
        self.assertNotIn("replica_state", local["state"])
        self.assertEqual(local["state"]["rank_state"]["rng"]["python"], "rank0-rng")
        target = {"state": {"rank_state": {"loader": {"owner": 1, "position": 8},
                                           "rng": {"python": "rank1-rng"}}}, "extra": {"owner": 1}}
        payload = torch.frombuffer(bytearray(encode(target)), dtype=torch.uint8)
        bank.install(1, 8, payload, {})
        combined = bank.metadata(1, 8)["state"]
        self.assertEqual(combined["rank_state"]["rng"]["python"], "rank1-rng")
        self.assertEqual(combined["rank_state"]["loader"], {"owner": 1, "position": 8})
        self.assertEqual(combined["replica_state"], {"scheduler": {"step": 8}, "progress": {"step": 8}})
        self.assertEqual(bank.metadata(1, 8)["extra"], {"owner": 1})
        bank.install(1, 7, payload, {})
        with self.assertRaisesRegex(RuntimeError, "exact replica provider metadata unavailable"):
            bank.metadata(1, 7)


class TestDirectDeviceTransport(unittest.TestCase):
    """Transfer actual donor tensor storage and verify the next real optimizer update."""

    @staticmethod
    def _build(kind, dtype):
        model = torch.nn.Linear(4, 4, dtype=dtype)
        if kind == "adamw":
            leaves = {"adamw": AdamW(model.parameters())}
        else:
            leaves = {"muon": Muon([model.weight], ns_steps=2), "adamw": AdamW([model.bias])}
        chain = ChainedOptimizer(model, leaves)
        optimizer = Float16OptimizerWithFloat16Params(chain, model) if dtype == torch.bfloat16 else chain
        return model, optimizer, ModelOptimizerState(model, optimizer)

    @staticmethod
    def _step(model, optimizer):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()

    def test_mirror_all_gathers_only_complete_replicas_in_actual_group_order(self):
        """Feature: Metadata all-gather on an exact replica process group.

        Description: Gather unequal payloads and buffers in a group's unsorted rank order.
        Expectation: Peer owners are correct, local snapshots stay intact, and padding is allocated only when needed.
        """
        for peer_size in (2, 19):
            with self.subTest(peer_size=peer_size):
                bank = DeviceMetadataBank(0, torch.device("cpu"))
                original = torch.tensor([3, 4], dtype=torch.uint8)
                peer = torch.arange(peer_size, dtype=torch.uint8)
                local_buffer = torch.tensor([3.0])
                bank.install(0, 8, original, {"counter": local_buffer})
                inputs = []
                group = object()
                work = SimpleNamespace(wait=lambda **kwargs: None)

                def _gather(outputs, value, **kwargs):
                    self.assertIs(kwargs["group"], group)
                    inputs.append(value)
                    if len(inputs) == 1:
                        outputs[0].fill_(peer.numel())
                        outputs[1].copy_(value)
                    elif len(inputs) == 2:
                        outputs[0].zero_()
                        outputs[0][:peer.numel()].copy_(peer)
                        outputs[1].copy_(value)
                    else:
                        outputs[0].fill_(55)
                        outputs[1].copy_(value)
                    return work

                with patch("hyper_parallel.core.fault_tolerance.transport.dist") as distributed:
                    distributed.get_process_group_ranks.return_value = [4, 0]
                    distributed.all_gather.side_effect = _gather
                    distributed.barrier.return_value = work
                    channel = ReplicaTransport(group, bank, object(), torch.device("cpu"))
                    channel.mirror_metadata(8)
                    distributed.new_group.assert_not_called()
                    self.assertEqual(distributed.all_gather.call_count, 3)
                    distributed.barrier.assert_called_once_with(group=group, async_op=True)
                    distributed.batch_isend_irecv.assert_not_called()
                self.assertEqual(inputs[1].data_ptr() == original.data_ptr(), peer_size == 2)
                self.assertEqual(bank.get(0, 8)["payload"].data_ptr(), original.data_ptr())
                self.assertIs(bank.get(0, 8)["buffers"]["counter"], local_buffer)
                torch.testing.assert_close(bank.get(0, 8)["payload"], original, rtol=0, atol=0)
                torch.testing.assert_close(bank.get(4, 8)["payload"], peer, rtol=0, atol=0)
                torch.testing.assert_close(bank.get(4, 8)["buffers"]["counter"], torch.tensor([55.0]))

    def test_repair_combines_target_rng_with_exact_donor_scheduler(self):
        """Feature: Replica-provider reconstruction during device repair.

        Description: Transfer a target rank's RNG and the donor's committed scheduler snapshot.
        Expectation: Received state combines both sources without mirroring the replica provider.
        """
        _, _, state = self._build("adamw", torch.float32)
        _, _, target_state = self._build("adamw", torch.float32)
        bank, target_bank = (DeviceMetadataBank(rank, torch.device("cpu")) for rank in (0, 1))
        bank.put(8, {"state": {"rank_state": {"rng": {"python": "rank0-rng"}},
                               "replica_state": {"scheduler": {"step": 8}}},
                     "extra": {}}, {})
        target = {"state": {"rank_state": {"rng": {"python": "rank1-rng"}}}, "extra": {}}
        bank.install(1, 8, torch.frombuffer(bytearray(encode(target)), dtype=torch.uint8), {})
        messages = []
        with patch("hyper_parallel.core.fault_tolerance.transport.dist") as distributed:
            distributed.get_process_group_ranks.return_value = [0, 1]
            distributed.send.side_effect = lambda tensor, **kwargs: messages.append(tensor.clone())
            distributed.recv.side_effect = lambda tensor, **kwargs: tensor.copy_(messages.pop(0))
            sender = ReplicaTransport(object(), bank, state, torch.device("cpu"))
            receiver = ReplicaTransport(object(), target_bank, target_state, torch.device("cpu"))
            sender.send(1, 8)
            receiver.receive(0, 8)
        self.assertFalse(messages)
        self.assertEqual(target_bank.metadata(1, 8)["state"],
                         {"rank_state": {"rng": {"python": "rank1-rng"}},
                          "replica_state": {"scheduler": {"step": 8}}})
        self.assertNotIn("replica_state", decode(target_bank.get(1, 8)["payload"].numpy().tobytes())["state"])

    def test_device_send_restores_adamw_muon_and_bf16_main_state(self):
        """Feature: Direct device-state repair with an empty replica-provider mapping.

        Description: Transfer AdamW/Muon and BF16 main state, then execute another optimizer step.
        Expectation: Target tensors match the donor before and after the next real update.
        """
        for kind, dtype in (("adamw", torch.float32), ("muon_adamw", torch.float32),
                            ("muon_adamw", torch.bfloat16)):
            with self.subTest(kind=kind, dtype=dtype):
                model, optimizer, state = self._build(kind, dtype)
                self._step(model, optimizer)
                target, target_optimizer, target_state = self._build(kind, dtype)
                bank = DeviceMetadataBank(0, torch.device("cpu"))
                target_bank = DeviceMetadataBank(1, torch.device("cpu"))
                bank.put(8, {"state": {"rank_state": {"rng": {"python": "rank0-rng"}},
                                       "replica_state": {}}, "extra": {}}, {})
                target_metadata = {"state": {"rank_state": {"rng": {"python": "rank1-rng"}}}, "extra": {}}
                payload = torch.frombuffer(bytearray(encode(target_metadata)),
                                           dtype=torch.uint8)
                bank.install(1, 8, payload, {})
                messages, pointers = [], []

                def _send(tensor, **kwargs):
                    del kwargs
                    pointers.append(tensor.data_ptr())
                    messages.append(tensor.clone())

                with patch("hyper_parallel.core.fault_tolerance.transport.dist") as distributed:
                    distributed.get_process_group_ranks.return_value = [0, 1]
                    distributed.send.side_effect = _send
                    distributed.recv.side_effect = lambda tensor, **kwargs: tensor.copy_(messages.pop(0))
                    sender = ReplicaTransport(object(), bank, state, torch.device("cpu"))
                    receiver = ReplicaTransport(object(), target_bank, target_state, torch.device("cpu"))
                    sender.send(1, 8)
                    receiver.receive(0, 8)
                self.assertIn(model.weight.data_ptr(), pointers)
                self.assertFalse(messages)
                for (_, actual), (_, expected) in zip(target_state.tensor_items(), state.tensor_items()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                target_state.restore_rank_state({}, {})
                self._step(model, optimizer)
                self._step(target, target_optimizer)
                for (_, actual), (_, expected) in zip(target_state.tensor_items(), state.tensor_items()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                metadata = decode(target_bank.get(1, 8)["payload"].numpy().tobytes())
                self.assertEqual(metadata["state"], {"rank_state": {"rng": {"python": "rank1-rng"}}})
                self.assertEqual(target_bank.metadata(1, 8)["state"]["replica_state"], {})
