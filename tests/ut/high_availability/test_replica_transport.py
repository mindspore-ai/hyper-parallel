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
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel import DTensor, SkipDTensorDispatch
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _DEVICE_MESH_MAP
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, FSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.high_availability.config import WorkerRecoveryConfig
from hyper_parallel.high_availability.group_registry import GroupDescriptor, GroupRegistry
from hyper_parallel.high_availability.recovery_state import ModelOptimizerState, clone_metadata
from hyper_parallel.high_availability.replica_topology import (
    ReplicaTopology, device_replica_ranks, model_replica_candidates,
)
from hyper_parallel.high_availability.replica_transport import (
    DeviceMetadataBank,
    deserialize_metadata,
    ReplicaTransport,
    serialize_metadata,
)
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS
from hyper_parallel.models._transformers.model_builder import _materialize_and_load_model


class TestRecoveryConfig(unittest.TestCase):
    """Controller hosting never requires user-supplied replica pairs."""

    def test_simple_configuration(self):
        self.assertEqual(WorkerRecoveryConfig(True, "192.168.1.1").controller_port, 8000)
        self.assertIsNone(WorkerRecoveryConfig(True).controller_port)
        with self.assertRaisesRegex(ValueError, "controller_ip"):
            WorkerRecoveryConfig(True, controller_port=8000)


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
        patcher = patch("hyper_parallel.high_availability.replica_topology.dist")
        self.dist = patcher.start()
        self.addCleanup(patcher.stop)
        registry_dist = patch("hyper_parallel.high_availability.group_registry.dist", self.dist)
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

    def test_exact_registered_members_are_reused_and_registered_once(self):
        """Feature: Reuse an exact HCCL group from the recovery catalog.

        Description: Resolve a registered group whose internal rank order differs from sorted replica order.
        Expectation: The same handle is reused and registered once without new group creation.
        """
        self.registry.groups.append(GroupDescriptor((6, 4, 2, 0), "hccl", "training", self.group))
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
        Expectation: The hashed group-local handle is registered without global caching or bootstrap collectives.
        """
        self.registry.reboot = True
        self.dist.new_group.return_value = self.group
        with patch.dict("os.environ", {"TORCH_DIST_INIT_BARRIER": "0"}):
            self.assertIs(self.topology.get_metadata_group(0, self.registry), self.group)
        self.dist.new_group.assert_called_once_with(ranks=[0, 2, 4, 6], backend="hccl",
                                                     use_local_synchronization=True, group_desc="recovery_metadata")
        self.dist.barrier.assert_not_called()
        self.dist.all_gather.assert_not_called()
        self.assertNotIn("(0, 2, 4, 6)", EXISTING_COMM_GROUPS)
        self.assertIs(self.topology.get_metadata_group(0, self.registry), self.group)
        self.assertEqual(self.dist.new_group.call_count, 1)
        self.registry.rebuild()
        self.assertIs(self.dist.reinit_process_group.call_args.kwargs["group"], self.group)

    def test_first_exact_registered_group_wins_over_supersets_and_global_cache(self):
        """Feature: Stable metadata group selection from the recovery catalog.

        Description: Register a superset and two exact groups, while the global cache contains another handle.
        Expectation: The earliest exact registered handle is reused without changing the global cache or catalog.
        """
        superset, second, cached = object(), object(), object()
        self.registry.groups.extend([
            GroupDescriptor(tuple(range(8)), "hccl", "larger", superset),
            GroupDescriptor((0, 2, 4, 6), "hccl", "first", self.group),
            GroupDescriptor((0, 2, 4, 6), "hccl", "second", second),
        ])
        EXISTING_COMM_GROUPS["(0, 2, 4, 6)"] = cached
        self.assertIs(self.topology.get_metadata_group(0, self.registry), self.group)
        self.assertEqual(len(self.registry.groups), 3)
        self.assertIs(EXISTING_COMM_GROUPS["(0, 2, 4, 6)"], cached)
        self.dist.new_group.assert_not_called()

    def test_superset_is_not_reused_for_replica_collectives(self):
        """Feature: Metadata collectives use exactly the equivalent replica members.

        Description: Resolve the replica group when the catalog contains only a larger group.
        Expectation: A new exact group is created and registered instead of involving unrelated ranks.
        """
        larger = object()
        self.registry.groups.append(GroupDescriptor(tuple(range(8)), "hccl", "larger", larger))
        self.dist.new_group.return_value = self.group
        with patch.dict("os.environ", {"TORCH_DIST_INIT_BARRIER": "0"}):
            self.assertIs(self.topology.get_metadata_group(0, self.registry), self.group)
        self.dist.new_group.assert_called_once_with(ranks=[0, 2, 4, 6], backend="hccl",
                                                  use_local_synchronization=True, group_desc="recovery_metadata")
        self.assertEqual([descriptor.ranks for descriptor in self.registry.groups],
                         [tuple(range(8)), (0, 2, 4, 6)])

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

    def test_owned_snapshots_are_not_copied_again(self):
        """Keep owned CPU snapshots even when the worker uses another device."""
        bank = DeviceMetadataBank(0, torch.device("meta"))
        buffer = torch.tensor([8.0])
        snapshot = {"state": {"rank_state": {"rng": torch.tensor([8])},
                          "replica_state": {"scheduler": {"step": 8}}}, "extra": {}}
        snapshot_metadata = {**snapshot, "buffers": {"counter": buffer}}
        training_state = snapshot_metadata["state"]
        rank_metadata = snapshot_metadata.copy()
        rank_metadata.pop("state")
        rank_metadata["rank_state"] = training_state["rank_state"]
        bank.put_snapshot(8, rank_metadata, training_state["replica_state"])
        self.assertIs(bank._replica_snapshot["state"], snapshot["state"]["replica_state"])
        self.assertIs(bank.get_rank_metadata(0, 8)["rank_state"],
                      snapshot["state"]["rank_state"])
        recovered = bank.get_rank_metadata(0, 8)
        replica_state = bank.get_replica_state(8)
        self.assertIs(recovered["buffers"]["counter"], buffer)
        self.assertEqual(buffer.device, torch.device("cpu"))
        self.assertIs(replica_state, snapshot["state"]["replica_state"])
        self.assertIs(recovered["rank_state"], snapshot["state"]["rank_state"])
        self.assertIs(recovered, rank_metadata)
        self.assertNotIn("replica_state", bank.get_rank_metadata(0, 8)["rank_state"])
        self.assertEqual(recovered["rank_state"]["rng"].item(), 8)
        self.assertEqual(replica_state["scheduler"]["step"], 8)

    def test_local_snapshots_skip_encoding_and_replace_previous_snapshot(self):
        """Keep only the latest independent CPU snapshot and buffers without encoding or transfers."""
        bank = DeviceMetadataBank(0, torch.device("cpu"))
        buffer = torch.tensor([8.0])
        with patch("hyper_parallel.high_availability.replica_transport.metadata_to_device") as to_device, \
                patch("hyper_parallel.high_availability.replica_transport.serialize_metadata") as encode, \
                patch("hyper_parallel.high_availability.replica_transport.deserialize_metadata") as decode:
            for step in (7, 8, 9):
                metadata = {"state": {"rank_state": {"rng": torch.tensor([step]), "loader": {"cursor": step}},
                                      "replica_state": {"scheduler": {"step": step}}},
                            "extra": {"counter": step}}
                snapshot_metadata = {**clone_metadata(metadata), "buffers": {"counter": buffer}}
                training_state = snapshot_metadata["state"]
                rank_metadata = snapshot_metadata.copy()
                rank_metadata.pop("state")
                rank_metadata["rank_state"] = training_state["rank_state"]
                bank.put_snapshot(step, rank_metadata, training_state["replica_state"])
                metadata["state"]["rank_state"]["rng"].zero_()
                metadata["state"]["rank_state"]["loader"]["cursor"] = -1
                metadata["state"]["replica_state"]["scheduler"]["step"] = -1
                metadata["extra"]["counter"] = -1
            saved = bank.get_rank_metadata(0, 9)
            replica_state = bank.get_replica_state(9)
            self.assertEqual(saved["rank_state"]["rng"].item(), 9)
            self.assertEqual(saved["rank_state"]["loader"]["cursor"], 9)
            self.assertEqual(replica_state["scheduler"]["step"], 9)
            self.assertEqual(saved["extra"]["counter"], 9)
            self.assertIs(saved["rank_state"],
                          bank.get_rank_metadata(0, 9)["rank_state"])
            self.assertNotIn("replica_state", bank.get_rank_metadata(0, 9)["rank_state"])
            self.assertIs(bank.get_rank_metadata(0, 9)["buffers"]["counter"], buffer)
            self.assertIsInstance(bank.get_rank_metadata(0, 9), dict)
            for step in (7, 8):
                with self.assertRaisesRegex(RuntimeError, "exact training metadata unavailable"):
                    bank.get_rank_metadata(0, step)
            to_device.assert_not_called()
            encode.assert_not_called()
            decode.assert_not_called()

    def test_owner_metadata_retains_only_latest_snapshot(self):
        """Feature: Exact-step rank metadata with one retained snapshot.

        Description: Commit three snapshots with empty replica-provider mappings.
        Expectation: Only the latest owner metadata remains readable; earlier steps are unavailable.
        """
        bank = DeviceMetadataBank(1, torch.device("cpu"))
        for step in (7, 8, 9):
            snapshot_metadata = {
                "state": {"rank_state": {"rng_owner": 1, "cursor": step}, "replica_state": {}},
                "extra": {},
                "buffers": {},
            }
            training_state = snapshot_metadata["state"]
            rank_metadata = snapshot_metadata.copy()
            rank_metadata.pop("state")
            rank_metadata["rank_state"] = training_state["rank_state"]
            bank.put_snapshot(step, rank_metadata, training_state["replica_state"])
        self.assertEqual(set(bank.get_rank_metadata(1, 9)), {"rank_state", "extra", "buffers"})
        metadata = bank.get_rank_metadata(1, 9)
        self.assertEqual(metadata["rank_state"]["rng_owner"], 1)
        for step in (7, 8):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                bank.get_rank_metadata(1, step)

    def test_donor_providers_stay_local_while_target_rank_state_is_mirrored(self):
        """Feature: Separation of replica state from rank-owned mirrored state.

        Description: Save a mapping of scheduler/progress snapshots and install a peer's rank encoded_metadata.
        Expectation: Shared state stays local until repair and merges with exact peer RNG/cursor/extra.
        """
        bank = DeviceMetadataBank(0, torch.device("cpu"))
        for step in (7, 8, 9):
            metadata = {"state": {"rank_state": {"loader": {"owner": 0, "position": step},
                                                  "rng": {"python": "rank0-rng"}},
                                  "replica_state": {"scheduler": {"step": step}, "progress": {"step": step}}},
                        "extra": {"owner": 0}}
            snapshot_metadata = {**clone_metadata(metadata), "buffers": {}}
            training_state = snapshot_metadata["state"]
            rank_metadata = snapshot_metadata.copy()
            rank_metadata.pop("state")
            rank_metadata["rank_state"] = training_state["rank_state"]
            bank.put_snapshot(step, rank_metadata, training_state["replica_state"])
            metadata["state"]["replica_state"]["scheduler"]["step"] = -1
        local = bank.get_rank_metadata(0, 9)
        self.assertNotIn("replica_state", local["rank_state"])
        self.assertEqual(local["rank_state"]["rng"]["python"], "rank0-rng")
        target = {"rank_state": {"loader": {"owner": 1, "position": 9},
                                           "rng": {"python": "rank1-rng"}}, "extra": {"owner": 1}}
        bank.store_rank_snapshot(1, 9, {**target, "buffers": {}})
        rank_metadata = bank.get_rank_metadata(1, 9)
        replica_state = bank.get_replica_state(9)
        self.assertEqual(rank_metadata["rank_state"]["rng"]["python"], "rank1-rng")
        self.assertEqual(rank_metadata["rank_state"]["loader"], {"owner": 1, "position": 9})
        self.assertEqual(replica_state, {"scheduler": {"step": 9}, "progress": {"step": 9}})
        self.assertEqual(bank.get_rank_metadata(1, 9)["extra"], {"owner": 1})
        bank.store_rank_snapshot(1, 7, {**target, "buffers": {}})
        with self.assertRaisesRegex(RuntimeError, "exact training metadata unavailable"):
            bank.get_rank_metadata(1, 9)
        with self.assertRaisesRegex(RuntimeError, "exact replica provider metadata unavailable"):
            bank.get_replica_state(7)


class TestDirectDeviceTransport(unittest.TestCase):
    """Exercise tensor staging and full state transfers without accelerator hardware."""

    def test_host_tensor_is_staged_before_chunked_send(self):
        """Feature: Host state is staged on the transport device before sending.

        Description: Send a noncontiguous CPU tensor using simulated accelerator staging and small chunks.
        Expectation: Preserve source storage and values; send all staged elements in order, including the tail.
        """
        transport = ReplicaTransport.__new__(ReplicaTransport)
        transport.device = torch.device("meta")
        source = torch.arange(15, dtype=torch.float32).reshape(3, 5).t()
        original = source.clone()
        pointer = source.data_ptr()
        staged_sizes = []
        sent = []

        def stage(block, *, device):
            self.assertEqual(device, transport.device)
            staged_sizes.append(block.numel())
            return block.clone()

        with patch.object(torch.Tensor, "to", stage), \
                patch("hyper_parallel.high_availability.replica_transport._TRANSFER_CHUNK_BYTES", 16), \
                patch("hyper_parallel.high_availability.replica_transport.dist.send",
                      side_effect=lambda tensor, **kwargs: sent.append(tensor.clone())) as send:
            transport._send_tensor(source, 1, "repair-group")
        self.assertEqual(staged_sizes, [4, 4, 4, 3])
        self.assertEqual([tensor.numel() for tensor in sent], [4, 4, 4, 3])
        torch.testing.assert_close(torch.cat(sent), original.contiguous().view(-1))
        self.assertEqual(source.data_ptr(), pointer)
        self.assertEqual(source.device.type, "cpu")
        torch.testing.assert_close(source, original)
        for call in send.call_args_list:
            self.assertEqual(call.kwargs, {"dst": 1, "group": "repair-group"})

    def test_same_device_send_reuses_contiguous_source_storage(self):
        """Feature: Existing communication-device tensors do not require another full copy.

        Description: Send a contiguous tensor already on the simulated communication device.
        Expectation: The transmitted view shares source storage and preserves values.
        """
        transport = ReplicaTransport.__new__(ReplicaTransport)
        transport.device = torch.device("cpu")
        source = torch.arange(8, dtype=torch.float32)
        with patch("hyper_parallel.high_availability.replica_transport.dist.send") as send:
            transport._send_tensor(source, 1)
        send.assert_called_once()
        transmitted = send.call_args.args[0]
        self.assertEqual(transmitted.data_ptr(), source.data_ptr())
        torch.testing.assert_close(transmitted, source)

    @staticmethod
    def _build(kind, dtype):
        """Build the model and optimizer with initialized state for replica transfer."""
        model = torch.nn.Linear(4, 4, dtype=dtype)
        if kind == "adamw":
            leaves = {"adamw": AdamW(model.parameters())}
        else:
            leaves = {"muon": Muon([model.weight], ns_steps=2), "adamw": AdamW([model.bias])}
        chain = ChainedOptimizer(model, leaves)
        optimizer = (Float16OptimizerWithFloat16Params(chain, model)
                     if dtype in (torch.float16, torch.bfloat16) else chain)
        return model, optimizer, ModelOptimizerState(model, optimizer)

    @staticmethod
    def _step(model, optimizer):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()

    def test_mirror_all_gathers_only_complete_replicas_in_actual_group_order(self):
        """Mirror complete rank metadata, transmitting buffers outside the encoded fields."""
        for padding in (0, 10000):
            with self.subTest(padding=padding):
                bank = DeviceMetadataBank(0, torch.device("cpu"))
                metadata = {"rank_state": {}, "extra": {"padding": "x" * 100}}
                peer_metadata = {"rank_state": {}, "extra": {"padding": "x" * padding}}
                peer = torch.frombuffer(bytearray(serialize_metadata(peer_metadata)), dtype=torch.uint8)
                local_buffer = torch.tensor([3.0])
                rank_metadata = {**metadata, "buffers": {"counter": local_buffer}}
                bank.store_rank_snapshot(0, 8, rank_metadata)
                inputs = []
                group = object()
                work = SimpleNamespace(wait=lambda **kwargs: None)

                def _gather(outputs, value, context=(group, inputs, peer, work), **kwargs):
                    expected_group, inputs, peer, work = context
                    self.assertIs(kwargs["group"], expected_group)
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

                with patch("hyper_parallel.high_availability.replica_transport.dist") as distributed:
                    distributed.get_process_group_ranks.return_value = [4, 0]
                    distributed.all_gather.side_effect = _gather
                    distributed.barrier.return_value = work
                    channel = ReplicaTransport(group, bank, object(), torch.device("cpu"))
                    channel.mirror_metadata(8)
                    self.assertEqual(distributed.all_gather.call_count, 3)
                    distributed.barrier.assert_called_once_with(group=group, async_op=True)
                local_length = int(inputs[0].item())
                encoded_fields = deserialize_metadata(inputs[1][:local_length])
                self.assertEqual(encoded_fields, metadata)
                self.assertNotIn("buffers", encoded_fields)
                self.assertIs(bank.get_rank_metadata(0, 8), rank_metadata)
                received = bank.get_rank_metadata(4, 8)
                self.assertEqual(received["rank_state"], peer_metadata["rank_state"])
                self.assertEqual(received["extra"], peer_metadata["extra"])
                torch.testing.assert_close(received["buffers"]["counter"], torch.tensor([55.0]))

    def test_repair_reuses_donor_metadata_when_mirroring_is_disabled(self):
        """Skip normal-step communication and repair from the donor's exact local snapshot."""
        _, _, state = self._build("adamw", torch.float32)
        _, _, target_state = self._build("adamw", torch.float32)
        bank, target_bank = (DeviceMetadataBank(rank, torch.device("cpu")) for rank in (0, 1))
        metadata = {"state": {"rank_state": {"rng": {"python": "shared-rng"}},
                              "replica_state": {"scheduler": {"step": 8}}}, "extra": {}}
        snapshot_metadata = {**metadata, "buffers": {}}
        training_state = snapshot_metadata["state"]
        rank_metadata = snapshot_metadata.copy()
        rank_metadata.pop("state")
        rank_metadata["rank_state"] = training_state["rank_state"]
        bank.put_snapshot(8, rank_metadata, training_state["replica_state"])
        messages = []
        with patch("hyper_parallel.high_availability.replica_transport.dist") as distributed:
            distributed.get_process_group_ranks.return_value = [0, 1]
            distributed.send.side_effect = lambda tensor, **kwargs: messages.append(tensor.clone())
            distributed.recv.side_effect = lambda tensor, **kwargs: tensor.copy_(messages.pop(0))
            sender = ReplicaTransport(object(), bank, state, torch.device("cpu"), mirror_metadata=False)
            receiver = ReplicaTransport(object(), target_bank, target_state, torch.device("cpu"),
                                        mirror_metadata=False)
            sender.mirror_metadata(8)
            distributed.all_gather.assert_not_called()
            distributed.send.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "exact training metadata unavailable"):
                sender.send(1, 9)
            distributed.send.assert_not_called()
            sender.send(1, 8)
            receiver.receive(0, 8)
        self.assertFalse(messages)
        self.assertEqual(target_bank.get_rank_metadata(1, 8), rank_metadata)
        self.assertEqual(target_bank.get_replica_state(8), training_state["replica_state"])
        self.assertIsInstance(target_bank.get_rank_metadata(1, 8), dict)
        with self.assertRaisesRegex(RuntimeError, "exact training metadata unavailable"):
            bank.get_rank_metadata(1, 8)

    def test_repair_receives_cpu_buffer_snapshots_without_mutating_model(self):
        """Keep received buffers as CPU snapshots until the unified rollback stage."""
        bank, target_bank = (DeviceMetadataBank(rank, torch.device("cpu")) for rank in (0, 1))
        expected = torch.tensor([8.0, 9.0])
        metadata = {"rank_state": {}, "extra": {}, "buffers": {"counter": expected}}
        bank.put_snapshot(8, metadata, {})
        destination = torch.zeros_like(expected)
        pointer = destination.data_ptr()
        sender_state = MagicMock()
        sender_state.metadata.return_value = {}
        sender_state.tensor_items.return_value = []
        sender_state.buffers = {"counter": torch.zeros_like(expected)}
        receiver_state = MagicMock()
        receiver_state.metadata.return_value = {}
        receiver_state.tensor_items.return_value = []
        receiver_state.buffers = {"counter": SimpleNamespace(_local_tensor=destination)}
        messages = []
        with patch("hyper_parallel.high_availability.replica_transport.dist") as distributed:
            distributed.get_process_group_ranks.return_value = [0, 1]
            distributed.send.side_effect = lambda tensor, **kwargs: messages.append(tensor.clone())
            distributed.recv.side_effect = lambda tensor, **kwargs: tensor.copy_(messages.pop(0))
            sender = ReplicaTransport(object(), bank, sender_state, torch.device("cpu"), mirror_metadata=False)
            receiver = ReplicaTransport(object(), target_bank, receiver_state, torch.device("cpu"),
                                        mirror_metadata=False)
            receiver_tensor = receiver._receive_tensor
            with patch.object(receiver, "_receive_tensor", wraps=receiver_tensor) as receive_tensor:
                sender.send(1, 8)
                receiver.receive(0, 8)
                self.assertNotEqual(receive_tensor.call_args_list[-1].args[0].data_ptr(), pointer)
            torch.testing.assert_close(sender_state.buffers["counter"], torch.zeros_like(expected))
            receiver.send(0, 8)
            sender.receive(1, 8)
        self.assertFalse(messages)
        torch.testing.assert_close(destination, torch.zeros_like(expected))
        torch.testing.assert_close(sender_state.buffers["counter"], torch.zeros_like(expected))
        saved = target_bank.get_rank_metadata(1, 8)["buffers"]["counter"]
        self.assertEqual(saved.device, torch.device("cpu"))
        self.assertNotEqual(saved.data_ptr(), pointer)
        torch.testing.assert_close(saved, expected)
        torch.testing.assert_close(bank.get_rank_metadata(0, 8)["buffers"]["counter"], expected)

    def test_repair_combines_target_rng_with_exact_donor_scheduler(self):
        """Feature: Replica-provider reconstruction during device repair.

        Description: Transfer a target rank's RNG and the donor's committed scheduler snapshot.
        Expectation: Received state combines both sources without mirroring the replica provider.
        """
        _, _, state = self._build("adamw", torch.float32)
        _, _, target_state = self._build("adamw", torch.float32)
        bank, target_bank = (DeviceMetadataBank(rank, torch.device("cpu")) for rank in (0, 1))
        snapshot_metadata = {
            "state": {"rank_state": {"rng": {"python": "rank0-rng"}},
                      "replica_state": {"scheduler": {"step": 8}}},
            "extra": {},
            "buffers": {},
        }
        training_state = snapshot_metadata["state"]
        rank_metadata = snapshot_metadata.copy()
        rank_metadata.pop("state")
        rank_metadata["rank_state"] = training_state["rank_state"]
        bank.put_snapshot(8, rank_metadata, training_state["replica_state"])
        target = {"rank_state": {"rng": {"python": "rank1-rng"}}, "extra": {}}
        bank.store_rank_snapshot(1, 8, {**target, "buffers": {}})
        messages = []
        with patch("hyper_parallel.high_availability.replica_transport.dist") as distributed:
            distributed.get_process_group_ranks.return_value = [0, 1]
            distributed.send.side_effect = lambda tensor, **kwargs: messages.append(tensor.clone())
            distributed.recv.side_effect = lambda tensor, **kwargs: tensor.copy_(messages.pop(0))
            sender = ReplicaTransport(object(), bank, state, torch.device("cpu"))
            receiver = ReplicaTransport(object(), target_bank, target_state, torch.device("cpu"))
            sender.send(1, 8)
            receiver.receive(0, 8)
        self.assertFalse(messages)
        self.assertEqual(target_bank.get_rank_metadata(1, 8)["rank_state"],
                         {"rng": {"python": "rank1-rng"}})
        self.assertEqual(target_bank.get_replica_state(8), {"scheduler": {"step": 8}})
        target_metadata = target_bank.get_rank_metadata(1, 8)
        self.assertNotIn("replica_state", target_metadata["rank_state"])

    def test_device_send_restores_adamw_muon_and_low_precision_main_state(self):
        """Feature: Direct device-state repair with an empty replica-provider mapping.

        Description: Transfer AdamW/Muon with FP16/BF16 main state, then execute another optimizer step.
        Expectation: Target tensors match the donor before and after the next real update.
        """
        for kind, dtype in (
            ("adamw", torch.float32),
            ("muon_adamw", torch.float32),
            ("muon_adamw", torch.bfloat16),
            ("adamw", torch.float16),
            ("muon_adamw", torch.float16),
        ):
            with self.subTest(kind=kind, dtype=dtype):
                model, optimizer, state = self._build(kind, dtype)
                self._step(model, optimizer)
                target, target_optimizer, target_state = self._build(kind, dtype)
                bank = DeviceMetadataBank(0, torch.device("cpu"))
                target_bank = DeviceMetadataBank(1, torch.device("cpu"))
                snapshot_metadata = {
                    "state": {"rank_state": {"rng": {"python": "rank0-rng"}}, "replica_state": {}},
                    "extra": {},
                    "buffers": {},
                }
                training_state = snapshot_metadata["state"]
                rank_metadata = snapshot_metadata.copy()
                rank_metadata.pop("state")
                rank_metadata["rank_state"] = training_state["rank_state"]
                bank.put_snapshot(8, rank_metadata, training_state["replica_state"])
                target_metadata = {"rank_state": {"rng": {"python": "rank1-rng"}}, "extra": {}}
                bank.store_rank_snapshot(1, 8, {**target_metadata, "buffers": {}})
                messages, pointers = [], []

                def _send(tensor, context=(messages, pointers), **kwargs):
                    messages, pointers = context
                    del kwargs
                    pointers.append(tensor.data_ptr())
                    messages.append(tensor.clone())

                with patch("hyper_parallel.high_availability.replica_transport.dist") as distributed:
                    distributed.get_process_group_ranks.return_value = [0, 1]
                    distributed.send.side_effect = _send
                    distributed.recv.side_effect = (
                        lambda tensor, context=(messages,), **kwargs: tensor.copy_(context[0].pop(0))
                    )
                    sender = ReplicaTransport(object(), bank, state, torch.device("cpu"))
                    receiver = ReplicaTransport(object(), target_bank, target_state, torch.device("cpu"))
                    sender.send(1, 8)
                    receiver.receive(0, 8)
                self.assertIn(model.weight.data_ptr(), pointers)
                self.assertFalse(messages)
                for (_, actual), (_, expected) in zip(target_state.tensor_items(), state.tensor_items()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                target_state.restore_model_state({}, {})
                self._step(model, optimizer)
                self._step(target, target_optimizer)
                for (_, actual), (_, expected) in zip(target_state.tensor_items(), state.tensor_items()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                metadata = target_bank.get_rank_metadata(1, 8)
                self.assertEqual(metadata["rank_state"], {"rng": {"python": "rank1-rng"}})
                self.assertEqual(target_bank.get_replica_state(8), {})


class TestCPUOffloadRepair(unittest.TestCase):
    """Exercise real host-shard materialization and alias-preserving repair with simulated transport."""

    def tearDown(self) -> None:
        """Clear the local mesh cache after each native shard test."""
        _DEVICE_MESH_MAP.clear()

    @staticmethod
    def _build(reboot=False):
        """Build the model and optimizer with initialized state for replica transfer."""
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))
        info = object.__new__(FSDPMeshInfo)
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0):
            info.mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",), _init_backend=False)
        info.shard_mesh_dim = 0
        info.replicate_mesh_dim = None
        info.shard_mesh_rank = 0
        info.shard_mesh_size = 1
        info.shard_process_group = None
        param = HSDPParamV2(
            model.weight,
            ParamModuleInfo(model, "weight", [], []),
            info,
            mp_policy=MixedPrecisionPolicy(),
            offload_policy=CPUOffloadPolicy(pin_memory=False),
            device=torch.device("cpu"),
        )
        if reboot:
            registry = GroupRegistry()
            registry.reboot = True
            with registry:
                _materialize_and_load_model(
                    model,
                    is_meta_device=True,
                    device=torch.device("cpu"),
                    load_base_model=True,
                    pretrained_path="unused",
                    weights_mapping=None,
                )
        else:
            model.to_empty(device="cpu")
        with torch.no_grad():
            param.reset_sharded_param()
            model.weight.fill_(2)
        optimizer = AdamW(model.parameters())
        return model, optimizer, param, ModelOptimizerState(model, optimizer)

    @staticmethod
    def _step(model, optimizer):
        """Populate committed optimizer state before simulating a replica repair."""
        for parameter in model.parameters():
            parameter.grad = DTensor.from_local(
                torch.ones_like(parameter._local_tensor),
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
                run_check=False,
            )
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            optimizer.step()
        optimizer.zero_grad()

    def test_meta_offload_shard_repair_preserves_communication_view_and_next_update(self) -> None:
        """Feature: CPU-offloaded persistent shards are restored in place.

        Description: Materialize native HSDP host shards, repair an empty optimizer, and continue real updates.
        Expectation: Parameter and communication-view storage stay aliased and states match the donor.
        """
        with patch("torch.distributed.get_rank", return_value=0):
            model, optimizer, param, state = self._build()
            self._step(model, optimizer)
            target, target_optimizer, target_param, target_state = self._build(reboot=True)
            identity = target.weight
            pointer = target_param._sharded_param_data.data_ptr()
            bank = DeviceMetadataBank(0, torch.device("cpu"))
            target_bank = DeviceMetadataBank(1, torch.device("cpu"))
            snapshot_metadata = {
                "state": {"rank_state": {}, "replica_state": {}}, "extra": {}, "buffers": {},
            }
            training_state = snapshot_metadata["state"]
            rank_metadata = snapshot_metadata.copy()
            rank_metadata.pop("state")
            rank_metadata["rank_state"] = training_state["rank_state"]
            bank.put_snapshot(8, rank_metadata, training_state["replica_state"])
            bank.store_rank_snapshot(1, 8, {"rank_state": {}, "extra": {}, "buffers": {}})
            messages = []
            with patch("hyper_parallel.high_availability.replica_transport.dist") as distributed:
                distributed.get_process_group_ranks.return_value = [0, 1]
                distributed.send.side_effect = lambda tensor, **kwargs: messages.append(tensor.clone())
                distributed.recv.side_effect = lambda tensor, **kwargs: tensor.copy_(messages.pop(0))
                sender = ReplicaTransport(object(), bank, state, torch.device("cpu"))
                receiver = ReplicaTransport(object(), target_bank, target_state, torch.device("cpu"))
                sender.send(1, 8)
                receiver.receive(0, 8)
            target_state.restore_model_state({}, {})
            self.assertIs(target.weight, identity)
            self.assertEqual(target_param._sharded_param_data.data_ptr(), pointer)
            self.assertEqual(target.weight._local_tensor.data_ptr(), pointer)
            self.assertEqual(target.weight.device.type, "cpu")
            self.assertTrue(target_param.offload_to_cpu)
            self.assertTrue(param.offload_to_cpu)
            for (_, actual), (_, expected) in zip(target_state.tensor_items(), state.tensor_items()):
                self.assertEqual(actual.device.type, "cpu")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self._step(model, optimizer)
            self._step(target, target_optimizer)
            for (_, actual), (_, expected) in zip(target_state.tensor_items(), state.tensor_items()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(messages)
