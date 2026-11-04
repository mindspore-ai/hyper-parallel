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
from hyper_parallel.core.fault_tolerance.state import ModelOptimizerState
from hyper_parallel.core.fault_tolerance.topology import (
    ReplicaTopology, device_replica_ranks, model_replica_candidates,
)
from hyper_parallel.core.fault_tolerance.transport import DeviceMetadataBank, ReplicaTransport, decode, encode
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


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


class TestDeviceMetadataBank(unittest.TestCase):
    """The persistent bank has rank metadata, never model or optimizer values."""

    def test_owner_metadata_and_bounded_versions(self):
        bank = DeviceMetadataBank(1, torch.device("cpu"))
        for step in (7, 8, 9):
            bank.put(step, {"rng_owner": 1, "cursor": step}, {})
        self.assertEqual(set(bank.get(1, 8)), {"owner", "step", "payload", "buffers"})
        self.assertEqual(decode(bank.get(1, 8)["payload"].numpy().tobytes())["rng_owner"], 1)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            bank.get(1, 7)

    def test_donor_providers_stay_local_while_target_rank_state_is_mirrored(self):
        bank = DeviceMetadataBank(0, torch.device("cpu"))
        bank.replica_providers = frozenset({"scheduler", "logs"})
        for step in (7, 8, 9):
            metadata = {"state": {"loader": {"owner": 0, "position": step}, "python": "rank0-rng",
                                  "providers": {"scheduler": {"step": step}, "logs": {"source_rank": 0},
                                                "custom": {"owner": 0}}}, "extra": {}}
            bank.put(step, metadata, {})
            metadata["state"]["providers"]["scheduler"]["step"] = -1
        local = decode(bank.get(0, 8)["payload"].numpy().tobytes())
        self.assertEqual(local["state"]["providers"], {"custom": {"owner": 0}})
        self.assertEqual(local["state"]["python"], "rank0-rng")
        target = {"state": {"loader": {"owner": 1, "position": 8}, "python": "rank1-rng",
                             "providers": {"custom": {"owner": 1}}}, "extra": {}}
        payload = torch.frombuffer(bytearray(encode(target)), dtype=torch.uint8)
        bank.install(1, 8, payload, {})
        combined = bank.metadata(1, 8)["state"]
        self.assertEqual(combined["python"], "rank1-rng")
        self.assertEqual(combined["loader"], {"owner": 1, "position": 8})
        self.assertEqual(combined["providers"], {"scheduler": {"step": 8}, "logs": {"source_rank": 0},
                                                 "custom": {"owner": 1}})
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

    def test_mirror_reuses_unpadded_payload_and_preserves_variable_lengths(self):
        for peer_size in (2, 19):
            with self.subTest(peer_size=peer_size):
                bank = DeviceMetadataBank(0, torch.device("cpu"))
                original = torch.tensor([3, 4], dtype=torch.uint8)
                peer = torch.arange(peer_size, dtype=torch.uint8)
                bank.install(0, 8, original, {})
                inputs = []

                def _gather(outputs, value, **kwargs):
                    del kwargs
                    inputs.append(value)
                    if len(inputs) == 1:
                        outputs[0].fill_(original.numel())
                        outputs[1].fill_(peer.numel())
                    else:
                        outputs[0].copy_(value)
                        outputs[1][:peer.numel()].copy_(peer)
                    return SimpleNamespace(is_completed=lambda: True, wait=lambda **kwargs: None)

                with patch("hyper_parallel.core.fault_tolerance.transport.dist") as distributed:
                    distributed.get_process_group_ranks.return_value = [0, 1]
                    distributed.all_gather.side_effect = _gather
                    channel = ReplicaTransport(object(), bank, object(), torch.device("cpu"))
                    channel.mirror_metadata(8)
                self.assertEqual(inputs[1].data_ptr() == original.data_ptr(), peer_size == 2)
                torch.testing.assert_close(bank.get(0, 8)["payload"], original, rtol=0, atol=0)
                torch.testing.assert_close(bank.get(1, 8)["payload"], peer, rtol=0, atol=0)

    def test_repair_combines_target_rng_with_exact_donor_scheduler(self):
        _, _, state = self._build("adamw", torch.float32)
        _, _, target_state = self._build("adamw", torch.float32)
        bank, target_bank = (DeviceMetadataBank(rank, torch.device("cpu")) for rank in (0, 1))
        for item in (bank, target_bank):
            item.replica_providers = frozenset({"scheduler"})
        bank.put(8, {"state": {"python": "rank0-rng", "providers": {"scheduler": {"step": 8}}},
                     "extra": {}}, {})
        target = {"state": {"python": "rank1-rng", "providers": {}}, "extra": {}}
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
                         {"python": "rank1-rng", "providers": {"scheduler": {"step": 8}}})
        self.assertEqual(decode(target_bank.get(1, 8)["payload"].numpy().tobytes())["state"]["providers"], {})

    def test_device_send_restores_adamw_muon_and_bf16_main_state(self):
        for kind, dtype in (("adamw", torch.float32), ("muon_adamw", torch.float32),
                            ("muon_adamw", torch.bfloat16)):
            with self.subTest(kind=kind, dtype=dtype):
                model, optimizer, state = self._build(kind, dtype)
                self._step(model, optimizer)
                target, target_optimizer, target_state = self._build(kind, dtype)
                bank = DeviceMetadataBank(0, torch.device("cpu"))
                target_bank = DeviceMetadataBank(1, torch.device("cpu"))
                payload = torch.frombuffer(bytearray(encode({"state": "rank1-rng", "extra": {}})), dtype=torch.uint8)
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
                self.assertEqual(metadata["state"], "rank1-rng")
