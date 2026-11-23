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
"""Host Engram owner inference, sparse-state repair and interrupted-step cleanup."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.modules.engram.host_engram_table import HostEngramTable
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.high_availability.model_recovery import ModelRecoveryAdapter
from hyper_parallel.high_availability.recovery_state import ModelOptimizerState
from hyper_parallel.high_availability.replica_topology import ReplicaTopology, model_replica_candidates
from hyper_parallel.high_availability.replica_transport import DeviceMetadataBank, ReplicaTransport
from hyper_parallel.high_availability.validation import validate_recovery_model
from hyper_parallel.models._transformers.model_builder import _materialize_and_load_model
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import DeepseekV41HostState


def _build_host_model(ep_rank=0, ep_size=1, mesh=None, materialize=True, fp32_main=False):
    """Construct native Host storage and its normal dense/sparse optimizer coordinator."""
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(4, 4, dtype=torch.float32 if fp32_main else torch.bfloat16)
    model.embed = HostEngramTable(
        source_weight=torch.nn.Parameter(torch.empty(8, 4, device="meta")),
        logical_rows=7,
    )
    model.embed.weight = torch.nn.Parameter(torch.empty(8 // ep_size, 4, device="meta"))
    model.embed.bind_planned_shard(ep_rank=ep_rank, ep_size=ep_size, ep_group=object())
    external = DeepseekV41HostState(model, {"embed": model.embed}, mesh)
    model._hp_external_state = external
    if materialize:
        external.materialize_for_recovery()
        with torch.no_grad():
            model.embed.weight.copy_(torch.arange(model.embed.weight.numel()).reshape_as(model.embed.weight) / 10)
    dense = ChainedOptimizer(model, {"dense": AdamW(model.dense.parameters(), lr=0.01)})
    target = SimpleNamespace(build=lambda **kwargs: SimpleNamespace(get_optimizer=lambda: dense))
    config = SimpleNamespace(target=target)
    def wrap_dense(value):
        return Float16OptimizerWithFloat16Params(value, model) if fp32_main else value

    optimizer = external.build_optimizer(config, wrap_dense=wrap_dense)
    return model, external, optimizer


def _step(model, external, optimizer, rows=True):
    """Run native gradient preparation and the coordinated dense/SparseAdam update."""
    loss = model.dense(torch.ones(3, 4, dtype=model.dense.weight.dtype)).float().square().mean()
    if rows:
        loss = loss + model.embed(torch.tensor([0, 2, 0])).square().mean()
    loss.backward()
    external.prepare_optimizer_step(max_norm=0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach()


class TestHostEngramRecovery(unittest.TestCase):
    """Run real CPU table and optimizer code; simulate only inter-rank transport."""

    def setUp(self):
        """Preserve the caller's random state around deterministic model construction."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(13)

    def test_replacement_materializes_host_rows_without_loading_or_initializing(self):
        """Feature: Collective-free replacement initialization.

        Description: Follow the model-builder replacement branch with meta Host rows.
        Expectation: Allocate CPU storage in place without calling the ordinary weight loader.
        """
        model, external, _ = _build_host_model(materialize=False)
        parameter = model.embed.weight
        with patch("hyper_parallel.models._transformers.model_builder.is_recovery_bootstrapping", return_value=True), \
                patch.object(external, "materialize", side_effect=AssertionError("unexpected load")), \
                patch("hyper_parallel.models._transformers.model_builder._initialize_model_weights",
                      side_effect=AssertionError("unexpected initialization")):
            _materialize_and_load_model(
                model,
                is_meta_device=True,
                device="cpu",
                load_base_model=True,
                pretrained_path="/unused",
                weights_mapping=None,
            )
        self.assertIs(model.embed.weight, parameter)
        self.assertIs(external.parameters["embed.weight"], parameter)
        self.assertEqual(parameter.device.type, "cpu")
        self.assertEqual(parameter.dtype, torch.float32)
        external.check_identities()

    def test_dense_and_host_owner_replica_domains_are_intersected(self):
        """Feature: Full replicas include CPU table owner coordinates.

        Description: Combine dense replicated/sharded axes with an EP-owned Host table.
        Expectation: Only ranks holding both equivalent dense shards and Host rows remain.
        """
        moe = SimpleNamespace(
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("edp_replicate", "edp_shard", "ep"),
            rank_list=(4, 1, 6, 3, 0, 5, 2, 7),
        )
        mesh = SimpleNamespace(device_mesh=moe, fsdp_moe_mesh=moe)
        _, external, _ = _build_host_model(ep_rank=1, ep_size=2, mesh=mesh)
        self.assertEqual(external.recovery_replica_ranks(1), {1, 3, 5, 7})
        dense = SimpleNamespace(
            device_mesh=moe,
            placements=tuple(SimpleNamespace(is_replicate=lambda value=value: value) for value in (True, False, True)),
        )
        state = SimpleNamespace(parameters={"dense": dense, **external.parameters}, external_state=external)
        self.assertEqual(model_replica_candidates(state, 1, 8), (1, 5))

    def test_unique_host_owners_fail_before_registration(self):
        """Feature: Recovery requires an existing complete replica.

        Description: Use two EP owners without a same-owner replica axis.
        Expectation: Reject enabling recovery rather than pair different row intervals.
        """
        moe = SimpleNamespace(mesh_shape=(1, 2), mesh_dim_names=("edp_shard", "ep"), rank_list=(0, 1))
        mesh = SimpleNamespace(device_mesh=moe, fsdp_moe_mesh=moe)
        model, _, optimizer = _build_host_model(ep_rank=1, ep_size=2, mesh=mesh)
        state = ModelOptimizerState(model, optimizer)
        with self.assertRaisesRegex(ValueError, "no equivalent device replica"):
            ReplicaTopology.discover(state, 1, 2, reboot=True)

    def test_same_shape_different_host_row_intervals_cannot_repair(self):
        """Feature: Repair headers validate global row ownership.

        Description: Bind identically shaped tables from different EP owners.
        Expectation: Reject the donor schema before writing any tensors.
        """
        first, _, first_optimizer = _build_host_model(ep_size=2)
        second, _, second_optimizer = _build_host_model(ep_rank=1, ep_size=2)
        donor = ModelOptimizerState(first, first_optimizer)
        receiver = ModelOptimizerState(second, second_optimizer)
        with self.assertRaisesRegex(RuntimeError, "schema/layout mismatch"):
            receiver.prepare_receive(donor.metadata())

    def test_clean_discards_sparse_gradients_without_modifying_committed_state(self):
        """Feature: Surviving workers discard the interrupted global batch.

        Description: Accumulate pending sparse rows after a completed optimizer step.
        Expectation: Clear pending gradients and coordinator phase while preserving weights and moments.
        """
        model, external, optimizer = _build_host_model()
        _step(model, external, optimizer)
        before = {name: tensor.clone() for name, tensor in ModelOptimizerState(model, optimizer).tensor_items()}
        model.embed.append_rows(torch.tensor([1]), torch.ones(1, 4))
        external.prepare_optimizer_step(max_norm=0)
        self.assertEqual(optimizer.state, "PREPARED")
        ModelRecoveryAdapter(model).abort_iteration()
        self.assertEqual(optimizer.state, "ACCUMULATING")
        self.assertFalse(model.embed.pending_chunks)
        self.assertIsNone(model.embed.pending)
        self.assertIsNone(model.embed.weight.grad)
        for name, tensor in ModelOptimizerState(model, optimizer).tensor_items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_repair_and_repeated_sparse_updates_match_donor_exactly(self):
        """Feature: Direct Host weight and SparseAdam repair.

        Description: Clean interrupted survivors or start cold receivers, repair, then repeat after further training.
        Expectation: Preserve parameter identities and match all dense/Host tensors, scalar steps and continuation.
        """
        for donor_rows, warm_receiver, fp32_main in ((True, False, False), (True, True, True), (False, True, False)):
            with self.subTest(donor_rows=donor_rows, warm_receiver=warm_receiver, fp32_main=fp32_main):
                model, external, optimizer = _build_host_model(fp32_main=fp32_main)
                target, target_external, target_optimizer = _build_host_model(fp32_main=fp32_main)
                parameter = target.embed.weight
                _step(model, external, optimizer, rows=donor_rows)
                if warm_receiver:
                    _step(target, target_external, target_optimizer)
                donor_state = ModelOptimizerState(model, optimizer)
                target_state = ModelOptimizerState(target, target_optimizer)
                for step in (1, 2):
                    if warm_receiver or step > 1:
                        # A survivor discards the failed batch in clean; a new
                        # replacement already starts with empty iteration state.
                        target.embed(torch.tensor([1])).sum().backward()
                        target_external.prepare_optimizer_step(max_norm=0)
                        ModelRecoveryAdapter(target).abort_iteration()
                        self.assertEqual(target_optimizer.state, "ACCUMULATING")
                        self.assertIsNone(target.embed.pending)
                    self._repair(donor_state, target_state, step)
                    target_state.restore_rank_state({}, {})
                    self.assertIs(target.embed.weight, parameter)
                    self.assertEqual(target_state.metadata(), donor_state.metadata())
                    self.assertEqual(target_optimizer.state, "ACCUMULATING")
                    paired = zip(target_state.tensor_items(), donor_state.tensor_items())
                    for (name, actual), (other, expected) in paired:
                        self.assertEqual(name, other)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    loss = _step(model, external, optimizer)
                    recovered_loss = _step(target, target_external, target_optimizer)
                    torch.testing.assert_close(recovered_loss, loss, rtol=0, atol=0)
                    for (_, actual), (_, expected) in zip(target_state.tensor_items(), donor_state.tensor_items()):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _repair(self, donor_state, target_state, step):
        """Exercise the native repair header and payload order through a CPU message queue."""
        bank = DeviceMetadataBank(0, torch.device("cpu"))
        metadata = {"state": {"rank_state": {}, "replica_state": {}}, "extra": {}}
        bank.put(step, metadata, {})
        bank.install(1, step, bank.get(0, step)["payload"], {})
        target_bank = DeviceMetadataBank(1, torch.device("cpu"))
        messages = []
        with patch("hyper_parallel.high_availability.replica_transport.dist") as distributed, \
                patch("hyper_parallel.high_availability.replica_transport._TRANSFER_CHUNK_BYTES", 16):
            distributed.get_process_group_ranks.return_value = [0, 1]
            distributed.send.side_effect = lambda tensor, **kwargs: messages.append(tensor.clone())
            distributed.recv.side_effect = lambda tensor, **kwargs: tensor.copy_(messages.pop(0))
            ReplicaTransport(object(), bank, donor_state, torch.device("cpu")).send(1, step)
            ReplicaTransport(object(), target_bank, target_state, torch.device("cpu")).receive(0, step)
        self.assertFalse(messages)

    def test_unknown_external_state_and_unowned_sparse_optimizer_remain_rejected(self):
        """Feature: Only the adapted Host optimizer lifecycle is admitted.

        Description: Attach an unknown external object or an unrelated SparseAdam optimizer.
        Expectation: Preserve explicit errors for unsupported recovery contracts.
        """
        model = torch.nn.Linear(4, 4)
        model._hp_external_state = object()
        with self.assertRaisesRegex(ValueError, "external state"):
            validate_recovery_model(model)
        del model._hp_external_state
        with self.assertRaisesRegex(ValueError, "Host Engram SparseAdam"):
            ModelOptimizerState(model, torch.optim.SparseAdam(model.parameters()))

    def test_host_receive_uses_bounded_device_staging_and_preserves_noncontiguous_target(self):
        """Feature: Host repair does not allocate a full table on the device.

        Description: Receive small blocks into a transposed CPU destination via simulated device memory.
        Expectation: Allocate at most one block on device and preserve destination storage and order.
        """
        target = torch.empty(3, 5).t()
        expected = torch.arange(15, dtype=target.dtype).reshape_as(target)
        messages = list(expected.contiguous().view(-1).split(4))
        allocated = []
        allocate = torch.empty
        pointer = target.data_ptr()
        transport = ReplicaTransport.__new__(ReplicaTransport)
        transport.device = torch.device("meta")

        def stage(size, **kwargs):
            allocated.append(size)
            return allocate(size, dtype=kwargs["dtype"], device="cpu")

        with patch("hyper_parallel.high_availability.replica_transport.torch.empty", side_effect=stage), \
                patch("hyper_parallel.high_availability.replica_transport._TRANSFER_CHUNK_BYTES", 16), \
                patch("hyper_parallel.high_availability.replica_transport.dist.recv",
                      side_effect=lambda block, **kwargs: block.copy_(messages.pop(0))):
            transport._receive_tensor(target, 0)
        self.assertEqual(allocated, [4])
        self.assertEqual(target.data_ptr(), pointer)
        self.assertFalse(messages)
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
