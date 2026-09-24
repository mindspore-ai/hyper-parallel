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
"""CPU contracts for grouped experts and physical worker publication."""

import asyncio
from copy import deepcopy
import sys
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

from rl.roles.weight_sync.layout import (
    build_direct_reshard_plan, pack_direct_bucket, resolve_destination_layouts, resolve_source_layouts,
)
from rl.roles.weight_sync.model_adapter import ModelWeightAdapter, _native_expert_descriptions
from rl.roles.weight_sync import vllm_worker
from rl.roles.weight_sync.ipc import IPCWeightTransport, PhysicalRolloutWorker
from rl.roles.weight_sync.packed_weight import PackedWeight, unpack_packed_weights

from hyper_parallel.core.dtensor.placement_types import Shard


class TestMoEWeightSync(unittest.TestCase):
    """Verify numerical expert publication independently of NPU kernels."""

    def setUp(self) -> None:
        """Construct small grouped-expert tensors and their mapping contract."""
        self.config = SimpleNamespace(num_experts=2, hidden_size=3, moe_intermediate_size=4)
        registration = SimpleNamespace(family="qwen3_moe", actor_weight_name=lambda name: name)
        self.adapter = ModelWeightAdapter(registration)
        self.adapter.bind_source(SimpleNamespace(config=self.config))
        self.name = "model.layers.0.mlp.experts.gate_up_proj"
        self.value = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
        self.metadata = self.adapter.packed_metadata([
            PackedWeight(self.name, "float32", tuple(self.value.shape), 4).worker_metadata(),
        ])

    def test_full_gather_projection_values(self) -> None:
        """Canonical HF gate/up weights have the correct expert and axes."""
        actual = dict(unpack_packed_weights(self.value.view(torch.uint8).flatten(), self.metadata))
        for expert in range(2):
            for projection, offset in (("gate", 0), ("up", 4)):
                name = f"model.layers.0.mlp.experts.{expert}.{projection}_proj.weight"
                torch.testing.assert_close(actual[name], self.value[expert, :, offset:offset + 4].T)

        down_name = "model.layers.0.mlp.experts.down_proj"
        down = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        metadata = self.adapter.packed_metadata([
            PackedWeight(down_name, "float32", tuple(down.shape), 4).worker_metadata(),
        ])
        weights = dict(unpack_packed_weights(down.view(torch.uint8).flatten(), metadata))
        for expert in range(2):
            torch.testing.assert_close(weights[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"], down[expert].T)

    def test_invalid_expert_slices_fail(self) -> None:
        """Missing, duplicate and out-of-bounds expert metadata cannot silently load."""
        for failure in ("missing", "duplicate", "bounds"):
            metadata = deepcopy(self.metadata)
            slices = metadata[0]["canonical_experts"]
            if failure == "missing":
                slices.pop()
            elif failure == "duplicate":
                slices.append(deepcopy(slices[0]))
            else:
                slices[0]["starts"][0] = 2
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                unpack_packed_weights(self.value.view(torch.uint8).flatten(), metadata)

    def test_direct_routes_preserve_physical_expert_storage(self) -> None:
        """EP/EDP source shards reconstruct non-contiguous rollout expert ownership."""
        self.config.num_experts = 4
        for projection, shape, suffix in (("gate_up_proj", (4, 3, 8), "w13_weight"),
                                          ("down_proj", (4, 4, 3), "w2_weight")):
            name = f"model.layers.0.mlp.experts.{projection}"
            full = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
            values, descriptions = {}, []
            for rank in range(4):
                ep_rank, edp_rank = divmod(rank, 2)
                local = full.chunk(2, dim=0)[ep_rank].chunk(2, dim=1)[edp_rank].contiguous()
                mesh = SimpleNamespace(ndim=2, size=lambda dim: 2,
                                       get_coordinate=lambda ep=ep_rank, edp=edp_rank: [ep, edp])
                shard = SimpleNamespace(shape=shape, placements=(Shard(0), Shard(1)), device_mesh=mesh,
                                        to_local=lambda value=local: value)
                values[rank] = {name: local}
                descriptions.append(self.adapter.direct_source_descriptions({name: shard}, rank))
            sources = resolve_source_layouts(descriptions)
            workers = []
            target_name = f"model.layers.0.mlp.experts.{suffix}"
            targets = [torch.zeros((2,) + shape[1:]) for _ in range(2)]
            for rank in range(2):
                mapping = [0, -1, 1, -1] if rank == 0 else [-1, 0, -1, 1]
                layer = SimpleNamespace(ep_size=2, tp_size=1, tp_rank=0, expert_map=torch.tensor(mapping))
                tensors = _native_expert_descriptions(target_name, targets[rank], layer, self.config)
                workers.append({'worker_rank': rank, 'dp_rank': rank, 'tp_rank': 0, 'tp_size': 1, 'tensors': tensors})
            shapes = {item.name: item.global_shape for item in sources}
            destinations = resolve_destination_layouts(workers, shapes)
            plan = build_direct_reshard_plan(sources, destinations, source_world_size=4, bucket_size_bytes=32)
            self.assertEqual(plan.destination_worker_size, 2)
            for (source_rank, rank), buckets in plan.buckets.items():
                for bucket in buckets:
                    packed = pack_direct_bucket(values[source_rank], bucket, torch.device("cpu"))
                    for entry in bucket.entries:
                        value = packed.narrow(0, entry.buffer_offset, entry.num_bytes).view(torch.float32)
                        region = tuple(slice(start, start + length)
                                       for start, length in zip(entry.destination_starts, entry.lengths))
                        targets[rank][region].copy_(value.reshape(entry.lengths))
            for rank in range(2):
                torch.testing.assert_close(targets[rank], full[rank::2])
            workers[1]["tp_rank"] = 1
            with self.assertRaisesRegex(ValueError, "coordinates"):
                resolve_destination_layouts(workers, shapes)

    def test_full_gather_reload_owns_buffers_and_requires_complete_weights(self) -> None:
        """IPC inputs survive producer release; partial or duplicate updates fail."""
        gate, up = "model.layers.0.mlp.experts.0.gate_proj.weight", "model.layers.0.mlp.experts.0.up_proj.weight"
        loaded = []
        model = SimpleNamespace(load_weights=lambda weights: loaded.extend(weights) or {weights[0][0]})
        worker = SimpleNamespace(_hyper_layerwise_reload=True, _hyper_expected_weights={gate, up},
                                 _hyper_received_weights=set(), model_config=SimpleNamespace(hf_config=self.config))
        reload_module = SimpleNamespace(finalize_layerwise_reload=lambda *args: None,
                                        initialize_layerwise_reload=lambda *args: None)
        source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        expected = source.clone()
        with patch.dict(sys.modules, {"vllm.model_executor.model_loader.reload": reload_module}):
            vllm_worker._load_moe_checkpoint_weights(worker, model, [(gate, source)])
            source.zero_()
            torch.testing.assert_close(loaded[0][1], expected)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                vllm_worker._finalize_moe_checkpoint_reload(worker)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                vllm_worker._load_moe_checkpoint_weights(worker, model, [(gate, source)])

    def test_expert_ack_requires_every_physical_worker(self) -> None:
        """A successful first DP replica cannot hide missing expert workers."""
        transport = IPCWeightTransport(data_parallel_size=2, tensor_parallel_size=1)
        context = SimpleNamespace(workers=(PhysicalRolloutWorker(0, 0, "a"), PhysicalRolloutWorker(1, 0, "b")))
        results = [{'received': True, 'dp_rank': 0, 'tp_rank': 0, 'physical_device_id': "a", 'bytes': 32}]
        with self.assertRaisesRegex(RuntimeError, "every physical worker"):
            transport._validate_results(results, context, -1, 32, require_all_workers=True)
        results.append({'received': True, 'dp_rank': 1, 'tp_rank': 0, 'physical_device_id': "b", 'bytes': 32})
        transport._validate_results(results, context, -1, 32, require_all_workers=True)

    def test_dp_rpc_aggregates_weights_and_preserves_other_methods(self) -> None:
        """Weight RPC acknowledgements cover every engine without changing other RPCs."""
        class Client:
            """Expose the fixed-image DP client contract without starting engines."""

            core_engines = (0, 1)

            async def collective_rpc_async(
                self, method: str, timeout: Any = None, args: tuple = (), kwargs: Any = None,
            ) -> list:
                """Return the original result for non-weight RPCs."""
                del timeout, args, kwargs
                return ["original", method]

            async def _call_utility_async(self, *args, engine):
                return [{'engine': engine, 'method': args[1]}]

        with (
            patch.dict(sys.modules, {"vllm.v1.engine.core_client": SimpleNamespace(DPLBAsyncMPClient=Client)}),
            patch.object(vllm_worker, "_patch_state", vllm_worker._PatchState()),
        ):
            vllm_worker._patch_dp_weight_rpc()
            installed = Client.collective_rpc_async
            vllm_worker._patch_dp_weight_rpc()
            self.assertIs(Client.collective_rpc_async, installed)
            client = Client()
            results = asyncio.run(client.collective_rpc_async("get_policy_version"))
            self.assertEqual(results, [{'engine': 0, 'method': "get_policy_version"},
                                       {'engine': 1, 'method': "get_policy_version"}])
            self.assertEqual(asyncio.run(client.collective_rpc_async("other")), ["original", "other"])

    def test_native_expert_map_rejects_duplicate_local_ids(self) -> None:
        """Two global experts must never alias one local expert slot."""
        layer = SimpleNamespace(ep_size=2, tp_size=1, tp_rank=0, expert_map=torch.tensor([0, 0]))
        with self.assertRaisesRegex(ValueError, "uniquely"):
            _native_expert_descriptions("model.layers.0.mlp.experts.w13_weight", self.value, layer, self.config)
