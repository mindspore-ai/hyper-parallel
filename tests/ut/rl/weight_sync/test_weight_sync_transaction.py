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
"""CPU contracts for one-shot direct and full-gather publication."""
# pylint: disable=forbidden-backend-import,missing-public-docstring,missing-public-type-hints,protected-access

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from rl.roles.model_setup import ModelRegistration, resolve_vllm_model
from rl.roles.weight_sync import vllm_worker
from rl.roles.weight_sync.layout import pack_direct_bucket
from rl.roles.weight_sync.packed_weight import PackedWeightAck
from rl.roles.weight_sync.sync import (
    ActorRolloutWeightSync,
    PolicySnapshot,
    coordinator_call,
    synchronized_call,
)
from rl.roles.weight_sync.transfer import (
    DirectReshardStrategy,
    FullGatherStrategy,
    WeightPublisher,
    WeightSource,
)
from rl.roles.weight_sync.vllm_client import VLLMWeightSyncClientMixin


def _model():
    return resolve_vllm_model(ModelRegistration(
        "qwen", "qwen3", "/model", "/tokenizer",
        "Qwen3ForCausalLM", "qwen3", "qwen3", False,
    ), "hyper")


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    """Provide a single-rank CPU runtime for real weight publication tests."""
    handle = SimpleNamespace(
        current_device=lambda: 0,
        current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        dist,
        "all_gather_object",
        lambda results, value: results.__setitem__(0, value),
    )
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device("cpu"))
    monkeypatch.setattr(torch, 'get_device_module', lambda _kind=None: handle)
    monkeypatch.setattr(
        torch.get_device_module(),
        'current_stream',
        lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(dist, "broadcast", lambda _tensor, src: None)


class LocalWorkers(VLLMWeightSyncClientMixin):
    """Execute actual worker writes and version commits on CPU."""

    base_url = "http://shared"
    is_server_owner = True

    def _request(self, method, route, payload=None, timeout=None, base_url=None):
        """Reject HTTP calls: this fixture exercises real updates through local workers."""
        del payload, timeout, base_url
        raise AssertionError(f"Unexpected HTTP request in local worker fixture: {method} {route}")

    def __init__(self):
        self.events = []
        self.rpc_methods = []
        self.paused = False
        self.sleeping = False
        self.resume_failure = False
        self.workers = []

        def make_loader(parameter_map, worker_rank):
            def load_weights(weights, require_all=True):
                del require_all
                loaded = set()
                for name, value in weights:
                    target = parameter_map[name]
                    if "norm" in name:
                        target.copy_(value)
                    else:
                        target.copy_(
                            value[:, worker_rank * 2:(worker_rank + 1) * 2]
                        )
                    loaded.add(name)
                return loaded

            return load_weights

        for rank in range(2):
            parameters = {
                "model.norm.weight": torch.zeros(4),
                "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 2),
            }
            model = SimpleNamespace(
                named_parameters=lambda values=parameters: values.items(),
                load_weights=make_loader(parameters, rank),
            )
            self.workers.append(SimpleNamespace(
                parameters=parameters,
                model_runner=SimpleNamespace(get_model=lambda value=model: value),
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["HyperQwen3ForCausalLM"]
                    )
                ),
                _check_weight_transfer_engine=lambda: None,
                _weight_update_active=False,
                _hyper_loaded_policy_version=0,
                _hyper_pending_policy_version=None,
                tp_rank=rank,
            ))

    def get_world_size(self, base_url=None):
        return 2

    def pause(self):
        self.events.append("pause")
        self.paused = True

    def is_paused(self):
        return self.paused

    def sleep(self, level=1, mode="wait"):
        del level, mode
        self.events.append("sleep")
        self.sleeping = True
        self.paused = True

    def is_sleeping(self):
        return self.sleeping

    def wake_up(self, tags):
        self.events.append("wake:" + ",".join(tags))
        self.sleeping = False

    def resume(self):
        self.events.append("resume")
        self.paused = False
        if self.resume_failure:
            raise RuntimeError("resume failed")

    def start_weight_update(self):
        self.events.append("start")
        for worker in self.workers:
            worker._weight_update_active = True
            worker._hyper_pending_policy_version = None

    def finish_weight_update(self):
        self.events.append("finish")
        for worker in self.workers:
            vllm_worker._finish_custom_weight_update(worker)

    def collective_rpc(self, method, kwargs=None, base_url=None):
        del kwargs, base_url
        self.rpc_methods.append(method)
        if method == "get_direct_reshard_layout":
            return [self._layout(worker) for worker in self.workers]
        if method == "get_policy_version":
            return [vllm_worker.get_policy_version(worker) for worker in self.workers]
        raise AssertionError(method)

    @staticmethod
    def _layout(worker):
        return {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": worker.tp_rank,
            "tp_size": 2,
            "physical_device_id": f"npu-{worker.tp_rank}",
            "tensors": [
                {
                    "name": name,
                    "dtype_name": "float32",
                    "element_size": 4,
                    "local_shape": list(value.shape),
                    "placement": "replicate" if "norm" in name else "shard",
                    "shard_dim": None if "norm" in name else 1,
                }
                for name, value in worker.parameters.items()
            ],
        }

    def apply(self, rank, metadata, packed, version):
        worker = self.workers[rank]
        vllm_worker._validate_update(worker, version, transport="CPU test")
        vllm_worker._apply_direct_bucket(
            worker.parameters,
            packed,
            metadata,
            transport="CPU test",
        )
        worker._hyper_pending_policy_version = version


class TensorTransport:
    """Replace wire I/O while keeping real planning, packing and worker writes."""

    def __init__(self, name):
        self.name = name
        self.fail = False
        self.closed = False

    def prepare(self, unused_client, destination_tp_size):
        del unused_client
        if destination_tp_size != 2:
            raise ValueError("unexpected destination size")

    def prepare_packed(self, unused_client):
        del unused_client
        return "packed"

    def transfer_direct(self, client, state, plan, version):
        for (unused_source, target), buckets in plan.buckets.items():
            for bucket in buckets:
                packed = pack_direct_bucket(state, bucket, torch.device("cpu"))
                client.apply(target, bucket.worker_metadata(), packed, version)
                if self.fail:
                    raise RuntimeError("direct transfer failed")

    def send_packed_bucket(
        self, client, unused_context, index, metadata, total_bytes, packed, version,
    ):
        """Load a complete packed bucket into each in-process worker."""
        del unused_context
        assert packed is not None and packed.numel() == total_bytes
        for worker in client.workers:
            vllm_worker._validate_update(
                worker,
                version,
                transport="CPU packed test",
            )
            vllm_worker._load_packed_weights(worker, packed, metadata)
            worker._hyper_pending_policy_version = version
        if self.fail:
            raise RuntimeError("full-gather transfer failed")
        return PackedWeightAck(index, total_bytes, len(client.workers))

    def close(self):
        self.closed = True


def _publication(deployment, strategy):
    """Assemble an in-process publisher, transport and versioned worker client."""
    source = WeightSource(_model(), 16)
    selected = (
        DirectReshardStrategy(
            source,
            data_parallel_size=1,
            tensor_parallel_size=2,
        )
        if strategy == "direct_reshard"
        else FullGatherStrategy(source)
    )
    wire = TensorTransport("ipc" if deployment == "colocated" else "hccl")
    publisher = WeightPublisher(selected, wire)
    client = LocalWorkers()
    controller = ActorRolloutWeightSync("qwen", deployment, lambda: client, publisher)
    state = {
        "model.norm.weight": torch.arange(4, dtype=torch.float32) + 1,
        "model.layers.0.mlp.down_proj.weight": (
            torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1
        ),
    }
    return publisher, wire, client, controller, state


def _assert_parameters(client, state):
    for worker in client.workers:
        torch.testing.assert_close(
            worker.parameters["model.norm.weight"],
            state["model.norm.weight"],
        )
        torch.testing.assert_close(
            worker.parameters["model.layers.0.mlp.down_proj.weight"],
            state["model.layers.0.mlp.down_proj.weight"][
                :, worker.tp_rank * 2:(worker.tp_rank + 1) * 2
            ],
        )


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
@pytest.mark.parametrize("strategy", ["direct_reshard", "full_gather"])
def test_publication_writes_parameters_and_commits_version(deployment, strategy):
    """Verify parameter contents and policy commit across deployment and strategy combinations."""
    publisher, wire, client, controller, state = _publication(deployment, strategy)
    controller.prepare_for_training()
    controller.update_weights(PolicySnapshot(1, "qwen", state))
    if deployment == "colocated":
        assert controller.policy_version == 0
        controller.prepare_for_rollout()
    _assert_parameters(client, state)
    assert controller.policy_version == 1
    assert publisher.last_strategy == strategy
    assert ("get_direct_reshard_layout" in client.rpc_methods) is (
        strategy == "direct_reshard"
    )
    assert client.events.index("start") < client.events.index("finish")
    assert client.events.index("finish") < client.events.index("resume")
    controller.close()
    assert wire.closed


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
@pytest.mark.parametrize("strategy", ["direct_reshard", "full_gather"])
def test_transfer_failure_propagates_without_recovery(deployment, strategy):
    """Verify transfer errors propagate without committing or attempting compensation."""
    publisher, wire, client, controller, state = _publication(deployment, strategy)
    wire.fail = True
    controller.prepare_for_training()
    with pytest.raises(RuntimeError, match="transfer failed"):
        controller.update_weights(PolicySnapshot(1, "qwen", state))
    assert controller.policy_version == 0
    assert client.paused
    assert "finish" not in client.events
    assert "resume" not in client.events
    assert publisher.last_strategy is None


@pytest.mark.parametrize("deployment", ["colocated", "disjoint"])
def test_resume_failure_propagates_without_compensation(deployment):
    """Verify resume errors propagate without rolling back an acknowledged update."""
    unused_publisher, unused_wire, client, controller, state = _publication(
        deployment, "full_gather"
    )
    client.resume_failure = True
    controller.prepare_for_training()
    with pytest.raises(RuntimeError, match="resume failed"):
        controller.update_weights(PolicySnapshot(1, "qwen", state))
        controller.prepare_for_rollout()
    assert controller.policy_version == 0
    assert client.events[-1] == "resume"


def test_strategies_cache_plans_and_reject_missing_sources():
    """Verify plan reuse and rejection of a missing source model."""
    for strategy_name in ("direct_reshard", "full_gather"):
        publisher, wire, client, unused_controller, state = _publication(
            "disjoint", strategy_name
        )
        publisher.strategy.prepare(client, state, wire)
        plan = (
            publisher.strategy._plan
            if strategy_name == "direct_reshard"
            else publisher.strategy._buckets
        )
        publisher.strategy.prepare(client, state, wire)
        cached = (
            publisher.strategy._plan
            if strategy_name == "direct_reshard"
            else publisher.strategy._buckets
        )
        assert cached is plan
        with pytest.raises(ValueError, match="missing"):
            publisher.strategy.prepare(
                client,
                {"model.norm.weight": state["model.norm.weight"]},
                wire,
            )


def test_coordinator_and_synchronized_failures_are_not_swallowed():
    def fail():
        raise ValueError("local operation failed")

    for call in (coordinator_call, synchronized_call):
        with pytest.raises(ValueError, match="local operation failed"):
            call("test", fail)
