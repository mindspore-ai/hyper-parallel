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
"""CPU unit tests for production rollout-worker weight synchronization logic."""
# Local fakes replace vLLM/NPU boundaries while worker functions execute unchanged.
# pylint: disable=forbidden-backend-import,missing-public-docstring,protected-access,unnecessary-lambda

import base64
import pickle
import sys
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

import rl.roles.weight_sync.vllm_worker as worker_module
from rl.roles.weight_sync.vllm_worker import (
    get_direct_reshard_layout,
    get_policy_version,
    init_direct_reshard_group,
    init_packed_weight_group,
    install_vllm_weight_sync_hooks,
    receive_direct_reshard,
    receive_ipc_direct_reshard,
    receive_ipc_packed_weights,
    receive_packed_weights,
)


class _Placement:
    """Expose the placement predicates consumed by worker layout metadata."""

    def __init__(self, shard_dim: Optional[int]) -> None:
        self.dim = shard_dim

    def is_shard(self) -> bool:
        return self.dim is not None

    def is_replicate(self) -> bool:
        return self.dim is None


class _Model(torch.nn.Module):
    """Provide replicated norm and sharded matrix parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("norm_weight", torch.nn.Parameter(torch.tensor([1.0, 2.0])))
        self.register_parameter("matrix_weight", torch.nn.Parameter(torch.arange(4.0).reshape(2, 2)))
        self._tp_placements = {
            "model.norm.weight": (_Placement(None),),
            "model.matrix.weight": (_Placement(0),),
        }

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return iter(
            (
                ("model.norm.weight", self.norm_weight),
                ("model.matrix.weight", self.matrix_weight),
            )
        )


def _worker(model: Any) -> Any:
    """Build one Hyper Qwen3 worker around the supplied model."""
    return SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: model),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["HyperQwen3ForCausalLM"])
        ),
        parallel_config=SimpleNamespace(data_parallel_index=0, data_parallel_size=1),
        _hyper_loaded_policy_version=1,
        _hyper_pending_policy_version=None,
    )


def test_worker_reports_layout_and_committed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker layout and committed version remain available without weight hashing."""
    tp_group = SimpleNamespace(rank_in_group=0, world_size=2)
    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(get_tp_group=lambda: tp_group))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        SimpleNamespace(npu_generate_uuid=lambda: "host-4"),
    )
    monkeypatch.setenv("HYPER_RL_ROLLOUT_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 7)
    model = _Model()
    worker = _worker(model)

    layout = get_direct_reshard_layout(worker)

    assert layout["dp_rank"] == 0
    assert layout["tp_rank"] == 0
    assert layout["tp_size"] == 2
    assert layout["physical_device_id"] == "host-4"
    assert [(item["name"], item["placement"], item["shard_dim"]) for item in layout["tensors"]] == [
        ("model.matrix.weight", "shard", 0),
        ("model.norm.weight", "replicate", None),
    ]
    assert get_policy_version(worker) == {"version": 1}


class _ReceiveModel(torch.nn.Module):
    """Expose one physical TP-local parameter receiving fragments."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((2, 2)))
        self.load_modes = []

    def load_weights(self, weights: Any, *, require_all: bool = True) -> set[str]:
        """Load complete test parameters through the vLLM model boundary."""
        self.load_modes.append(require_all)
        loaded = set()
        with torch.no_grad():
            for name, value in weights:
                if name != "weight":
                    raise ValueError(name)
                self.weight.copy_(value)
                loaded.add(name)
        return loaded


def _receive_worker() -> Any:
    """Build an active worker ready to receive versioned weights."""
    model = _ReceiveModel()
    return SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: model),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["HyperQwen3ForCausalLM"])
        ),
        _weight_update_active=True,
        _check_weight_transfer_engine=lambda: None,
        _hyper_loaded_policy_version=0,
        _hyper_pending_policy_version=None,
    )


def _bucket_metadata() -> list[dict[str, Any]]:
    """Return one complete 2x2 float32 destination write."""
    return [
        {
            "total_bytes": 16,
            "entries": [
                {
                    "name": "weight",
                    "dtype_name": "float32",
                    "element_size": 4,
                    "destination_starts": [0, 0],
                    "lengths": [2, 2],
                    "buffer_offset": 0,
                    "num_bytes": 16,
                }
            ],
        }
    ]


def _packed_metadata() -> list[dict[str, Any]]:
    """Return one complete parameter for vLLM load_weights."""
    return [{
        "name": "weight",
        "dtype_name": "float32",
        "shape": [2, 2],
        "buffer_offset": 0,
        "num_bytes": 16,
    }]


def test_worker_receives_direct_reshard_and_commits_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HCCL receive writes the TP-local slice before committing the pending version."""
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class Group:
        device = torch.device("cpu")

        @staticmethod
        def broadcast(packed: torch.Tensor, src: int) -> None:
            assert src == 0
            packed.copy_(source.view(torch.uint8).view(-1))

    synchronizations: list[str] = []
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": 0,
            "tp_size": 1,
            "physical_device_id": "host-0",
        },
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: SimpleNamespace(
                synchronize=lambda: synchronizations.append("sync")
            )
        ),
    )
    worker = _receive_worker()
    worker._hyper_direct_reshard_groups = {"route": Group()}

    result = receive_direct_reshard(
        worker,
        group_id="route",
        target_tp_rank=0,
        buckets=_bucket_metadata(),
        policy_version=1,
        expected_data_parallel_size=1,
        expected_tensor_parallel_size=1,
    )

    torch.testing.assert_close(worker.model_runner.get_model().weight, source)
    assert result == {
        "received": True,
        "dp_rank": 0,
        "tp_rank": 0,
        "bytes": 16,
        "bucket_count": 1,
    }
    assert worker._hyper_pending_policy_version == 1
    assert worker._hyper_loaded_policy_version == 0
    worker_module._finish_custom_weight_update(worker)
    assert worker._hyper_loaded_policy_version == 1
    assert worker._hyper_pending_policy_version is None
    assert synchronizations == ["sync"]


def test_worker_receives_packed_hccl_weights_through_model_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed HCCL passes complete parameters to Hyper load_weights incrementally."""
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class Group:
        device = torch.device("cpu")

        @staticmethod
        def broadcast(packed: torch.Tensor, src: int) -> None:
            assert src == 0
            packed.copy_(source.view(torch.uint8).view(-1))

    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": 0,
            "tp_size": 1,
            "physical_device_id": "host-0",
        },
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: SimpleNamespace(synchronize=lambda: None)
        ),
    )
    worker = _receive_worker()
    worker._hyper_packed_weight_groups = {"packed": Group()}

    result = receive_packed_weights(
        worker,
        group_id="packed",
        metadata=_packed_metadata(),
        total_bytes=16,
        policy_version=1,
    )

    model = worker.model_runner.get_model()
    torch.testing.assert_close(model.weight, source)
    assert model.load_modes == [False]
    assert result["bytes"] == 16
    assert worker._hyper_pending_policy_version == 1


def test_worker_receives_ipc_reshard_and_releases_buffers_after_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPC receive imports one buffer, writes its parameter, and synchronizes before release."""
    source = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    synchronizations: list[str] = []
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": 0,
            "tp_size": 1,
            "physical_device_id": "host-4",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        SimpleNamespace(npu_generate_uuid=lambda: "host-4"),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.multiprocessing.reductions",
        SimpleNamespace(
            rebuild_npu_tensor=lambda *_args: source.view(torch.uint8).view(-1).clone()
        ),
    )
    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(current_device_index=lambda: 0),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: SimpleNamespace(
                synchronize=lambda: synchronizations.append("sync")
            )
        ),
    )
    payload = {
        "tensor_parallel_size": 1,
        "worker_topology": [
            {"physical_device_id": "host-4", "dp_rank": 0, "tp_rank": 0}
        ],
        "buckets_by_target": {
            0: [
                {
                    "ipc_handles": {"host-4": [None] * 7},
                    "metadata": _bucket_metadata()[0],
                }
            ]
        },
    }
    encoded = base64.b64encode(pickle.dumps(payload)).decode("ascii")
    worker = _receive_worker()

    result = receive_ipc_direct_reshard(
        worker,
        payload_pickled=encoded,
        policy_version=2,
    )

    torch.testing.assert_close(worker.model_runner.get_model().weight, source)
    assert result == {
        "received": True,
        "dp_rank": 0,
        "tp_rank": 0,
        "physical_device_id": "host-4",
        "bytes": 16,
        "bucket_count": 1,
    }
    assert worker._hyper_pending_policy_version == 2
    assert synchronizations == ["sync"]


@pytest.mark.parametrize("load_failure", [False, True])
def test_worker_receives_ipc_packed_weights_through_model_loader(
    monkeypatch: pytest.MonkeyPatch,
    load_failure: bool,
) -> None:
    """Packed IPC synchronizes imported storage even when the loader fails."""
    source = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    events = []
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": 0,
            "tp_size": 1,
            "physical_device_id": "host-4",
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        SimpleNamespace(npu_generate_uuid=lambda: "host-4"),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu.multiprocessing.reductions",
        SimpleNamespace(
            rebuild_npu_tensor=lambda *_args: source.view(torch.uint8).view(-1).clone()
        ),
    )
    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(current_device_index=lambda: 0),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append("sync"))
        ),
    )
    worker = _receive_worker()
    payload = {
        "ipc_handles": {"host-4": [None] * 7},
        "metadata": _packed_metadata(),
        "total_bytes": 16,
        "worker_topology": [{
            "physical_device_id": "host-4",
            "dp_rank": 0,
            "tp_rank": 0,
        }],
    }

    if load_failure:
        def fail_load(*_args: Any, **_kwargs: Any) -> None:
            events.append("load-error")
            raise RuntimeError("loader failed")

        monkeypatch.setattr(worker.model_runner.get_model(), "load_weights", fail_load)
        with pytest.raises(RuntimeError, match="loader failed"):
            receive_ipc_packed_weights(
                worker,
                payload_pickled=base64.b64encode(pickle.dumps(payload)).decode("ascii"),
                policy_version=2,
            )
        assert events == ["load-error", "sync"]
        assert worker._hyper_pending_policy_version is None
        assert get_policy_version(worker) == {"version": 0}
        return

    result = receive_ipc_packed_weights(
        worker,
        payload_pickled=base64.b64encode(pickle.dumps(payload)).decode("ascii"),
        policy_version=2,
    )

    model = worker.model_runner.get_model()
    torch.testing.assert_close(model.weight, source)
    assert model.load_modes == [False]
    assert result["bytes"] == 16
    assert worker._hyper_pending_policy_version == 2


def test_worker_initializes_direct_group_for_target_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker group setup assigns the expected direct receiver rank."""
    group_calls = []

    class Engine:
        @staticmethod
        def _stateless_init_process_group(*args: Any, **kwargs: Any) -> Any:
            group_calls.append((args, kwargs))
            return f"group-{args[2]}"

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        SimpleNamespace(HCCLWeightTransferEngine=Engine),
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 1,
            "dp_size": 2,
            "tp_rank": 1,
            "tp_size": 2,
            "physical_device_id": "host-3",
        },
    )
    monkeypatch.setattr(
        worker_module.torch,
        'get_device_module',
        lambda _device_type=None: SimpleNamespace(current_device=lambda: 3),
    )
    monkeypatch.setattr(worker_module.torch.accelerator, 'current_accelerator', lambda: torch.device("npu"))
    transfer_engine = SimpleNamespace(
        model_update_group=None,
        _stateless_init_process_group=Engine._stateless_init_process_group,
    )
    worker = SimpleNamespace(
        weight_transfer_engine=transfer_engine,
        _check_weight_transfer_engine=lambda: None,
    )

    direct = init_direct_reshard_group(
        worker,
        group_id="direct-tp1",
        target_tp_rank=1,
        master_address="127.0.0.1",
        master_port=9000,
        world_size=3,
        expected_data_parallel_size=2,
        expected_tensor_parallel_size=2,
    )
    assert direct == {
        "joined": True,
        "dp_rank": 1,
        "tp_rank": 1,
        "group_rank": 2,
        "group_id": "direct-tp1",
    }
    assert worker._hyper_direct_reshard_groups == {"direct-tp1": "group-2"}
    assert [call[0][2] for call in group_calls] == [2]

    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": 1,
            "tp_size": 2,
            "physical_device_id": "host-1",
        },
    )
    skip_worker = SimpleNamespace()
    skipped_group = init_direct_reshard_group(
        skip_worker,
        group_id="direct-tp0",
        target_tp_rank=0,
        master_address="127.0.0.1",
        master_port=9000,
        world_size=2,
        expected_data_parallel_size=1,
        expected_tensor_parallel_size=2,
    )
    skipped_receive = receive_direct_reshard(
        skip_worker,
        group_id="direct-tp0",
        target_tp_rank=0,
        buckets=[],
        policy_version=1,
        expected_data_parallel_size=1,
        expected_tensor_parallel_size=2,
    )

    assert skipped_group == {
        "joined": False,
        "dp_rank": 0,
        "tp_rank": 1,
        "group_rank": None,
    }
    assert skipped_receive == {
        "received": False,
        "dp_rank": 0,
        "tp_rank": 1,
        "bytes": 0,
    }


def test_worker_initializes_one_packed_group_for_all_rollout_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed full-gather assigns a unique DP-major receiver rank."""
    calls = []

    class Engine:
        @staticmethod
        def _stateless_init_process_group(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return "packed-group"

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        SimpleNamespace(HCCLWeightTransferEngine=Engine),
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 1,
            "dp_size": 2,
            "tp_rank": 1,
            "tp_size": 2,
            "physical_device_id": "host-3",
        },
    )
    monkeypatch.setattr(
        worker_module.torch,
        'get_device_module',
        lambda _kind=None: SimpleNamespace(current_device=lambda: 3),
    )
    monkeypatch.setattr(worker_module.torch.accelerator, 'current_accelerator', lambda: torch.device("npu"))
    worker = SimpleNamespace()

    result = init_packed_weight_group(
        worker,
        group_id="packed",
        master_address="127.0.0.1",
        master_port=9000,
        world_size=5,
        expected_data_parallel_size=2,
        expected_tensor_parallel_size=2,
    )

    assert result["group_rank"] == 4
    assert worker._hyper_packed_weight_groups == {"packed": "packed-group"}
    assert calls[0][0][2:4] == (4, 5)


def test_worker_hook_installation_runs_custom_update_and_wake_lifecycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed vLLM hooks commit custom weights and keep the scheduler paused."""
    original_calls = []

    class WorkerBase:
        pass

    class NPUWorker:
        """Record the original worker hooks before installing weight-update adapters."""
        @staticmethod
        def start_weight_update(worker: Any, is_checkpoint_format: bool = True) -> None:
            original_calls.append(("start", worker, is_checkpoint_format))

        @staticmethod
        def update_weights(worker: Any, update_info: dict[str, Any]) -> None:
            original_calls.append(("update", worker, update_info))

        @staticmethod
        def finish_weight_update(worker: Any) -> None:
            original_calls.append(("finish", worker))

    class EngineCore:
        @staticmethod
        def wake_up(engine_core: Any, tags: Any = None) -> None:
            original_calls.append(("wake", engine_core, tags))

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.worker_base",
        SimpleNamespace(WorkerBase=WorkerBase),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.worker",
        SimpleNamespace(NPUWorker=NPUWorker),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.core",
        SimpleNamespace(EngineCore=EngineCore),
    )
    monkeypatch.setattr(worker_module, "_patch_state", worker_module._PatchState())

    install_vllm_weight_sync_hooks(private_lifecycle=True)

    assert getattr(WorkerBase, "receive_direct_reshard") is receive_direct_reshard
    assert getattr(WorkerBase, "receive_packed_weights") is receive_packed_weights
    assert (
        getattr(WorkerBase, "receive_ipc_packed_weights")
        is receive_ipc_packed_weights
    )
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["HyperQwen3ForCausalLM"])
        ),
        _hyper_loaded_policy_version=0,
        _hyper_pending_policy_version=None,
        _weight_update_active=False,
        _is_checkpoint_format=True,
        _check_weight_transfer_engine=lambda: None,
        _check_nz_disabled=lambda: None,
    )
    NPUWorker.start_weight_update(worker)
    worker._hyper_pending_policy_version = 1
    NPUWorker.finish_weight_update(worker)

    awakened = []
    engine_core = SimpleNamespace(
        model_executor=SimpleNamespace(
            wake_up=lambda tags: awakened.extend(tags),
        )
    )
    EngineCore.wake_up(engine_core, ["weights", worker_module.KEEP_SCHEDULER_PAUSED_TAG])

    native_worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["OtherForCausalLM"])
        ),
        _hyper_loaded_policy_version=0,
        _hyper_pending_policy_version=None,
    )
    NPUWorker.start_weight_update(native_worker, is_checkpoint_format=True)
    native_worker._hyper_pending_policy_version = 2
    NPUWorker.update_weights(
        native_worker,
        {"name": "native"},
    )
    NPUWorker.finish_weight_update(native_worker)
    EngineCore.wake_up(engine_core, ["kv_cache"])

    assert worker._hyper_loaded_policy_version == 1
    assert worker._hyper_pending_policy_version is None
    assert worker._weight_update_active is False
    assert original_calls == [
        ("start", native_worker, True),
        ("update", native_worker, {"name": "native"}),
        ("finish", native_worker),
        ("wake", engine_core, ["kv_cache"]),
    ]
    assert awakened == ["weights"]
    assert native_worker._hyper_loaded_policy_version == 2
    assert native_worker._hyper_pending_policy_version is None


def test_finish_weight_update_commits_pending_version() -> None:
    """A completed worker update commits its pending integer version."""
    worker = SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=1,
        _hyper_loaded_policy_version=0,
        _check_weight_transfer_engine=lambda: None,
    )
    worker_module._finish_custom_weight_update(worker)

    assert worker._hyper_loaded_policy_version == 1
    assert worker._hyper_pending_policy_version is None


def test_worker_reports_native_qwen3_fused_qkv_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native Qwen3 decomposes fused QKV storage into canonical projection layouts."""
    model = SimpleNamespace(
        named_parameters=lambda: (
            (
                "model.layers.0.self_attn.qkv_proj.weight",
                torch.empty(10, 6),
            ),
            ("model.norm.weight", torch.empty(6)),
        )
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: model),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                num_attention_heads=3,
                num_key_value_heads=1,
                head_dim=2,
                hidden_size=6,
                vocab_size=8,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(
            get_tp_group=lambda: SimpleNamespace(rank_in_group=0, world_size=1)
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {
            "dp_rank": 0,
            "dp_size": 1,
            "tp_rank": 0,
            "tp_size": 1,
            "physical_device_id": "host-0",
        },
    )

    layout = get_direct_reshard_layout(worker)

    names = {item["name"] for item in layout["tensors"]}
    assert names == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.norm.weight",
    }
