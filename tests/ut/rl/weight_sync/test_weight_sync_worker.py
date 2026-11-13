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
import rl.weight_sync.vllm_worker as worker_module
import torch
from rl.weight_sync.vllm_worker import (
    _native_weight_layout,
    get_policy_version,
    init_packed_weight_group,
    install_vllm_weight_sync_hooks,
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
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
        ),
        parallel_config=SimpleNamespace(data_parallel_index=0, data_parallel_size=1),
        _rl_loaded_policy_version=1,
        _rl_pending_policy_version=None,
    )




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
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
        ),
        _weight_update_active=True,
        _check_weight_transfer_engine=lambda: None,
        _rl_loaded_policy_version=0,
        _rl_pending_policy_version=None,
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
    worker._rl_packed_weight_groups = {"packed": Group()}

    result = receive_packed_weights(
        worker,
        group_id="packed",
        metadata=_packed_metadata(),
        total_bytes=16,
        policy_version=1,
    )

    model = worker.model_runner.get_model()
    torch.testing.assert_close(model.weight, source)
    assert model.load_modes == [True]
    assert result["bytes"] == 16
    assert worker._rl_pending_policy_version == 1




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
        assert worker._rl_pending_policy_version is None
        assert get_policy_version(worker) == {"version": 0}
        return

    result = receive_ipc_packed_weights(
        worker,
        payload_pickled=base64.b64encode(pickle.dumps(payload)).decode("ascii"),
        policy_version=2,
    )

    model = worker.model_runner.get_model()
    torch.testing.assert_close(model.weight, source)
    assert model.load_modes == [True]
    assert result["bytes"] == 16
    assert worker._rl_pending_policy_version == 2




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
    assert worker._rl_packed_weight_groups == {"packed": "packed-group"}
    assert calls[0][0][2:4] == (4, 5)


@pytest.mark.parametrize("engine_owned_reload", [False, True])
def test_worker_hook_installation_runs_custom_update_and_wake_lifecycles(
    monkeypatch: pytest.MonkeyPatch,
    engine_owned_reload: bool,
) -> None:
    """Installed vLLM hooks commit custom weights and keep the scheduler paused."""
    original_calls = []
    reset_calls = []

    class WorkerBase:
        pass

    class NPUWorker:
        """Record the original worker hooks before installing weight-update adapters."""
        @staticmethod
        def wake_up(worker: Any, tags: Any = None) -> None:
            original_calls.append(("worker_wake", worker, tags))

        @staticmethod
        def start_weight_update(worker: Any, is_checkpoint_format: bool = True) -> None:
            original_calls.append(("start", worker, is_checkpoint_format))

        @staticmethod
        def update_weights(worker: Any, update_info: dict[str, Any]) -> None:
            original_calls.append(("update", worker, update_info))

        @staticmethod
        def finish_weight_update(worker: Any) -> None:
            original_calls.append(("finish", worker))

    if engine_owned_reload:
        def start_engine_update(worker: Any) -> None:
            """Match the no-format-argument Ascend 0.30 worker signature."""
            original_calls.append(("start", worker))

        NPUWorker.start_weight_update = staticmethod(start_engine_update)

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
    class DPLBAsyncMPClient:
        """Keep hook installation isolated from the real vLLM engine client."""

        async def collective_rpc_async(
            self, method: str, timeout: Optional[float] = None, args: tuple = (), kwargs: Optional[dict] = None,
        ) -> list:
            """Expose the installed client method without importing vLLM."""
            del method, timeout, args, kwargs
            return []

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.core_client",
        SimpleNamespace(DPLBAsyncMPClient=DPLBAsyncMPClient),
    )
    monkeypatch.setattr(worker_module, "_patch_state", worker_module._PatchState())

    install_vllm_weight_sync_hooks(private_lifecycle=True)

    assert not hasattr(WorkerBase, "receive_direct_reshard")
    assert getattr(WorkerBase, "receive_packed_weights") is receive_packed_weights
    assert (
        getattr(WorkerBase, "receive_ipc_packed_weights")
        is receive_ipc_packed_weights
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(reset_lora_state=lambda: reset_calls.append("reset")),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
        ),
        _rl_loaded_policy_version=0,
        _rl_pending_policy_version=None,
        _weight_update_active=False,
        _is_checkpoint_format=True,
        _check_weight_transfer_engine=lambda: None,
        _check_nz_disabled=lambda: None,
    )
    NPUWorker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="already active"):
        NPUWorker.start_weight_update(worker)
    with pytest.raises(RuntimeError, match="pending policy version"):
        NPUWorker.finish_weight_update(worker)
    assert worker._weight_update_active
    assert not reset_calls
    worker._rl_pending_policy_version = 1
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
        _rl_loaded_policy_version=0,
        _rl_pending_policy_version=None,
    )
    NPUWorker.start_weight_update(native_worker, is_checkpoint_format=True)
    native_worker._rl_pending_policy_version = 2
    NPUWorker.update_weights(
        native_worker,
        {"name": "native"},
    )
    NPUWorker.finish_weight_update(native_worker)
    EngineCore.wake_up(engine_core, ["kv_cache"])

    assert worker._rl_loaded_policy_version == 1
    assert worker._rl_pending_policy_version is None
    assert worker._weight_update_active is False
    assert reset_calls == (["reset"] if engine_owned_reload else [])
    assert original_calls == [
        (("start", native_worker) if engine_owned_reload else ("start", native_worker, True)),
        ("update", native_worker, {"name": "native"}),
        ("finish", native_worker),
        ("wake", engine_core, ["kv_cache"]),
    ]
    assert awakened == ["weights"]
    assert native_worker._rl_loaded_policy_version == 2

    graph_worker = SimpleNamespace(**vars(worker))
    graph_worker.model_config = SimpleNamespace(enforce_eager=False, hf_config=SimpleNamespace(
        architectures=[worker_module.NATIVE_QWEN3_5_ARCHITECTURE]))
    graph_worker._weight_update_active = True
    graph_worker._rl_pending_policy_version = 3
    graph_worker._rl_expected_weights = {"model.norm.weight"}
    graph_worker._rl_received_weights = {"model.norm.weight"}
    NPUWorker.finish_weight_update(graph_worker)
    assert graph_worker._rl_graph_refresh_pending, (
        f"Missing pending refresh: actual={graph_worker._rl_graph_refresh_pending}, expected=True")
    refreshed = []
    monkeypatch.setattr(worker_module, "_refresh_ascend_decode_graphs",
                        lambda value: refreshed.append((value, original_calls[-1])))
    NPUWorker.wake_up(graph_worker, ["weights"])
    assert not refreshed, f"Refresh ran before KV wake: actual={refreshed}, expected=[]"
    NPUWorker.wake_up(graph_worker, ["kv_cache"])
    expected = [(graph_worker, ("worker_wake", graph_worker, ["kv_cache"]))]
    assert refreshed == expected, f"Refresh did not follow KV wake: actual={refreshed}, expected={expected}"
    assert native_worker._rl_pending_policy_version is None


def test_finish_weight_update_commits_pending_version() -> None:
    """A completed worker update commits its pending integer version."""
    worker = SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _rl_pending_policy_version=1,
        _rl_loaded_policy_version=0,
        _check_weight_transfer_engine=lambda: None,
    )
    worker_module._finish_custom_weight_update(worker)

    assert worker._rl_loaded_policy_version == 1
    assert worker._rl_pending_policy_version is None


def test_finish_weight_update_does_not_commit_when_lora_reset_fails() -> None:
    """The 0.30 runner reset must succeed before publishing the new version."""
    def fail_reset() -> None:
        raise RuntimeError("LoRA reset failed")

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(reset_lora_state=fail_reset),
        _weight_update_active=True,
        _rl_pending_policy_version=2,
        _rl_loaded_policy_version=1,
        _check_weight_transfer_engine=lambda: None,
    )
    with pytest.raises(RuntimeError, match="LoRA reset failed"):
        worker_module._finish_custom_weight_update(worker, reset_lora_state=True)
    assert worker._weight_update_active
    assert worker._rl_loaded_policy_version == 1
    assert worker._rl_pending_policy_version == 2


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

    layout = _native_weight_layout(worker)

    names = {item["name"] for item in layout["tensors"]}
    assert names == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.norm.weight",
    }
