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
from hashlib import sha256
import json
import pickle
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

import rl.roles.weight_sync.vllm_worker as worker_module
from rl.roles.weight_sync.vllm_worker import (
    abort_weight_update,
    get_all_parameter_manifest,
    get_direct_reshard_layout,
    get_policy_weight_fingerprint,
    get_policy_version,
    init_direct_reshard_group,
    install_vllm_weight_sync_hooks,
    prepare_direct_reshard_layout,
    receive_direct_reshard,
    receive_ipc_direct_reshard,
    verify_policy_weight_identity,
    write_parameter_manifest,
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


def test_worker_builds_layout_manifest_and_policy_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Worker topology, layout, full manifest, and norm fingerprint share identity."""
    tp_group = SimpleNamespace(rank_in_group=0, world_size=2)
    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(get_tp_group=lambda: tp_group))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        SimpleNamespace(npu_generate_uuid=lambda: "host-4"),
    )
    monkeypatch.setenv("HYPER_RL_ROLLOUT_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(worker_module.platform, "get_rank", lambda: 7)
    model = _Model()
    worker = _worker(model)

    layout = get_direct_reshard_layout(worker)
    fingerprint = get_policy_weight_fingerprint(worker)
    manifest = get_all_parameter_manifest(worker)
    basic_dir = tmp_path / "basic"
    written = write_parameter_manifest(
        worker,
        output_dir=str(basic_dir),
        strategy="direct_reshard",
        policy_version=1,
        rollout_replica_rank=0,
        expected_data_parallel_size=1,
        oracle_run_id="run-1",
    )
    verify = verify_policy_weight_identity(worker, 1, fingerprint)

    assert layout["dp_rank"] == 0
    assert layout["tp_rank"] == 0
    assert layout["tp_size"] == 2
    assert layout["physical_device_id"] == "host-4"
    assert [(item["name"], item["placement"], item["shard_dim"]) for item in layout["tensors"]] == [
        ("model.matrix.weight", "shard", 0),
        ("model.norm.weight", "replicate", None),
    ]
    assert fingerprint["version"] == 1
    assert fingerprint["rank"] == 7
    assert fingerprint["tensor_count"] == 1
    assert manifest["parameter_count"] == 2
    assert manifest["total_bytes"] == 24
    assert get_policy_version(worker) == {"version": 1}
    assert verify == {"version": 1, "digest": fingerprint["digest"]}
    assert written["written"]
    output = next(basic_dir.glob("direct_reshard-*.json"))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == manifest["manifest_sha256"]
    assert payload["oracle_run_id"] == "run-1"

    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    expected = dict(manifest)
    expected.update(
        {
            "oracle_run_id": "run-1",
            "policy_version": 1,
            "dp_size": 1,
            "source_manifest_sha256": "source-manifest",
        }
    )
    (expected_dir / "version1-dp0-tp0.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    oracle = dict(expected)
    oracle.update(
        {
            "strategy": "full_gather",
            "manifest_sha256": "oracle-manifest",
        }
    )
    (oracle_dir / "full_gather-version1-replica0-dp0-tp0.json").write_text(
        json.dumps(oracle), encoding="utf-8"
    )

    matched = write_parameter_manifest(
        worker,
        output_dir=str(tmp_path / "verified"),
        strategy="direct_reshard",
        policy_version=1,
        rollout_replica_rank=0,
        expected_data_parallel_size=1,
        oracle_run_id="run-1",
        oracle_dir=str(oracle_dir),
        oracle_strategy="full_gather",
        expected_dir=str(expected_dir),
    )

    assert matched["source_match"] is True
    assert matched["oracle_comparable"] is True
    assert matched["oracle_match"] is True


class _ReceiveModel(torch.nn.Module):
    """Expose one physical TP-local parameter receiving fragments."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((2, 2)))


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
        "worker_topology": [
            {"physical_device_id": "host-4", "dp_rank": 0, "tp_rank": 0}
        ],
        "buckets_by_tp": {
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
        worker_module.platform,
        "get_device_handle",
        lambda _device_type: SimpleNamespace(current_device=lambda: 3),
    )
    monkeypatch.setattr(worker_module.platform, "device_type", lambda: "npu")
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


def test_worker_aborts_active_transaction_and_restores_stable_version() -> None:
    """Abort clears pending state while preserving the last committed policy."""
    worker = SimpleNamespace(
        _hyper_loaded_policy_version=1,
        _hyper_pending_policy_version=2,
        _hyper_pending_content_fragments={2: {"fragment": {}}},
        _hyper_pending_content_tp_rank=0,
        _weight_update_active=True,
        _is_checkpoint_format=False,
    )

    result = abort_weight_update(worker, restore_policy_version=1)
    assert result == {
        "aborted": True,
        "was_active": True,
        "pending_version": 2,
        "restored_version": 1,
    }
    assert worker._hyper_loaded_policy_version == 1
    assert worker._hyper_pending_policy_version is None


def test_worker_hook_installation_runs_custom_update_and_wake_lifecycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed vLLM hooks commit custom weights and keep the scheduler paused."""
    original_calls = []

    class WorkerBase:
        pass

    class NPUWorker:
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

    assert getattr(WorkerBase, "prepare_direct_reshard_layout") is prepare_direct_reshard_layout
    assert getattr(WorkerBase, "receive_direct_reshard") is receive_direct_reshard
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
    worker._hyper_pending_content_tp_rank = None
    NPUWorker.finish_weight_update(worker)

    awakened = []
    engine_core = SimpleNamespace(
        model_executor=SimpleNamespace(
            wake_up=lambda tags: awakened.extend(tags),
            collective_rpc=lambda method: (
                [{"prepared": True}]
                if method == "prepare_direct_reshard_layout"
                else []
            ),
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


def test_deepseek_runtime_contract_reports_required_vllm_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime introspection reports absorbed MLA and FusedMoE ownership."""

    class AbsorbedMLA:
        @staticmethod
        def process_weights_after_loading(_dtype: object) -> None:
            return None

    fused_type = type("FusedMoE", (), {})
    modules = [SimpleNamespace(mla_attn=AbsorbedMLA()) for _ in range(3)]
    modules.extend(fused_type() for _ in range(2))
    for module in modules[3:]:
        module.w13_weight = object()
        module.w2_weight = object()
    ownership = {
        "outer_model": "Transformers DeepseekV3ForCausalLM",
        "absorbed_mla": "vLLM paged cache",
        "routed_experts": "vLLM FusedMoE",
    }
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            get_model=lambda: SimpleNamespace(
                modules=lambda: modules,
                hyper_component_ownership=ownership,
            )
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            hf_config=SimpleNamespace(
                num_hidden_layers=3,
                first_k_dense_replace=1,
                q_lora_rank=None,
                kv_lora_rank=8,
            ),
        ),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
    )
    monkeypatch.setattr(worker_module, "_is_deepseek_v3_worker", lambda _worker: True)
    monkeypatch.setattr(
        worker_module,
        "_is_hyper_deepseek_v3_worker",
        lambda _worker: True,
    )

    contract = worker_module.get_deepseek_v3_runtime_contract(worker)

    assert contract["absorbed_mla_layer_count"] == 3
    assert contract["fused_moe_layer_count"] == 2
    assert contract["q_lora_rank"] is None
    assert contract["kv_lora_rank"] == 8
    assert contract["component_ownership"] == ownership


def test_finish_weight_update_commits_source_content_identity() -> None:
    """A received direct fragment commits canonical content with its version."""
    worker = SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=1,
        _hyper_loaded_policy_version=0,
        _hyper_pending_content_tp_rank=0,
        _check_weight_transfer_engine=lambda: None,
    )
    target = torch.tensor([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    entry = {
        "name": "physical.weight",
        "canonical_name": "canonical.weight",
        "canonical_starts": [0, 0],
        "destination_starts": [0, 0],
        "destination_permutation": [1, 0],
        "lengths": [2, 3],
        "dtype_name": "float32",
    }

    # Source bytes and canonical metadata are independent of worker-produced state.
    key = '["canonical.weight",[0,0],[2,3],"float32"]'
    source_bytes = struct.pack("<6f", 0, 1, 2, 3, 4, 5)
    fragment_digest = sha256(key.encode() + source_bytes).hexdigest()
    expected = {
        "algorithm": "sha256-canonical-fragments-v1",
        "fragment_count": 1,
        "total_bytes": 24,
        "digest": sha256(json.dumps([key, fragment_digest, 24], separators=(",", ":")).encode()).hexdigest(),
        "fragments": {key: {"sha256": fragment_digest, "num_bytes": 24}},
    }
    worker_module._record_direct_content_fragment(worker, 1, entry, target)
    worker_module._finish_custom_weight_update(worker)
    result = worker_module.verify_direct_content_identity(
        worker,
        expected_version=1,
        expected_by_tp_rank={"0": expected},
    )

    assert result["verified"] is True
    assert worker._hyper_loaded_content_identity == expected
    assert result["total_bytes"] == target.numel() * target.element_size()
    assert worker._hyper_loaded_policy_version == 1
    assert worker._hyper_pending_policy_version is None


def test_deepseek_commit_refreshes_moe_and_absorbed_mla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek commit restores FusedMoE layout and every absorbed MLA layer."""
    refreshes = []

    def layer(index: int) -> SimpleNamespace:
        return SimpleNamespace(
            mla_attn=SimpleNamespace(
                process_weights_after_loading=lambda dtype: refreshes.append(
                    (index, dtype)
                )
            )
        )

    moe = SimpleNamespace(
        w13_weight=torch.nn.Parameter(torch.empty(2, 4, 3)),
        w2_weight=torch.nn.Parameter(torch.empty(2, 3, 2)),
    )

    def process_moe(module: SimpleNamespace) -> None:
        module.w13_weight = torch.nn.Parameter(
            module.w13_weight.data.transpose(1, 2).contiguous()
        )
        module.w2_weight = torch.nn.Parameter(
            module.w2_weight.data.transpose(1, 2).contiguous()
        )

    moe.quant_method = SimpleNamespace(process_weights_after_loading=process_moe)
    model = SimpleNamespace(modules=lambda: (layer(0), layer(1), moe))
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV3ForCausalLM"],
                num_hidden_layers=2,
                n_routed_experts=2,
                moe_intermediate_size=2,
                hidden_size=3,
            ),
        ),
        model_runner=SimpleNamespace(get_model=lambda: model),
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=1,
        _hyper_loaded_policy_version=0,
        _hyper_pending_content_tp_rank=0,
        _hyper_pending_content_fragments={
            1: {"fragment": {"sha256": "digest", "num_bytes": 4}}
        },
        _check_weight_transfer_engine=lambda: None,
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {"dp_size": 1, "tp_size": 1, "ep_size": 1},
    )

    worker_module._finish_custom_weight_update(worker)

    assert refreshes == [(0, torch.bfloat16), (1, torch.bfloat16)]
    assert tuple(moe.w13_weight.shape) == (2, 3, 4)
    assert tuple(moe.w2_weight.shape) == (2, 2, 3)
    assert worker._hyper_loaded_moe_refresh_count == 1
    assert worker._hyper_loaded_mla_refresh_count == 2


def test_qwen3_moe_commit_restores_common_physical_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3-MoE commit restores each common expert leaf before publication."""
    moe = SimpleNamespace(
        hyper_local_expert_leaf=True,
        w13_weight=torch.nn.Parameter(torch.empty(2, 4, 3)),
        w2_weight=torch.nn.Parameter(torch.empty(2, 3, 2)),
    )

    def ensure_physical_weight_layout() -> None:
        moe.w13_weight = torch.nn.Parameter(
            moe.w13_weight.data.transpose(1, 2).contiguous()
        )
        moe.w2_weight = torch.nn.Parameter(
            moe.w2_weight.data.transpose(1, 2).contiguous()
        )

    moe.ensure_physical_weight_layout = ensure_physical_weight_layout
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                architectures=["HyperQwen3MoeForCausalLM"],
                num_hidden_layers=1,
                num_experts=2,
                moe_intermediate_size=2,
                hidden_size=3,
            ),
        ),
        model_runner=SimpleNamespace(
            get_model=lambda: SimpleNamespace(modules=lambda: (moe,))
        ),
        _weight_update_active=True,
        _is_checkpoint_format=True,
        _hyper_pending_policy_version=1,
        _hyper_loaded_policy_version=0,
        _hyper_pending_content_tp_rank=0,
        _hyper_pending_content_fragments={
            1: {"fragment": {"sha256": "digest", "num_bytes": 4}}
        },
        _check_weight_transfer_engine=lambda: None,
    )
    monkeypatch.setattr(
        worker_module,
        "_rollout_worker_topology",
        lambda _worker: {"dp_size": 1, "tp_size": 1, "ep_size": 1},
    )

    worker_module._finish_custom_weight_update(worker)

    assert tuple(moe.w13_weight.shape) == (2, 3, 4)
    assert tuple(moe.w2_weight.shape) == (2, 2, 3)
    assert worker._hyper_loaded_moe_refresh_count == 1
    assert worker._hyper_loaded_mla_refresh_count == 0


@pytest.mark.parametrize(
    ("family", "ep_size", "runtime_layout"),
    [
        ("qwen3_moe", 1, False),
        ("deepseek_v3", 4, True),
    ],
)
def test_native_moe_layout_describes_tp_or_ep_owned_experts(
    family: str,
    ep_size: int,
    runtime_layout: bool,
) -> None:
    """Native MoE metadata shards intermediate storage for EP1 and experts for EP4."""
    config = SimpleNamespace(
        num_experts=8,
        n_routed_experts=8,
        hidden_size=6,
        moe_intermediate_size=12,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=2,
    )
    local_experts, local_intermediate = (2, 12) if ep_size == 4 else (8, 3)
    w13 = torch.empty(local_experts, 2 * local_intermediate, 6)
    w2 = torch.empty(local_experts, 6, local_intermediate)
    if runtime_layout:
        w13 = w13.transpose(1, 2)
        w2 = w2.transpose(1, 2)
    expert_map = (
        torch.tensor([-1, -1, -1, -1, 0, 1, -1, -1])
        if ep_size == 4
        else None
    )
    leaf = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        expert_map=expert_map,
    )
    parameters = {
        "model.layers.0.mlp.experts.w13_weight": w13,
        "model.layers.0.mlp.experts.w2_weight": w2,
        "model.layers.0.mlp.gate.weight": torch.empty(8, 6),
    }
    if family == "qwen3_moe":
        parameters["model.layers.0.self_attn.qkv_proj.weight"] = torch.empty(10, 6)
    model = SimpleNamespace(
        named_parameters=parameters.items,
        named_modules=lambda: (("model.layers.0.mlp.experts", leaf),),
    )

    descriptions = worker_module._native_moe_direct_tensors(
        model,
        config,
        1,
        4,
        family=family,
        ep_rank=2 if ep_size == 4 else 0,
        ep_size=ep_size,
    )
    config.architectures = [
        "Qwen3MoeForCausalLM"
        if family == "qwen3_moe"
        else "DeepseekV3ForCausalLM"
    ]
    manifest = worker_module._native_moe_ownership_manifest(
        SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            model_runner=SimpleNamespace(get_model=lambda: model),
        ),
        {"ep_size": ep_size, "ep_rank": 2 if ep_size == 4 else 0},
    )["model.layers.0.mlp.experts"]

    assert manifest["local_experts"] == local_experts
    assert manifest["global_to_local"] == (
        None if expert_map is None else expert_map.tolist()
    )
    expert_descriptions = [
        item for item in descriptions if ".experts." in item["name"]
    ]
    assert len(expert_descriptions) == 3
    for item in expert_descriptions:
        expected_shard_dim = (
            0
            if ep_size == 4
            else (2 if "down_proj" in item["name"] else 1)
        )
        assert item["placement"] == "shard"
        assert item["shard_dim"] == expected_shard_dim
    assert all(
        item["placement"] == "replicate"
        for item in descriptions
        if ".experts." not in item["name"]
    )


def test_worker_reports_hyper_qwen3_moe_direct_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hyper Qwen3-MoE exposes dense TP placement and EP-local expert regions."""
    parameters = {
        "model.layers.0.mlp.experts.w13_weight": torch.empty(2, 6, 4),
        "model.layers.0.mlp.experts.w2_weight": torch.empty(2, 4, 3),
        "model.layers.0.mlp.gate.weight": torch.empty(8, 4),
    }
    expert = SimpleNamespace(
        hyper_local_expert_leaf=True,
        local_expert_count=2,
    )
    model = SimpleNamespace(
        _tp_placements={
            "model.layers.0.mlp.gate.weight": (_Placement(None),),
        },
        named_parameters=parameters.items,
        named_buffers=lambda: (),
        named_modules=lambda: (("model.layers.0.mlp.experts", expert),),
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: model),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["HyperQwen3MoeForCausalLM"],
                num_experts=8,
                moe_intermediate_size=3,
                hidden_size=4,
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
            "dp_size": 4,
            "tp_rank": 0,
            "tp_size": 1,
            "ep_rank": 0,
            "ep_size": 4,
            "physical_device_id": "host-0",
        },
    )

    layout = get_direct_reshard_layout(worker)

    tensors = {item["name"]: item for item in layout["tensors"]}
    assert layout["ep_size"] == 4
    assert tensors["model.layers.0.mlp.gate.weight"]["placement"] == "replicate"
    for name in (
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
    ):
        assert tensors[name]["placement"] == "shard"
        assert tensors[name]["shard_dim"] == 0


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
