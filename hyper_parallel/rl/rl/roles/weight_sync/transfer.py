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
"""Actor-to-rollout weight preparation, transport, and verification."""
from concurrent.futures import ThreadPoolExecutor
import socket
import tempfile
from typing import Any, Mapping, Optional, Protocol
from hyper_parallel import get_platform
from rl.roles.weight_sync.sync import (
    POLICY_FINGERPRINT_ALGORITHM,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
    aggregate_policy_fingerprint,
    canonical_policy_weight_name,
    is_policy_fingerprint_weight,
    policy_fingerprint_header,
    policy_tensor_fingerprint,
    policy_weight_fingerprint,
    synchronized_call,
    synchronize_error,
    verify_policy_fingerprints,
)
platform = get_platform()
HYPER_MODEL_IMPLEMENTATION = "hyper"
NATIVE_MODEL_IMPLEMENTATION = "native"
SUPPORTED_MODEL_IMPLEMENTATIONS = (
    HYPER_MODEL_IMPLEMENTATION,
    NATIVE_MODEL_IMPLEMENTATION,
)
HYPER_QWEN3_5_ARCHITECTURE = "HyperQwen3_5ForCausalLM"
NATIVE_QWEN3_5_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
def _state_dict(payload: Any, *, full: bool, operation: str) -> dict[str, Any]:
    """Extract one model state dict and validate its tensor-only contract."""
    state_dict = (
        dict(payload)
        if isinstance(payload, Mapping)
        else platform.get_model_state_dict(
            payload,
            full_state_dict=full,
            cpu_offload=False,
        )
    )
    invalid = next(
        ((name, value) for name, value in state_dict.items() if not platform.is_tensor(value)),
        None,
    )
    if invalid is not None:
        name, value = invalid
        state_dict.clear()
        raise ValueError(
            f"{operation} state entry {name!r} must be a tensor, got {type(value)!r}"
        )
    return state_dict
def normalize_model_implementation(value: Any) -> str:
    """Validate one rollout-side vLLM model implementation."""
    implementation = str(value or HYPER_MODEL_IMPLEMENTATION).strip().lower()
    if implementation not in SUPPORTED_MODEL_IMPLEMENTATIONS:
        raise ValueError(
            "rollout.vllm.model_implementation must be 'hyper' or 'native', "
            f"got {value!r}"
        )
    return implementation
def architecture_for_implementation(implementation: str) -> str:
    """Return the vLLM architecture selected for rollout."""
    normalized = normalize_model_implementation(implementation)
    if normalized == HYPER_MODEL_IMPLEMENTATION:
        return HYPER_QWEN3_5_ARCHITECTURE
    return NATIVE_QWEN3_5_ARCHITECTURE
def map_actor_state_dict(
    state_dict: Mapping[str, Any],
    implementation: str,
) -> dict[str, Any]:
    """Map policy Actor names to the selected rollout model namespace."""
    normalized = normalize_model_implementation(implementation)
    mapped = {}
    for name, tensor in state_dict.items():
        mapped_name = name
        if (
            normalized == NATIVE_MODEL_IMPLEMENTATION
            and name.startswith("model.")
            and not name.startswith("model.language_model.")
        ):
            mapped_name = f"model.language_model.{name.removeprefix('model.')}"
        if mapped_name in mapped:
            raise ValueError(
                f"vLLM policy-name mapping collision: {name!r} maps to {mapped_name!r}"
            )
        mapped[mapped_name] = tensor
    return mapped
class WeightTransfer(Protocol):
    """Move one policy Actor snapshot into an existing rollout model."""
    def transfer(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Load one policy snapshot into rollout."""
class _RefitCompatibleTransfer:
    """Keep the previous refitter entrypoint without duplicating it per backend."""
    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        self.transfer(client, snapshot)
class CPUWeightTransfer(_RefitCompatibleTransfer):
    """Stage full Actor weights on CPU and reload every rollout worker."""
    def __init__(self, model_implementation: str = "hyper") -> None:
        self._model_implementation = normalize_model_implementation(model_implementation)
    @staticmethod
    def _cpu_state_dict(payload: Any) -> dict[str, Any]:
        state_dict = _state_dict(payload, full=True, operation="vLLM refit")
        try:
            return {
                name: tensor.detach().to("cpu").contiguous()
                for name, tensor in state_dict.items()
            }
        finally:
            state_dict.clear()
    _synchronize_error = staticmethod(synchronize_error)
    def transfer(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Reload a complete CPU snapshot and invalidate rollout caches."""
        cpu_state_dict = map_actor_state_dict(
            self._cpu_state_dict(snapshot.payload),
            self._model_implementation,
        )
        checkpoint_dir = None
        try:
            staging_error = None
            try:
                from safetensors.torch import save_file  # pylint: disable=C0415
                checkpoint_dir = tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
                    prefix="hyper-vllm-refit-"
                )
                save_file(cpu_state_dict, f"{checkpoint_dir.name}/model.safetensors")
            except Exception as error:  # pylint: disable=W0718
                staging_error = error
            self._synchronize_error(staging_error, "checkpoint staging")
            synchronized_call(
                "weight reload",
                lambda: client.collective_rpc(
                    "reload_weights",
                    kwargs={
                        "weights_path": checkpoint_dir.name,
                        "is_checkpoint_format": True,
                    },
                ),
            )
            def reset_prefix_cache() -> None:
                result = client.reset_prefix_cache(
                    reset_running_requests=True,
                    reset_connector=False,
                )
                if result is False:
                    raise RuntimeError(
                        "vLLM refused to reset its prefix cache after loading Actor weights"
                    )
            synchronized_call("prefix-cache reset", reset_prefix_cache)
        finally:
            if checkpoint_dir is not None:
                checkpoint_dir.cleanup()
            cpu_state_dict.clear()
def _open_port() -> int:
    """Return an unused loopback port for HCCL rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
class HCCLWeightTransfer(_RefitCompatibleTransfer):
    """Transfer full Actor weights to a disjoint rollout server over HCCL."""
    def __init__(self, model_implementation: str = "hyper") -> None:
        self._model_implementation = normalize_model_implementation(model_implementation)
        self._group = None
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
    _policy_fingerprint = staticmethod(policy_weight_fingerprint)
    _verify_fingerprints = staticmethod(verify_policy_fingerprints)
    @staticmethod
    def _device_state_dict(payload: Any) -> dict[str, Any]:
        """Extract a single-rank full state dict that remains on device."""
        if platform.get_world_size() != 1:
            raise ValueError(
                "The initial external vLLM HCCL refitter supports one training rank; "
                f"received world_size={platform.get_world_size()}"
            )
        state_dict = _state_dict(payload, full=True, operation="vLLM HCCL refit")
        try:
            cpu_entry = next(
                (name for name, tensor in state_dict.items() if str(tensor.device).startswith("cpu")),
                None,
            )
            if cpu_entry is not None:
                raise ValueError(
                    f"vLLM HCCL refit state entry {cpu_entry!r} must remain on NPU"
                )
            return {
                name: tensor.detach().contiguous()
                for name, tensor in state_dict.items()
            }
        finally:
            state_dict.clear()
    def _ensure_group(self, client: VLLMWeightSyncClientMixin) -> Any:
        """Create the shared trainer/server HCCL transfer group once."""
        if self._group is not None:
            return self._group
        inference_world_size = client.get_world_size()
        init_info = {
            "master_address": "127.0.0.1",
            "master_port": _open_port(),
            "rank_offset": 1,
            "world_size": inference_world_size + 1,
        }
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
            HCCLWeightTransferEngine,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            server_init = executor.submit(client.init_weight_transfer, init_info)
            group = HCCLWeightTransferEngine.trainer_init(init_info)
            try:
                server_init.result(timeout=180)
            except Exception:
                group = None
                raise
        self._group = group
        return self._group
    def transfer(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Pause rollout, transfer the policy, verify it, and resume."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("HCCLWeightRefitter requires an external vLLM HTTP client")
        state_dict = map_actor_state_dict(
            self._device_state_dict(snapshot.payload),
            self._model_implementation,
        )
        try:
            group = self._ensure_group(client)
            names = list(state_dict)
            dtype_names = [
                str(state_dict[name].dtype).rsplit(".", maxsplit=1)[-1]
                for name in names
            ]
            shapes = [list(state_dict[name].shape) for name in names]
            total_bytes = sum(
                state_dict[name].numel() * state_dict[name].element_size()
                for name in names
            )
            packed_buffer_size_bytes = total_bytes + 128 * 2**20
            update_info = {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": True,
                "packed_buffer_size_bytes": packed_buffer_size_bytes,
                "packed_num_buffers": 1,
            }
            client.pause()
            client.start_weight_update()
            platform.get_current_stream().synchronize()
            from vllm_ascend.distributed.weight_transfer.hccl_engine import (  # pylint: disable=C0415
                HCCLTrainerSendWeightsArgs,
                HCCLWeightTransferEngine,
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                server_update = executor.submit(client.receive_weights, update_info)
                HCCLWeightTransferEngine.trainer_send_weights(
                    iterator=iter(state_dict.items()),
                    trainer_args=HCCLTrainerSendWeightsArgs(
                        group=group,
                        packed=True,
                        packed_buffer_size_bytes=packed_buffer_size_bytes,
                        packed_num_buffers=1,
                    ),
                )
                server_update.result(timeout=600)
            client.finish_weight_update()
            expected_fingerprint = self._policy_fingerprint(state_dict)
            self._verify_fingerprints(
                expected_fingerprint,
                client.get_policy_weight_fingerprints(snapshot.version),
            )
            self.last_policy_fingerprint = expected_fingerprint
            client.resume()
        finally:
            state_dict.clear()
    def close(self) -> None:
        self._group = None
class NPUIPCWeightTransfer(_RefitCompatibleTransfer):
    """Transfer full FSDP Actor weights to colocated rollout replicas."""
    def __init__(self, model_implementation: str = "hyper") -> None:
        self._model_implementation = normalize_model_implementation(model_implementation)
        self._failed_state_dict: dict[str, Any] = {}
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
    _policy_fingerprint = staticmethod(policy_weight_fingerprint)
    _verify_fingerprints = staticmethod(verify_policy_fingerprints)
    @staticmethod
    def _local_state_dict(payload: Any) -> dict[str, Any]:
        return _state_dict(payload, full=False, operation="vLLM NPU IPC refit")
    @staticmethod
    def _device_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Move gathered policy tensors to the current NPU if needed."""
        device_handle = platform.get_device_handle(platform.device_type())
        target_device = f"{platform.device_type()}:{device_handle.current_device()}"
        device_state_dict = {}
        for name, tensor in state_dict.items():
            if str(tensor.device).startswith("cpu"):
                tensor = tensor.to(target_device)
            device_state_dict[name] = tensor.detach().contiguous()
        return device_state_dict
    @staticmethod
    def _validate_metadata(state_dict: Mapping[str, Any]) -> None:
        """Require all FSDP ranks to expose identical tensor metadata."""
        world_size = platform.get_world_size()
        if world_size <= 1:
            return
        metadata = [
            (
                name,
                str(tensor.dtype),
                tuple(tensor.shape),
                platform.get_tensor_distribution_spec(tensor),
            )
            for name, tensor in state_dict.items()
        ]
        gathered: list[Any] = [None] * world_size
        platform.all_gather_object(gathered, metadata)
        if any(rank_metadata != metadata for rank_metadata in gathered):
            raise RuntimeError("NPU IPC refit metadata differs across FSDP ranks")
    @staticmethod
    def _gather_endpoints(client: VLLMWeightSyncClientMixin) -> list[str]:
        world_size = platform.get_world_size()
        endpoints = [""] * world_size
        platform.all_gather_object(endpoints, client.base_url)
        if len(set(endpoints)) != world_size:
            raise RuntimeError(f"Colocated rollout endpoints must be unique, got {endpoints}")
        return endpoints
    @staticmethod
    def _run_control(operation: str, callback: Any) -> None:
        synchronized_call(operation, callback)
    def transfer(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Transfer Actor weights to every colocated rollout replica."""
        if not isinstance(client, VLLMWeightSyncClientMixin):
            raise ValueError("NPUIPCWeightRefitter requires an external vLLM HTTP client")
        local_state_dict = synchronized_call(
            "IPC local-state extraction",
            lambda: self._local_state_dict(snapshot.payload),
        )
        self._validate_metadata(local_state_dict)
        def stage_state_dict() -> dict[str, Any]:
            full_state_dict = platform.gather_state_dict(
                local_state_dict,
                cpu_offload=False,
            )
            try:
                return self._device_state_dict(
                    map_actor_state_dict(full_state_dict, self._model_implementation)
                )
            finally:
                full_state_dict.clear()
        try:
            state_dict = synchronized_call(
                "IPC full-state staging",
                stage_state_dict,
            )
        finally:
            local_state_dict.clear()
        try:
            endpoints = self._gather_endpoints(client)
            self._run_control("IPC weight wake", lambda: client.wake_up(("weights",)))
            self._run_control("IPC refit pause", client.pause)
            self._run_control("IPC refit start", client.start_weight_update)
            def setup_transport() -> tuple[Any, Any]:
                platform.get_current_stream().synchronize()
                from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
                    NPUIPCWeightTransferUpdateInfo,
                    npu_generate_uuid,
                )
                return NPUIPCWeightTransferUpdateInfo, npu_generate_uuid
            update_info_class, generate_uuid = synchronized_call(
                "IPC transport setup",
                setup_transport,
            )
            names = list(state_dict)
            dtype_names = [
                str(tensor.dtype).rsplit(".", maxsplit=1)[-1]
                for tensor in state_dict.values()
            ]
            shapes = [list(tensor.shape) for tensor in state_dict.values()]
            def build_local_handles() -> list[dict[Any, Any]]:
                npu_uuid = generate_uuid()
                return [
                    {npu_uuid: platform.get_tensor_ipc_rebuild_args(tensor)}
                    for tensor in state_dict.values()
                ]
            local_handles = synchronized_call(
                "IPC handle creation",
                build_local_handles,
            )
            world_size = platform.get_world_size()
            gathered_handles: list[Any] = [None] * world_size
            platform.all_gather_object(gathered_handles, local_handles)
            merged_handles = []
            for parameter_index in range(len(local_handles)):
                merged_handle = {}
                for rank_handles in gathered_handles:
                    merged_handle.update(rank_handles[parameter_index])
                merged_handles.append(merged_handle)
            update_info = synchronized_call(
                "IPC update construction",
                lambda: update_info_class(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    ipc_handles=merged_handles,
                    packed=False,
                ),
            )
            send_error = None
            if platform.get_rank() == 0:
                try:
                    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
                        requests = [
                            executor.submit(
                                client.receive_ipc_weights,
                                endpoint,
                                update_info,
                            )
                            for endpoint in endpoints
                        ]
                        for request in requests:
                            request.result()
                except Exception as error:  # pylint: disable=W0718
                    send_error = error
            try:
                synchronize_error(send_error, "IPC weight transfer")
            except Exception:
                self._failed_state_dict = state_dict
                state_dict = {}
                raise
            self._run_control(
                "IPC producer synchronization",
                platform.get_current_stream().synchronize,
            )
            self._run_control("IPC refit finish", client.finish_weight_update)
            expected_fingerprint = synchronized_call(
                "IPC Actor policy fingerprint",
                lambda: self._policy_fingerprint(state_dict),
            )
            def verify_local_fingerprint() -> None:
                self._verify_fingerprints(
                    expected_fingerprint,
                    client.get_policy_weight_fingerprints(snapshot.version),
                )
            self._run_control("IPC policy fingerprint", verify_local_fingerprint)
            self.last_policy_fingerprint = expected_fingerprint
        finally:
            state_dict.clear()
    def close(self) -> None:
        self._failed_state_dict.clear()
def build_weight_transfer(
    deployment: str,
    model_implementation: str,
) -> WeightTransfer:
    """Build the transport selected by the rollout deployment."""
    if deployment == "colocated":
        return NPUIPCWeightTransfer(model_implementation)
    return HCCLWeightTransfer(model_implementation)
# Stable aliases for callers that still use the previous refitter terminology.
CPUStateDictRefitter = CPUWeightTransfer
HCCLWeightRefitter = HCCLWeightTransfer
NPUIPCWeightRefitter = NPUIPCWeightTransfer
VLLMWeightRefitter = WeightTransfer
map_policy_state_dict = map_actor_state_dict
__all__ = [
    "CPUWeightTransfer",
    "HCCLWeightTransfer",
    "HYPER_MODEL_IMPLEMENTATION",
    "HYPER_QWEN3_5_ARCHITECTURE",
    "NATIVE_MODEL_IMPLEMENTATION",
    "NATIVE_QWEN3_5_ARCHITECTURE",
    "NPUIPCWeightTransfer",
    "POLICY_FINGERPRINT_ALGORITHM",
    "SUPPORTED_MODEL_IMPLEMENTATIONS",
    "WeightTransfer",
    "aggregate_policy_fingerprint",
    "architecture_for_implementation",
    "build_weight_transfer",
    "canonical_policy_weight_name",
    "is_policy_fingerprint_weight",
    "map_actor_state_dict",
    "normalize_model_implementation",
    "policy_fingerprint_header",
    "policy_tensor_fingerprint",
    "policy_weight_fingerprint",
    "verify_policy_fingerprints",
]
