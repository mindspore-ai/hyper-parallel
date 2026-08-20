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

from rl.roles.model import (
    HYPER_MODEL_IMPLEMENTATION,
    HYPER_QWEN3_ARCHITECTURE,
    HYPER_QWEN3_5_ARCHITECTURE,
    NATIVE_MODEL_IMPLEMENTATION,
    NATIVE_QWEN3_ARCHITECTURE,
    NATIVE_QWEN3_5_ARCHITECTURE,
    SUPPORTED_MODEL_IMPLEMENTATIONS,
    ModelRegistration,
    VLLMModelRegistration,
    architecture_for_implementation,
    normalize_model_implementation,
    resolve_vllm_model,
)
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

from hyper_parallel import get_platform
platform = get_platform()


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
def map_actor_state_dict(
    state_dict: Mapping[str, Any],
    model: VLLMModelRegistration,
) -> dict[str, Any]:
    """Map policy Actor names to the selected rollout model namespace."""
    mapped = {}
    for name, tensor in state_dict.items():
        mapped_name = model.actor_weight_name(name)
        if mapped_name is None:
            continue
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

    def transfer(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Load one policy snapshot into rollout."""
        raise NotImplementedError

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Route the historical refit verb to the canonical transfer operation."""
        self.transfer(client, snapshot)
class CPUWeightTransfer(_RefitCompatibleTransfer):
    """Stage full Actor weights on CPU and reload every rollout worker."""
    def __init__(self, model: VLLMModelRegistration) -> None:
        """Validate and store the rollout model identity."""
        self._model = model
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None
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
            self._model,
        )
        checkpoint_dir = None
        try:
            if isinstance(client, VLLMWeightSyncClientMixin):
                synchronized_call("CPU refit pause", client.pause)
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
                        "policy_version": snapshot.version,
                    },
                ),
            )
            synchronized_call(
                "weight reload commit",
                lambda: client.collective_rpc(
                    "commit_reloaded_weights",
                    kwargs={"policy_version": snapshot.version},
                ),
            )
            if isinstance(client, VLLMWeightSyncClientMixin):
                expected_fingerprint = policy_weight_fingerprint(cpu_state_dict)
                verify_policy_fingerprints(
                    expected_fingerprint,
                    client.get_policy_weight_fingerprints(),
                    expected_version=snapshot.version,
                )
                self.last_policy_fingerprint = expected_fingerprint
            def reset_prefix_cache() -> None:
                """Invalidate requests that could observe stale policy state."""
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
    def __init__(self, model: VLLMModelRegistration) -> None:
        """Validate identity and defer creation of the HCCL transfer group."""
        self._model = model
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
            self._model,
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
                server_update = executor.submit(
                    client.receive_weights,
                    update_info,
                    snapshot.version,
                )
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
                client.get_policy_weight_fingerprints(),
                expected_version=snapshot.version,
            )
            self.last_policy_fingerprint = expected_fingerprint
        finally:
            state_dict.clear()
    def close(self) -> None:
        """Release the cached transfer-group reference."""
        self._group = None
class NPUIPCWeightTransfer(_RefitCompatibleTransfer):
    """Transfer full FSDP Actor weights to colocated rollout replicas."""
    def __init__(self, model: VLLMModelRegistration) -> None:
        """Validate identity and initialize failed-transfer tensor ownership."""
        self._model = model
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
            """Gather full policy tensors and map them to rollout names."""
            full_state_dict = platform.gather_state_dict(
                local_state_dict,
                cpu_offload=False,
            )
            try:
                return self._device_state_dict(
                    map_actor_state_dict(
                        full_state_dict,
                        self._model,
                    )
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
                """Synchronize the producer stream before importing IPC helpers."""
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
                """Export one producer-side IPC handle per policy tensor."""
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
                                snapshot.version,
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
                """Compare every rollout worker against the transferred policy."""
                self._verify_fingerprints(
                    expected_fingerprint,
                    client.get_policy_weight_fingerprints(),
                    expected_version=snapshot.version,
                )
            self._run_control("IPC policy fingerprint", verify_local_fingerprint)
            self.last_policy_fingerprint = expected_fingerprint
        finally:
            state_dict.clear()
    def close(self) -> None:
        """Release tensors retained after a failed asynchronous transfer."""
        self._failed_state_dict.clear()
def build_weight_transfer(
    deployment: str,
    model: VLLMModelRegistration,
) -> WeightTransfer:
    """Build the transport selected by the rollout deployment."""
    if deployment == "colocated":
        return NPUIPCWeightTransfer(model)
    return HCCLWeightTransfer(model)


def _legacy_vllm_model(
    implementation: str,
    model_family: str,
    tie_word_embeddings: bool,
    native_uses_language_model_prefix: bool,
) -> VLLMModelRegistration:
    """Translate the retained refitter API into the canonical model contract."""
    if model_family == "qwen3":
        hf_architecture = "Qwen3ForCausalLM"
        model_type = "qwen3"
        text_model_type = "qwen3"
    elif model_family == "qwen3_5":
        hf_architecture = (
            "Qwen3_5ForConditionalGeneration"
            if native_uses_language_model_prefix
            else "Qwen3_5ForCausalLM"
        )
        model_type = "qwen3_5" if native_uses_language_model_prefix else "qwen3_5_text"
        text_model_type = "qwen3_5_text"
    else:
        raise ValueError(f"Unsupported vLLM model family: {model_family!r}")
    registration = ModelRegistration(
        name=model_family,
        hyper_model_name=model_family,
        weights_path="",
        tokenizer_path="",
        hf_architecture=hf_architecture,
        model_type=model_type,
        text_model_type=text_model_type,
        tie_word_embeddings=tie_word_embeddings,
    )
    return resolve_vllm_model(registration, implementation)


class CPUStateDictRefitter(CPUWeightTransfer):
    """Compatibility wrapper for the previous CPU refitter constructor."""

    def __init__(
        self,
        model_implementation: str = NATIVE_MODEL_IMPLEMENTATION,
        model_family: str = "qwen3_5",
        tie_word_embeddings: bool = True,
        native_uses_language_model_prefix: bool = True,
    ) -> None:
        """Translate legacy CPU refitter arguments to the model contract."""
        super().__init__(
            _legacy_vllm_model(
                model_implementation,
                model_family,
                tie_word_embeddings,
                native_uses_language_model_prefix,
            )
        )


class HCCLWeightRefitter(HCCLWeightTransfer):
    """Compatibility wrapper for the previous HCCL refitter constructor."""

    def __init__(
        self,
        model_implementation: str = NATIVE_MODEL_IMPLEMENTATION,
        model_family: str = "qwen3_5",
        tie_word_embeddings: bool = True,
        native_uses_language_model_prefix: bool = True,
    ) -> None:
        """Translate legacy HCCL refitter arguments to the model contract."""
        super().__init__(
            _legacy_vllm_model(
                model_implementation,
                model_family,
                tie_word_embeddings,
                native_uses_language_model_prefix,
            )
        )


class NPUIPCWeightRefitter(NPUIPCWeightTransfer):
    """Compatibility wrapper for the previous NPU IPC refitter constructor."""

    def __init__(
        self,
        model_implementation: str = NATIVE_MODEL_IMPLEMENTATION,
        model_family: str = "qwen3_5",
        tie_word_embeddings: bool = True,
        native_uses_language_model_prefix: bool = True,
    ) -> None:
        """Translate legacy IPC refitter arguments to the model contract."""
        super().__init__(
            _legacy_vllm_model(
                model_implementation,
                model_family,
                tie_word_embeddings,
                native_uses_language_model_prefix,
            )
        )


VLLMWeightRefitter = WeightTransfer


def map_policy_state_dict(
    state_dict: Mapping[str, Any],
    implementation: str,
    model_family: str = "qwen3_5",
    tie_word_embeddings: bool = True,
    native_uses_language_model_prefix: bool = True,
) -> dict[str, Any]:
    """Map policy names through the retained pre-refactor function signature."""
    return map_actor_state_dict(
        state_dict,
        _legacy_vllm_model(
            implementation,
            model_family,
            tie_word_embeddings,
            native_uses_language_model_prefix,
        ),
    )


__all__ = [
    "CPUWeightTransfer",
    "HCCLWeightTransfer",
    "HYPER_MODEL_IMPLEMENTATION",
    "HYPER_QWEN3_ARCHITECTURE",
    "HYPER_QWEN3_5_ARCHITECTURE",
    "NATIVE_MODEL_IMPLEMENTATION",
    "NATIVE_QWEN3_ARCHITECTURE",
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
    "map_policy_state_dict",
    "normalize_model_implementation",
    "policy_fingerprint_header",
    "policy_tensor_fingerprint",
    "policy_weight_fingerprint",
    "verify_policy_fingerprints",
]
