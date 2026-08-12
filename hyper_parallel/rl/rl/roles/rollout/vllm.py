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
"""Process-isolated vLLM rollout adapter and weight synchronization."""

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
import pickle
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Optional, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationRequest, GenerationResult
from rl.roles.rollout.registry import ROLLOUT_ENGINES
from rl.roles.rollout.vllm_policy import (
    aggregate_policy_fingerprint,
    architecture_for_implementation,
    is_policy_fingerprint_weight,
    map_policy_state_dict,
    normalize_model_implementation,
    policy_tensor_fingerprint,
)
from rl.roles.weight_sync import PolicySnapshot
from hyper_parallel import get_platform

platform = get_platform()

_DISTRIBUTED_ENVIRONMENT_VARIABLES = (
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
    "RANK_ID",
    "RANK_SIZE",
    "DEVICE_ID",
    "RANK_TABLE_FILE",
    "VLLM_DP_RANK",
    "VLLM_DP_RANK_LOCAL",
    "VLLM_DP_SIZE",
    "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT",
)


def _open_port() -> int:
    """Return an unused loopback TCP port for a local service."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _training_physical_device() -> str:
    """Resolve this trainer rank's physical NPU from its visible-device map."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible_devices is None:
        return str(local_rank)
    device_ids = [device.strip() for device in visible_devices.split(",")]
    if local_rank >= len(device_ids) or not device_ids[local_rank]:
        raise ValueError(
            f"LOCAL_RANK={local_rank} cannot be mapped through "
            f"ASCEND_RT_VISIBLE_DEVICES={visible_devices!r}"
        )
    return device_ids[local_rank]


def _synchronize_error(local_error: Optional[Exception], operation: str) -> None:
    """Make every training rank observe one local rollout-phase failure."""
    try:
        world_size = platform.get_world_size()
    except (RuntimeError, ValueError):
        world_size = 1
    if world_size <= 1:
        if local_error is not None:
            raise local_error
        return

    errors: list[Optional[str]] = [None] * world_size
    platform.all_gather_object(
        errors,
        None if local_error is None else str(local_error),
    )
    if any(error is not None for error in errors):
        raise RuntimeError(f"vLLM {operation} failed on at least one rank: {errors}")


def _policy_weight_fingerprint(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Hash replicated Qwen3.5 norm weights without copying the full policy."""
    tensor_digests = {}
    value_count = 0
    for name, tensor in sorted(state_dict.items(), key=lambda item: item[0]):
        if not is_policy_fingerprint_weight(name):
            continue
        values = platform.tensor_type_cast(
            tensor.detach().to("cpu").contiguous(),
            "float32",
        )
        canonical_name, tensor_digest = policy_tensor_fingerprint(
            name,
            tuple(values.shape),
            platform.tensor_to_numpy(values).tobytes(),
        )
        if canonical_name in tensor_digests:
            raise RuntimeError(f"Actor policy fingerprint has duplicate tensor {canonical_name!r}")
        tensor_digests[canonical_name] = tensor_digest
        value_count += int(values.numel())
    if not tensor_digests:
        raise RuntimeError("Actor policy fingerprint found no language-model norm tensors")
    return aggregate_policy_fingerprint(tensor_digests, value_count)


def _verify_policy_fingerprints(
    expected: Mapping[str, Any],
    actual: list[Mapping[str, Any]],
) -> None:
    """Require every vLLM worker fingerprint to match the Actor policy."""
    if not actual:
        raise RuntimeError("vLLM policy fingerprint returned no worker results")
    fields = ("algorithm", "tensor_count", "value_count", "digest")
    mismatches = [
        {field: result.get(field) for field in fields}
        for result in actual
        if any(result.get(field) != expected.get(field) for field in fields)
    ]
    if mismatches:
        expected_tensors = expected.get("tensors", {})
        tensor_mismatches = []
        for result in actual:
            actual_tensors = result.get("tensors", {})
            if actual_tensors != expected_tensors:
                tensor_mismatches.append(
                    {
                        "rank": result.get("rank"),
                        "missing": sorted(set(expected_tensors) - set(actual_tensors)),
                        "unexpected": sorted(set(actual_tensors) - set(expected_tensors)),
                        "changed": sorted(
                            name
                            for name in set(expected_tensors) & set(actual_tensors)
                            if expected_tensors[name] != actual_tensors[name]
                        ),
                    }
                )
        expected_summary = {field: expected.get(field) for field in fields}
        raise RuntimeError(
            "vLLM policy fingerprint mismatch: "
            f"expected={expected_summary}, "
            f"actual={mismatches}, tensor_mismatches={tensor_mismatches}"
        )


class _VLLMHTTPClient:
    """Synchronous token and RL control client for one vLLM server process."""

    def __init__(
        self,
        process: subprocess.Popen,
        base_url: str,
        model_name: str,
        request_timeout: float,
    ) -> None:
        """Bind one owned server process to its loopback endpoint."""
        self._process = process
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._request_timeout = request_timeout

    def _request(
        self,
        method: str,
        route: str,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if data is None else {"Content-Type": "application/json"}
        request = urllib_request.Request(
            f"{(base_url or self._base_url).rstrip('/')}/{route.lstrip('/')}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib_request.urlopen(  # nosec B310 - URL is fixed to the configured local server.
                request,
                timeout=self._request_timeout if timeout is None else timeout,
            ) as response:
                body = response.read()
        except urllib_error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP {method} {route} failed with status {error.code}: {body}"
            ) from error
        except (urllib_error.URLError, TimeoutError) as error:
            raise RuntimeError(f"vLLM HTTP {method} {route} failed: {error}") from error
        if not body:
            return {}
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            return {}
        if not isinstance(result, dict):
            raise RuntimeError(f"vLLM HTTP {method} {route} returned a non-object response")
        return result

    @property
    def base_url(self) -> str:
        """Return this rank-local server endpoint."""
        return self._base_url

    def wait_ready(self, startup_timeout: float) -> None:
        """Wait until the server health endpoint responds or the process exits."""
        deadline = time.monotonic() + startup_timeout
        last_error = None
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(f"vLLM server exited during startup with code {return_code}")
            try:
                self._request("GET", "health", timeout=2)
                return
            except RuntimeError as error:
                last_error = error
                time.sleep(1)
        raise RuntimeError(
            f"vLLM server did not become ready within {startup_timeout} seconds: {last_error}"
        )

    def _generate_completion_batch(
        self,
        prompts: list[list[int]],
        seed: Optional[int],
        settings: Any,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Execute and validate one OpenAI completion request."""
        payload = {
            "model": self._model_name,
            "prompt": prompts,
            "max_tokens": settings.max_new_tokens,
            "temperature": settings.temperature if settings.do_sample else 0.0,
            "top_p": settings.top_p,
            "top_k": settings.top_k if settings.top_k > 0 else -1,
            "n": 1,
            "logprobs": 1 if settings.collect_log_probs else None,
            "return_token_ids": True,
            "add_special_tokens": False,
        }
        if seed is not None:
            payload["seed"] = seed
        result = self._request("POST", "v1/completions", payload)
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != len(prompts):
            raise RuntimeError(
                "vLLM completion response count mismatch: "
                f"expected={len(prompts)}, received={len(choices) if isinstance(choices, list) else None}"
            )
        records = []
        for choice in sorted(choices, key=lambda item: int(item["index"])):
            token_ids = choice.get("token_ids")
            if not isinstance(token_ids, list):
                raise RuntimeError("vLLM completion response did not include token_ids")
            token_log_probs = None
            if settings.collect_log_probs:
                log_probs = choice.get("logprobs")
                if not isinstance(log_probs, dict) or not isinstance(log_probs.get("token_logprobs"), list):
                    raise RuntimeError("vLLM completion response did not include token_logprobs")
                raw_log_probs = log_probs["token_logprobs"]
                if len(raw_log_probs) != len(token_ids) or any(value is None for value in raw_log_probs):
                    raise RuntimeError(
                        "vLLM completion returned incomplete sampled-token log probabilities: "
                        f"tokens={len(token_ids)}, log_probs={len(raw_log_probs)}"
                    )
                token_log_probs = [float(value) for value in raw_log_probs]
            records.append(([int(token_id) for token_id in token_ids], token_log_probs))
        return records

    def generate_tokens(
        self,
        prompt_token_ids: list[list[int]],
        settings: Any,
        request_concurrency: int = 1,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Generate ordered token records with bounded seeded-request concurrency."""
        if request_concurrency <= 0:
            raise ValueError(f"vLLM request_concurrency must be positive, got {request_concurrency}")
        request_batches = (
            [(prompt_token_ids, None)]
            if settings.seed is None
            else [([prompt_ids], settings.seed + row) for row, prompt_ids in enumerate(prompt_token_ids)]
        )
        if len(request_batches) == 1 or request_concurrency == 1:
            batches = [
                self._generate_completion_batch(prompts, seed, settings)
                for prompts, seed in request_batches
            ]
        else:
            max_workers = min(request_concurrency, len(request_batches))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batches = list(
                    executor.map(
                        lambda item: self._generate_completion_batch(item[0], item[1], settings),
                        request_batches,
                    )
                )
        return [record for batch in batches for record in batch]

    def get_world_size(self) -> int:
        """Return the number of vLLM inference workers."""
        return int(self._request("GET", "get_world_size")["world_size"])

    def init_weight_transfer(self, init_info: Mapping[str, Any]) -> None:
        """Initialize the server side of the stateless HCCL transfer group."""
        self._request("POST", "init_weight_transfer_engine", {"init_info": dict(init_info)}, timeout=180)

    def pause(self) -> None:
        """Pause generation and invalidate request caches before refit."""
        self._request("POST", "pause?mode=abort&clear_cache=true")

    def sleep(self, level: int = 1, mode: str = "wait") -> None:
        """Drain generation and release tagged vLLM device memory."""
        self._request("POST", f"sleep?level={level}&mode={mode}")

    def wake_up(self, tags: tuple[str, ...]) -> None:
        """Restore selected vLLM memory tags without admitting new requests."""
        query = urllib_parse.urlencode([("tags", tag) for tag in tags])
        self._request("POST", f"wake_up?{query}")

    def is_sleeping(self) -> bool:
        """Return the server's combined scheduler/device sleep state."""
        value = self._request("GET", "is_sleeping").get("is_sleeping")
        if not isinstance(value, bool):
            raise RuntimeError("vLLM /is_sleeping did not return a boolean state")
        return value

    def start_weight_update(self) -> None:
        """Start Hyper checkpoint loading without direct parameter-name copies."""
        self._request("POST", "start_weight_update", {"is_checkpoint_format": True})

    def receive_weights(self, update_info: Mapping[str, Any]) -> None:
        """Block until the server receives and loads all HCCL weight buffers."""
        self._request("POST", "update_weights", {"update_info": dict(update_info)}, timeout=600)

    def receive_ipc_weights(self, base_url: str, update_info: Any) -> None:
        """Send one merged NPU IPC handle set to a rollout DP replica."""
        update_fields = asdict(update_info)
        ipc_handles = update_fields.pop("ipc_handles")
        update_fields["ipc_handles_pickled"] = base64.b64encode(
            pickle.dumps(ipc_handles)
        ).decode("ascii")
        self._request(
            "POST",
            "update_weights",
            {"update_info": update_fields},
            timeout=600,
            base_url=base_url,
        )

    def finish_weight_update(self) -> None:
        """Commit one completed Hyper weight update."""
        self._request("POST", "finish_weight_update")

    def get_policy_weight_fingerprints(
        self,
        version: int,
        base_url: Optional[str] = None,
    ) -> list[Mapping[str, Any]]:
        """Return one lightweight post-refit fingerprint per vLLM worker."""
        response = self._request(
            "POST",
            "collective_rpc",
            {
                "method": "get_policy_weight_fingerprint",
                "kwargs": {"version": str(version)},
            },
            base_url=base_url,
        )
        results = response.get("results")
        if not isinstance(results, list) or not all(isinstance(result, Mapping) for result in results):
            raise RuntimeError("vLLM policy fingerprint RPC returned invalid worker results")
        return results

    def resume(self) -> None:
        """Resume generation after refit."""
        self._request("POST", "resume")

    def close(self) -> None:
        """Terminate the server and all EngineCore descendants."""
        parent_running = self._process.poll() is None
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if not parent_running:
            # The process-group leader can exit before its EngineCore descendants.
            time.sleep(1)
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        try:
            self._process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait(timeout=10)
        else:
            # The API server leader may exit before EngineCore descendants.
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class VLLMWeightRefitter(Protocol):
    """Explicit version-sensitive bridge implemented by a concrete deployment."""

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Load one policy snapshot into an existing vLLM client."""


class CPUStateDictRefitter:
    """Gather Actor weights to CPU and synchronously reload every vLLM worker.

    This retained correctness gate gathers complete state, stages a temporary
    safetensors checkpoint, and reloads an injected in-process vLLM client. Online
    GRPO uses the process-isolated HCCL refitter below.
    """

    def __init__(self, model_implementation: str = "hyper") -> None:
        """Initialize the checkpoint namespace selected by the vLLM model."""
        self._model_implementation = normalize_model_implementation(model_implementation)

    @staticmethod
    def _cpu_state_dict(payload: Any) -> dict[str, Any]:
        """Return a detached full CPU state dict from an Actor or mapping."""
        if isinstance(payload, Mapping):
            state_dict = dict(payload)
        else:
            module = getattr(payload, "module", payload)
            state_dict = platform.get_model_state_dict(
                module,
                full_state_dict=True,
                cpu_offload=False,
            )

        cpu_state_dict = {}
        for name, tensor in state_dict.items():
            if not platform.is_tensor(tensor):
                raise ValueError(
                    f"vLLM refit state entry {name!r} must be a tensor, got {type(tensor)!r}"
                )
            cpu_state_dict[name] = tensor.detach().to("cpu").contiguous()
        state_dict.clear()
        return cpu_state_dict

    @staticmethod
    def _synchronize_error(local_error: Optional[Exception], operation: str) -> None:
        """Make every training rank observe a refit phase failure."""
        _synchronize_error(local_error, operation)

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Reload one complete CPU policy snapshot and invalidate cached prefixes."""
        cpu_state_dict = map_policy_state_dict(
            self._cpu_state_dict(snapshot.payload),
            self._model_implementation,
        )
        checkpoint_dir = None
        try:
            staging_error = None
            try:
                # safetensors is installed with the optional vLLM backend.
                from safetensors.torch import save_file  # pylint: disable=C0415

                checkpoint_dir = tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
                    prefix="hyper-vllm-refit-"
                )
                save_file(cpu_state_dict, f"{checkpoint_dir.name}/model.safetensors")
            except Exception as error:  # pylint: disable=W0718
                staging_error = error
            self._synchronize_error(staging_error, "checkpoint staging")

            reload_error = None
            try:
                client.collective_rpc(
                    "reload_weights",
                    kwargs={
                        "weights_path": checkpoint_dir.name,
                        "is_checkpoint_format": True,
                    },
                )
            except Exception as error:  # pylint: disable=W0718
                reload_error = error
            self._synchronize_error(reload_error, "weight reload")

            reset_error = None
            try:
                reset_result = client.reset_prefix_cache(
                    reset_running_requests=True,
                    reset_connector=False,
                )
                if reset_result is False:
                    raise RuntimeError(
                        "vLLM refused to reset its prefix cache after loading Actor weights"
                    )
            except Exception as error:  # pylint: disable=W0718
                reset_error = error
            self._synchronize_error(reset_error, "prefix-cache reset")
        finally:
            if checkpoint_dir is not None:
                checkpoint_dir.cleanup()
            cpu_state_dict.clear()


class HCCLWeightRefitter:
    """Publish full Actor weights to a disjoint vLLM server over native HCCL."""

    def __init__(self, model_implementation: str = "hyper") -> None:
        """Initialize without a trainer-side transfer group."""
        self._model_implementation = normalize_model_implementation(model_implementation)
        self._group = None
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None

    @staticmethod
    def _device_state_dict(payload: Any) -> dict[str, Any]:
        """Gather a complete contiguous device state dict for HCCL broadcast."""
        if platform.get_world_size() != 1:
            raise ValueError(
                "The initial external vLLM HCCL refitter supports one training rank; "
                f"received world_size={platform.get_world_size()}"
            )
        if isinstance(payload, Mapping):
            state_dict = dict(payload)
        else:
            module = getattr(payload, "module", payload)
            state_dict = platform.get_model_state_dict(
                module,
                full_state_dict=True,
                cpu_offload=False,
            )

        device_state_dict = {}
        for name, tensor in state_dict.items():
            if not platform.is_tensor(tensor):
                raise ValueError(
                    f"vLLM HCCL refit state entry {name!r} must be a tensor, got {type(tensor)!r}"
                )
            if str(tensor.device).startswith("cpu"):
                raise ValueError(f"vLLM HCCL refit state entry {name!r} must remain on NPU")
            device_state_dict[name] = tensor.detach().contiguous()
        state_dict.clear()
        return device_state_dict

    def _ensure_group(self, client: _VLLMHTTPClient) -> Any:
        """Create the independent trainer-to-inference HCCL communicator once."""
        if self._group is not None:
            return self._group
        inference_world_size = client.get_world_size()
        master_address = "127.0.0.1"
        master_port = _open_port()
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": inference_world_size + 1,
        }
        # vLLM-Ascend is optional and imported only by the selected HCCL refitter.
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

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Pause generation, publish one policy, clear caches, and resume."""
        if not isinstance(client, _VLLMHTTPClient):
            raise ValueError("HCCLWeightRefitter requires an external vLLM HTTP client")
        state_dict = map_policy_state_dict(
            self._device_state_dict(snapshot.payload),
            self._model_implementation,
        )
        try:
            group = self._ensure_group(client)
            names = list(state_dict)
            dtype_names = [str(state_dict[name].dtype).rsplit(".", maxsplit=1)[-1] for name in names]
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
                # One complete buffer keeps strict Hyper load_weights validation atomic.
                "packed_buffer_size_bytes": packed_buffer_size_bytes,
                "packed_num_buffers": 1,
            }
            client.pause()
            client.start_weight_update()

            # The packed producer uses private NPU streams; make optimizer writes
            # visible before those streams read the updated parameter storage.
            platform.get_current_stream().synchronize()

            # vLLM-Ascend is optional and imported only by the selected HCCL refitter.
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
            expected_fingerprint = _policy_weight_fingerprint(state_dict)
            _verify_policy_fingerprints(
                expected_fingerprint,
                client.get_policy_weight_fingerprints(snapshot.version),
            )
            self.last_policy_fingerprint = expected_fingerprint
            client.resume()
        finally:
            state_dict.clear()

    def close(self) -> None:
        """Release the trainer-side stateless HCCL communicator."""
        self._group = None


class NPUIPCWeightRefitter:
    """Publish full FSDP Actor weights to colocated TP1 rollout replicas."""

    def __init__(self, model_implementation: str = "hyper") -> None:
        """Initialize storage retained after an indeterminate HTTP transfer."""
        self._model_implementation = normalize_model_implementation(model_implementation)
        self._failed_state_dict: dict[str, Any] = {}
        self.last_policy_fingerprint: Optional[dict[str, Any]] = None

    @staticmethod
    def _local_state_dict(payload: Any) -> dict[str, Any]:
        """Extract one local state dictionary without entering tensor collectives."""
        if isinstance(payload, Mapping):
            state_dict = dict(payload)
        else:
            module = getattr(payload, "module", payload)
            state_dict = platform.get_model_state_dict(
                module,
                full_state_dict=False,
                cpu_offload=False,
            )
        for name, tensor in state_dict.items():
            if not platform.is_tensor(tensor):
                raise ValueError(
                    f"vLLM NPU IPC refit state entry {name!r} must be a tensor, got {type(tensor)!r}"
                )
        return state_dict

    @staticmethod
    def _device_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Stage one complete state dictionary on the current trainer NPU."""
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
        """Require identical names, shapes, dtypes, and order on every rank."""
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
    def _gather_endpoints(client: _VLLMHTTPClient) -> list[str]:
        """Collect one rank-local rollout endpoint per training rank."""
        world_size = platform.get_world_size()
        endpoints = [""] * world_size
        platform.all_gather_object(endpoints, client.base_url)
        if len(set(endpoints)) != world_size:
            raise RuntimeError(f"Colocated rollout endpoints must be unique, got {endpoints}")
        return endpoints

    @staticmethod
    def _run_control(operation: str, callback: Any) -> None:
        """Run one local control operation and synchronize its result."""
        local_error = None
        try:
            callback()
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        _synchronize_error(local_error, operation)

    def refit(self, client: Any, snapshot: PolicySnapshot) -> None:
        """Wake weights, update every colocated replica, and keep KV asleep."""
        if not isinstance(client, _VLLMHTTPClient):
            raise ValueError("NPUIPCWeightRefitter requires an external vLLM HTTP client")
        local_state_dict = {}
        extraction_error = None
        try:
            local_state_dict = self._local_state_dict(snapshot.payload)
        except Exception as error:  # pylint: disable=W0718
            extraction_error = error
        _synchronize_error(extraction_error, "IPC local-state extraction")
        self._validate_metadata(local_state_dict)

        state_dict = {}
        staging_error = None
        try:
            full_state_dict = platform.gather_state_dict(local_state_dict, cpu_offload=False)
            mapped_state_dict = map_policy_state_dict(
                full_state_dict,
                self._model_implementation,
            )
            state_dict = self._device_state_dict(mapped_state_dict)
            full_state_dict.clear()
        except Exception as error:  # pylint: disable=W0718
            staging_error = error
        finally:
            local_state_dict.clear()
        _synchronize_error(staging_error, "IPC full-state staging")
        try:
            endpoints = self._gather_endpoints(client)
            self._run_control("IPC weight wake", lambda: client.wake_up(("weights",)))
            # wake_up resumes scheduling; the external admission boundary remains
            # closed and this pause keeps EngineCore quiescent during refit.
            self._run_control("IPC refit pause", client.pause)
            self._run_control("IPC refit start", client.start_weight_update)
            update_info_class = None
            generate_uuid = None
            setup_error = None
            try:
                platform.get_current_stream().synchronize()
                # vLLM-Ascend is optional and loaded only by the selected IPC refitter.
                from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (  # pylint: disable=C0415
                    NPUIPCWeightTransferUpdateInfo,
                    npu_generate_uuid,
                )

                update_info_class = NPUIPCWeightTransferUpdateInfo
                generate_uuid = npu_generate_uuid
            except Exception as error:  # pylint: disable=W0718
                setup_error = error
            _synchronize_error(setup_error, "IPC transport setup")
            if update_info_class is None or generate_uuid is None:
                raise RuntimeError("NPU IPC transport setup failed without a synchronized error")

            names = list(state_dict)
            dtype_names = [str(tensor.dtype).split(".")[-1] for tensor in state_dict.values()]
            shapes = [list(tensor.shape) for tensor in state_dict.values()]
            local_handles = []
            handle_error = None
            try:
                npu_uuid = generate_uuid()
                local_handles = [
                    {npu_uuid: platform.get_tensor_ipc_rebuild_args(tensor)}
                    for tensor in state_dict.values()
                ]
            except Exception as error:  # pylint: disable=W0718
                handle_error = error
            _synchronize_error(handle_error, "IPC handle creation")

            world_size = platform.get_world_size()
            gathered_handles: list[Any] = [None] * world_size
            platform.all_gather_object(gathered_handles, local_handles)
            merged_handles = []
            for parameter_index in range(len(local_handles)):
                merged_handle = {}
                for rank_handles in gathered_handles:
                    merged_handle.update(rank_handles[parameter_index])
                merged_handles.append(merged_handle)
            update_info = None
            update_info_error = None
            try:
                update_info = update_info_class(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    ipc_handles=merged_handles,
                    packed=False,
                )
            except Exception as error:  # pylint: disable=W0718
                update_info_error = error
            _synchronize_error(update_info_error, "IPC update construction")
            if update_info is None:
                raise RuntimeError("NPU IPC update construction failed without a synchronized error")

            send_error = None
            if platform.get_rank() == 0:
                try:
                    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
                        requests = [
                            executor.submit(client.receive_ipc_weights, endpoint, update_info)
                            for endpoint in endpoints
                        ]
                        for request in requests:
                            request.result()
                except Exception as error:  # pylint: disable=W0718
                    send_error = error
            try:
                _synchronize_error(send_error, "IPC weight transfer")
            except Exception:
                # An HTTP timeout does not prove that a consumer stopped reading
                # shared storage. Keep every producer tensor until server teardown.
                self._failed_state_dict = state_dict
                state_dict = {}
                raise
            self._run_control(
                "IPC producer synchronization",
                platform.get_current_stream().synchronize,
            )
            self._run_control("IPC refit finish", client.finish_weight_update)
            expected_fingerprint = None
            fingerprint_error = None
            try:
                expected_fingerprint = _policy_weight_fingerprint(state_dict)
            except Exception as error:  # pylint: disable=W0718
                fingerprint_error = error
            _synchronize_error(fingerprint_error, "IPC Actor policy fingerprint")
            if expected_fingerprint is None:
                raise RuntimeError("IPC Actor policy fingerprint failed without a synchronized error")

            def verify_local_fingerprint() -> None:
                """Verify this rank-local rollout replica before version publication."""
                _verify_policy_fingerprints(
                    expected_fingerprint,
                    client.get_policy_weight_fingerprints(snapshot.version),
                )

            self._run_control("IPC policy fingerprint", verify_local_fingerprint)
            self.last_policy_fingerprint = expected_fingerprint
        finally:
            state_dict.clear()

    def close(self) -> None:
        """Release source storage retained through failed server teardown."""
        self._failed_state_dict.clear()


class VLLMGenerationEngine:
    """Adapt optional vLLM generation to the shared rollout contract."""

    name = "vllm"

    def __init__(
        self,
        model: ModelRegistration,
        config: Mapping[str, Any],
        client: Optional[Any] = None,
        refitter: Optional[VLLMWeightRefitter] = None,
    ) -> None:
        """Initialize the lazy vLLM client and optional weight refitter."""
        self._model = model
        self._config = dict(config.get("vllm", {}))
        self._deployment = str(self._config.get("deployment", "disjoint"))
        self._model_implementation = normalize_model_implementation(
            self._config.get("model_implementation", "hyper")
        )
        self._client = client
        self._refitter = refitter
        self.policy_version = 0
        self.policy_fingerprint: Optional[str] = None
        self.policy_fingerprint_changed = False
        self._pending_policy_version: Optional[int] = None
        self._phase = "rollout"

    @property
    def client_initialized(self) -> bool:
        """Return whether the optional vLLM client has been materialized."""
        return self._client is not None

    @property
    def phase(self) -> str:
        """Return the current colocated residency and publication phase."""
        return self._phase

    def _ensure_client(self) -> Any:
        """Return the existing client or launch a process-isolated vLLM server."""
        if self._client is not None:
            return self._client
        if self._model.hyper_model_name != "qwen3_5":
            raise ValueError(
                "The Hyper vLLM backend currently supports only hyper model 'qwen3_5', "
                f"got {self._model.hyper_model_name!r}"
            )
        visible_devices = self._config.get("visible_devices")
        if self._deployment == "colocated":
            visible_devices = _training_physical_device()
        elif visible_devices is None:
            raise ValueError(
                "rollout.vllm.visible_devices must select the NPU devices owned by the external vLLM server"
            )
        host = str(self._config.get("host", "127.0.0.1"))
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("The initial external vLLM server must bind to loopback")
        configured_port = self._config.get("port")
        if configured_port is None:
            port = _open_port()
        else:
            port = int(configured_port)
            if self._deployment == "colocated":
                port += int(os.environ.get("LOCAL_RANK", "0"))
        weight_transfer_backend = "ipc" if self._deployment == "colocated" else "nccl"
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            self._model.weights_path,
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            self._model.name,
            "--tokenizer",
            self._model.tokenizer_path,
            "--tensor-parallel-size",
            str(int(self._config.get("tensor_parallel_size", 1))),
            "--dtype",
            str(self._config.get("dtype", "bfloat16")),
            "--hf-overrides",
            json.dumps(
                {
                    "architectures": [
                        architecture_for_implementation(self._model_implementation)
                    ]
                }
            ),
            "--weight-transfer-config",
            json.dumps({"backend": weight_transfer_backend}),
        ]
        if bool(self._config.get("trust_remote_code", True)):
            command.append("--trust-remote-code")
        if bool(self._config.get("enforce_eager", True)):
            command.append("--enforce-eager")
        if bool(self._config.get("enable_prefix_caching", False)):
            command.append("--enable-prefix-caching")
        if bool(self._config.get("skip_mm_profiling", False)):
            command.append("--skip-mm-profiling")
        if self._deployment == "colocated":
            command.extend(
                (
                    "--enable-sleep-mode",
                    "--additional-config",
                    json.dumps({"weight_nz_mode": 0}),
                )
            )
        for key, option in (
            ("gpu_memory_utilization", "--gpu-memory-utilization"),
            ("kv_cache_memory_bytes", "--kv-cache-memory-bytes"),
            ("max_model_len", "--max-model-len"),
            ("max_num_seqs", "--max-num-seqs"),
            ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ):
            if key in self._config:
                command.extend((option, str(self._config[key])))

        server_environment = os.environ.copy()
        for variable in _DISTRIBUTED_ENVIRONMENT_VARIABLES:
            server_environment.pop(variable, None)
        server_environment.update(
            {
                "ASCEND_RT_VISIBLE_DEVICES": str(visible_devices),
                "VLLM_ASCEND_ENABLE_NZ": "0",
                "VLLM_BATCH_INVARIANT": "1" if bool(self._config.get("batch_invariant", False)) else "0",
                "VLLM_HOST_IP": "127.0.0.1",
                "VLLM_SERVER_DEV_MODE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            }
        )
        if self._deployment == "colocated":
            server_environment["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
            server_environment.pop("PYTORCH_NPU_ALLOC_CONF", None)
        process = subprocess.Popen(  # pylint: disable=R1732
            command,
            env=server_environment,
            shell=False,
            start_new_session=True,
        )
        client = _VLLMHTTPClient(
            process,
            f"http://{host}:{port}",
            self._model.name,
            request_timeout=float(self._config.get("request_timeout", 600)),
        )
        try:
            client.wait_ready(float(self._config.get("startup_timeout", 300)))
        except Exception:
            client.close()
            raise
        self._client = client
        return self._client

    def _capture_initial_policy_fingerprint(self, client: Any) -> None:
        """Record a verified worker baseline before the first colocated refit."""
        if self.policy_fingerprint is not None or not isinstance(client, _VLLMHTTPClient):
            return
        fingerprints = client.get_policy_weight_fingerprints(self.policy_version)
        if not fingerprints:
            raise RuntimeError("vLLM initial policy fingerprint returned no worker results")
        _verify_policy_fingerprints(fingerprints[0], fingerprints)
        self.policy_fingerprint = str(fingerprints[0]["digest"])

    def prepare_for_training(self) -> None:
        """Sleep colocated inference before FSDP training starts."""
        if self._deployment != "colocated":
            return
        if self._phase != "rollout":
            raise RuntimeError(f"Cannot prepare vLLM for training from phase {self._phase!r}")
        client = None
        local_error = None
        try:
            client = self._ensure_client()
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        _synchronize_error(local_error, "server startup")
        if client is None:
            raise RuntimeError("vLLM server startup failed without a synchronized error")
        try:
            self._capture_initial_policy_fingerprint(client)
            client.sleep(level=1, mode="wait")
            if not client.is_sleeping():
                raise RuntimeError("vLLM did not enter sleep mode before training")
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        _synchronize_error(local_error, "sleep before training")
        self._phase = "training"

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute one local variable-length generation request."""
        client = self._ensure_client()
        settings = request.settings
        prompt_token_ids = [
            ids[mask.bool()].detach().cpu().tolist()
            for ids, mask in zip(request.input_ids, request.attention_mask)
        ]
        started = time.perf_counter()
        if isinstance(client, _VLLMHTTPClient):
            completion_records = client.generate_tokens(
                prompt_token_ids,
                settings,
                request_concurrency=int(self._config.get("request_concurrency", 1)),
            )
        else:
            try:
                # vLLM is optional and needed here only for injected in-process test clients.
                from vllm import SamplingParams, TokensPrompt  # pylint: disable=C0415
            except ImportError as error:
                raise RuntimeError("The injected vLLM client requires the optional vLLM package") from error
            sampling = SamplingParams(
                n=1,
                max_tokens=settings.max_new_tokens,
                temperature=settings.temperature if settings.do_sample else 0.0,
                top_p=settings.top_p,
                top_k=settings.top_k if settings.top_k > 0 else -1,
                logprobs=1 if settings.collect_log_probs else None,
                seed=settings.seed,
            )
            prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
            outputs = client.generate(prompts, sampling_params=sampling, use_tqdm=False)
            completion_records = []
            for request_output in outputs:
                completion = request_output.outputs[0]
                token_log_probs = None
                if settings.collect_log_probs:
                    if completion.logprobs is None or len(completion.logprobs) != len(completion.token_ids):
                        raise RuntimeError(
                            "vLLM completion returned incomplete sampled-token log probabilities: "
                            f"tokens={len(completion.token_ids)}, "
                            f"log_probs={None if completion.logprobs is None else len(completion.logprobs)}"
                        )
                    missing_token_ids = [
                        token_id
                        for token_id, candidates in zip(completion.token_ids, completion.logprobs)
                        if token_id not in candidates
                    ]
                    if missing_token_ids:
                        raise RuntimeError(
                            "vLLM completion log probabilities omitted sampled token IDs: "
                            f"{missing_token_ids}"
                        )
                    token_log_probs = [
                        float(candidates[token_id].logprob)
                        for token_id, candidates in zip(completion.token_ids, completion.logprobs)
                    ]
                completion_records.append((list(completion.token_ids), token_log_probs))
        elapsed = time.perf_counter() - started
        response_ids = request.input_ids.new_full(
            (len(completion_records), settings.max_new_tokens), settings.pad_token_id
        )
        response_mask = response_ids.new_zeros(
            response_ids.shape,
            dtype=platform.tensor_dtype.bool,
        )
        rollout_log_probs = None
        if settings.collect_log_probs:
            rollout_log_probs = response_ids.new_zeros(
                response_ids.shape,
                dtype=platform.tensor_dtype.float32,
            )
        for row, (completion_token_ids, completion_log_probs) in enumerate(completion_records):
            tokens = completion_token_ids[: settings.max_new_tokens]
            if tokens:
                response_ids[row, : len(tokens)] = response_ids.new_tensor(tokens)
                response_mask[row, : len(tokens)] = True
            if rollout_log_probs is not None and completion_log_probs is not None:
                rollout_log_probs[row, : len(tokens)] = rollout_log_probs.new_tensor(
                    completion_log_probs[: len(tokens)]
                )
        sequences = platform.cat((request.input_ids, response_ids), dim=-1)
        return GenerationResult(sequences, rollout_log_probs, elapsed, response_mask)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate locally and make colocated replica failures globally fatal."""
        result = None
        local_error = None
        try:
            result = self._generate(request)
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        if self._deployment == "colocated":
            _synchronize_error(local_error, "generation")
        elif local_error is not None:
            raise local_error
        if result is None:
            raise RuntimeError("vLLM generation failed without a synchronized error")
        return result

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Refit and acknowledge a strictly newer matching policy snapshot."""
        if snapshot.model_name != self._model.name:
            raise ValueError(
                f"Policy snapshot model mismatch: expected={self._model.name}, "
                f"received={snapshot.model_name}"
            )
        if snapshot.version <= self.policy_version:
            raise ValueError(
                f"Policy snapshot version must increase: current={self.policy_version}, "
                f"received={snapshot.version}"
            )
        if self._refitter is None:
            raise NotImplementedError(
                "vLLM iterative training requires a concrete VLLMWeightRefitter; "
                "the adapter will not acknowledge a new policy version without loading it"
            )
        if self._deployment == "colocated" and self._phase != "training":
            raise RuntimeError(f"Cannot refit colocated vLLM from phase {self._phase!r}")
        self._refitter.refit(self._ensure_client(), snapshot)
        refit_fingerprint = getattr(self._refitter, "last_policy_fingerprint", None)
        if refit_fingerprint is not None:
            digest = str(refit_fingerprint["digest"])
            self.policy_fingerprint_changed = (
                self.policy_fingerprint is None or digest != self.policy_fingerprint
            )
            self.policy_fingerprint = digest
        if self._deployment == "colocated":
            self._pending_policy_version = snapshot.version
            self._phase = "refit"
        else:
            self.policy_version = snapshot.version

    def prepare_for_rollout(self) -> None:
        """Wake colocated KV memory and atomically expose the pending policy."""
        if self._deployment != "colocated":
            return
        if self._phase not in ("training", "refit"):
            raise RuntimeError(f"Cannot prepare vLLM for rollout from phase {self._phase!r}")
        if self._phase == "refit" and self._pending_policy_version is None:
            raise RuntimeError("Colocated refit completed without a pending policy version")
        client = self._ensure_client()
        local_error = None
        try:
            tags = ("kv_cache",) if self._phase == "refit" else ("weights", "kv_cache")
            client.wake_up(tags)
            client.resume()
            if client.is_sleeping():
                raise RuntimeError("vLLM remained sleeping after KV-cache wake and resume")
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        _synchronize_error(local_error, "wake before rollout")
        if self._pending_policy_version is not None:
            self.policy_version = self._pending_policy_version
        self._pending_policy_version = None
        self._phase = "rollout"

    def close(self) -> None:
        """Release weight-transfer resources and stop an owned vLLM server."""
        if isinstance(self._client, _VLLMHTTPClient):
            self._client.close()
            self._client = None
        if self._refitter is not None:
            close_refitter = getattr(self._refitter, "close", None)
            if callable(close_refitter):
                close_refitter()


@ROLLOUT_ENGINES.register("vllm")
def build_vllm_engine(
    config: Mapping[str, Any],
    model: ModelRegistration,
    actor: Any,
) -> VLLMGenerationEngine:
    """Build the registered optional vLLM generation engine."""
    del actor
    vllm_config = config.get("vllm", {})
    deployment = vllm_config.get("deployment", "disjoint")
    model_implementation = normalize_model_implementation(
        vllm_config.get("model_implementation", "hyper")
    )
    refitter = (
        NPUIPCWeightRefitter(model_implementation)
        if deployment == "colocated"
        else HCCLWeightRefitter(model_implementation)
    )
    return VLLMGenerationEngine(model, config, refitter=refitter)
