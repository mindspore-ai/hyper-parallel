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
"""Process-isolated vLLM rollout adapter."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationRequest, GenerationResult
from rl.roles.rollout.registry import ROLLOUT_ENGINES
from rl.roles.weight_sync import (
    ActorRolloutWeightSync,
    CPUWeightTransfer,
    HCCLWeightTransfer,
    NPUIPCWeightTransfer,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
    WeightTransfer,
    architecture_for_implementation,
    build_weight_transfer,
    normalize_model_implementation,
    policy_weight_fingerprint,
    synchronize_error,
    verify_policy_fingerprints,
)
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
    """Compatibility wrapper around the weight-sync error barrier."""
    synchronize_error(local_error, operation)
def _policy_weight_fingerprint(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper around the canonical Actor fingerprint."""
    return policy_weight_fingerprint(state_dict)
def _verify_policy_fingerprints(
    expected: Mapping[str, Any],
    actual: list[Mapping[str, Any]],
) -> None:
    """Compatibility wrapper around rollout fingerprint verification."""
    verify_policy_fingerprints(expected, actual)
class _VLLMHTTPClient(VLLMWeightSyncClientMixin):
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
VLLMWeightRefitter = WeightTransfer
class CPUStateDictRefitter(CPUWeightTransfer):
    """Compatibility name for the weight-sync CPU transfer."""
class HCCLWeightRefitter(HCCLWeightTransfer):
    """Compatibility name for the weight-sync HCCL transfer."""
    @staticmethod
    def _policy_fingerprint(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        return _policy_weight_fingerprint(state_dict)
    @staticmethod
    def _verify_fingerprints(
        expected: Mapping[str, Any],
        actual: list[Mapping[str, Any]],
    ) -> None:
        _verify_policy_fingerprints(expected, actual)
class NPUIPCWeightRefitter(NPUIPCWeightTransfer):
    """Compatibility name for the weight-sync NPU IPC transfer."""
    @staticmethod
    def _policy_fingerprint(state_dict: Mapping[str, Any]) -> dict[str, Any]:
        return _policy_weight_fingerprint(state_dict)
    @staticmethod
    def _verify_fingerprints(
        expected: Mapping[str, Any],
        actual: list[Mapping[str, Any]],
    ) -> None:
        _verify_policy_fingerprints(expected, actual)
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
        """Initialize the lazy vLLM client and policy synchronization."""
        self._model = model
        self._config = dict(config.get("vllm", {}))
        self._deployment = str(self._config.get("deployment", "disjoint"))
        self._model_implementation = normalize_model_implementation(
            self._config.get("model_implementation", "hyper")
        )
        self._client = client
        self._weight_sync = ActorRolloutWeightSync(
            model.name,
            self._deployment,
            self._ensure_client,
            refitter,
        )
    @property
    def client_initialized(self) -> bool:
        """Return whether the optional vLLM client has been materialized."""
        return self._client is not None
    @property
    def policy_version(self) -> int:
        """Return the policy version currently visible to rollout."""
        return self._weight_sync.policy_version
    @property
    def policy_fingerprint(self) -> Optional[str]:
        """Return the last verified rollout policy fingerprint."""
        return self._weight_sync.policy_fingerprint
    @property
    def policy_fingerprint_changed(self) -> bool:
        """Return whether the latest transfer changed the policy fingerprint."""
        return self._weight_sync.policy_fingerprint_changed
    @property
    def phase(self) -> str:
        """Return the current colocated residency and publication phase."""
        return self._weight_sync.phase

    def _server_endpoint(self) -> tuple[str, str, int]:
        """Resolve device ownership and the rank-local loopback endpoint."""
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
        return str(visible_devices), host, port

    def _server_command(self, host: str, port: int) -> list[str]:
        """Build the vLLM server command for this rollout configuration."""
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
        return command

    def _server_environment(self, visible_devices: str) -> dict[str, str]:
        """Build an isolated environment for the owned vLLM process."""
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
        return server_environment

    def _launch_client(self, visible_devices: str, host: str, port: int) -> Any:
        """Launch the owned server process and wait for its health endpoint."""
        process = subprocess.Popen(  # pylint: disable=R1732
            self._server_command(host, port),
            env=self._server_environment(visible_devices),
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
        return client

    def _ensure_client(self) -> Any:
        """Return the existing client or launch a process-isolated vLLM server."""
        if self._client is not None:
            return self._client
        if self._model.hyper_model_name != "qwen3_5":
            raise ValueError(
                "The Hyper vLLM backend currently supports only hyper model 'qwen3_5', "
                f"got {self._model.hyper_model_name!r}"
            )
        visible_devices, host, port = self._server_endpoint()
        client = self._launch_client(visible_devices, host, port)
        self._client = client
        return self._client
    def prepare_for_training(self) -> None:
        """Sleep colocated rollout before policy Actor training starts."""
        self._weight_sync.prepare_for_training()

    @staticmethod
    def _sampled_log_probs(completion: Any) -> list[float]:
        """Extract sampled-token log probabilities from one vLLM completion."""
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
        return [
            float(candidates[token_id].logprob)
            for token_id, candidates in zip(completion.token_ids, completion.logprobs)
        ]

    def _inprocess_completions(
        self,
        client: Any,
        prompt_token_ids: list[list[int]],
        settings: Any,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Generate through an injected in-process vLLM test client."""
        try:
            from vllm import SamplingParams, TokensPrompt  # pylint: disable=C0415
        except ImportError as error:
            raise RuntimeError(
                "The injected vLLM client requires the optional vLLM package"
            ) from error
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
        records = []
        for request_output in outputs:
            completion = request_output.outputs[0]
            token_log_probs = (
                self._sampled_log_probs(completion)
                if settings.collect_log_probs
                else None
            )
            records.append((list(completion.token_ids), token_log_probs))
        return records

    def _completion_records(
        self,
        client: Any,
        prompt_token_ids: list[list[int]],
        settings: Any,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Generate completions through HTTP or an injected local client."""
        if isinstance(client, _VLLMHTTPClient):
            return client.generate_tokens(
                prompt_token_ids,
                settings,
                request_concurrency=int(self._config.get("request_concurrency", 1)),
            )
        return self._inprocess_completions(client, prompt_token_ids, settings)

    @staticmethod
    def _build_generation_result(
        request: GenerationRequest,
        completion_records: list[tuple[list[int], Optional[list[float]]]],
        elapsed: float,
    ) -> GenerationResult:
        """Pad variable-length completions into the rollout tensor contract."""
        settings = request.settings
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

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute one local variable-length generation request."""
        client = self._ensure_client()
        prompt_token_ids = [
            ids[mask.bool()].detach().cpu().tolist()
            for ids, mask in zip(request.input_ids, request.attention_mask)
        ]
        started = time.perf_counter()
        completion_records = self._completion_records(
            client,
            prompt_token_ids,
            request.settings,
        )
        return self._build_generation_result(
            request,
            completion_records,
            time.perf_counter() - started,
        )
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
        """Transfer and publish a strictly newer policy snapshot."""
        self._weight_sync.update_weights(snapshot)
    def prepare_for_rollout(self) -> None:
        """Wake rollout memory and expose a completed policy transfer."""
        self._weight_sync.prepare_for_rollout()
    def close(self) -> None:
        """Release weight-transfer resources and stop an owned vLLM server."""
        if isinstance(self._client, _VLLMHTTPClient):
            self._client.close()
            self._client = None
        self._weight_sync.close()
@ROLLOUT_ENGINES.register("vllm")
def build_vllm_engine(
    config: Mapping[str, Any],
    model: ModelRegistration,
) -> VLLMGenerationEngine:
    """Build the registered optional vLLM generation engine."""
    vllm_config = config.get("vllm", {})
    deployment = vllm_config.get("deployment", "disjoint")
    model_implementation = normalize_model_implementation(
        vllm_config.get("model_implementation", "hyper")
    )
    weight_transfer = build_weight_transfer(deployment, model_implementation)
    return VLLMGenerationEngine(model, config, refitter=weight_transfer)
