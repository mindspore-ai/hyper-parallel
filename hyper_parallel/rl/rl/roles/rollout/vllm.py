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
import asyncio
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from rl.roles.model import (
    ModelRegistration,
    VLLMModelRegistration,
    resolve_vllm_model,
)
from rl.roles.rollout.base import GenerationRequest, GenerationResult
from rl.roles.rollout.registry import ROLLOUT_ENGINES
from rl.roles.rollout.topology import (
    VLLMRolloutTopology,
    resolve_vllm_rollout_topology,
)
from rl.roles.weight_sync import (
    ActorRolloutWeightSync,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
    WeightTransfer,
    build_weight_transfer,
    synchronized_call,
    synchronize_error,
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
_VLLM_SAMPLING_EPS = 1e-5
_VLLM_CHILD_CAPACITY_OVERSUBSCRIPTION = 2


def _load_aiohttp() -> Any:
    """Load the optional HTTP transport only when the vLLM client starts."""
    import aiohttp  # pylint: disable=import-outside-toplevel

    return aiohttp


@dataclass(frozen=True)
class _CompletionRequest:
    """One HTTP parent with its stable output row range and child cost."""

    ordinal: int
    start_row: int
    prompts: list[list[int]]
    seed: Optional[int]
    num_choices: int

    @property
    def child_count(self) -> int:
        """Return the number of vLLM child requests represented by this parent."""
        return len(self.prompts) * self.num_choices


class _VLLMHTTPClient(VLLMWeightSyncClientMixin):
    """Synchronous token and RL control client for one vLLM server process."""

    def __init__(
        self,
        process: Optional[subprocess.Popen],
        base_url: str,
        model_name: str,
        request_timeout: float,
    ) -> None:
        """Bind a persistent HTTP client and optional owned process to one endpoint."""
        self._process = process
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._request_timeout = request_timeout
        self._async_runtime_lock = threading.Lock()
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        self._async_session: Optional[Any] = None
        self._async_connection_limit = 0
        self._async_startup_error: Optional[BaseException] = None

    async def _create_async_session(self, connection_limit: int) -> None:
        """Create the persistent generation transport on its owning event loop."""
        aiohttp = _load_aiohttp()
        connector = aiohttp.TCPConnector(
            limit=connection_limit,
            limit_per_host=connection_limit,
            keepalive_timeout=60.0,
            force_close=False,
        )
        timeout = aiohttp.ClientTimeout(total=self._request_timeout)
        self._async_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )

    def _run_async_runtime(
        self,
        connection_limit: int,
        ready: threading.Event,
    ) -> None:
        """Own one persistent asyncio loop and HTTP session for generation."""
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._async_loop = loop
            loop.run_until_complete(self._create_async_session(connection_limit))
        except BaseException as error:  # pylint: disable=W0718  # Propagate startup to caller thread.
            self._async_startup_error = error
            ready.set()
        else:
            ready.set()
            loop.run_forever()
        finally:
            if loop is not None:
                session = self._async_session
                if session is not None and not session.closed:
                    loop.run_until_complete(session.close())
                loop.close()

    def _ensure_async_runtime(self, connection_limit: int) -> asyncio.AbstractEventLoop:
        """Start or return the persistent generation loop with a bounded pool."""
        if connection_limit <= 0:
            raise ValueError(f"vLLM child capacity must be positive, got {connection_limit}")
        with self._async_runtime_lock:
            if self._async_thread is not None and self._async_thread.is_alive():
                if connection_limit > self._async_connection_limit:
                    raise ValueError(
                        "vLLM child capacity cannot exceed the persistent connection pool: "
                        f"capacity={connection_limit}, pool={self._async_connection_limit}"
                    )
                if self._async_loop is None:
                    raise RuntimeError("vLLM async HTTP runtime has no event loop")
                return self._async_loop
            self._async_connection_limit = connection_limit
            self._async_startup_error = None
            ready = threading.Event()
            self._async_thread = threading.Thread(
                target=self._run_async_runtime,
                args=(connection_limit, ready),
                name="vllm-http-client",
                daemon=True,
            )
            self._async_thread.start()
            ready.wait()
            if self._async_startup_error is not None:
                error = self._async_startup_error
                self._async_thread.join()
                self._async_thread = None
                self._async_loop = None
                self._async_session = None
                self._async_connection_limit = 0
                raise RuntimeError(f"vLLM async HTTP runtime failed to start: {error}") from error
            if self._async_loop is None:
                raise RuntimeError("vLLM async HTTP runtime started without an event loop")
            return self._async_loop

    def _close_async_runtime(self) -> None:
        """Close the persistent session and stop its owning event loop."""
        with self._async_runtime_lock:
            loop = self._async_loop
            thread = self._async_thread
            session = self._async_session
            if loop is None or thread is None:
                return
            try:
                if session is not None and not session.closed and loop.is_running():
                    close_future = asyncio.run_coroutine_threadsafe(session.close(), loop)
                    close_future.result(timeout=10)
            finally:
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError("vLLM async HTTP runtime did not stop within 10 seconds")
                self._async_loop = None
                self._async_thread = None
                self._async_session = None
                self._async_connection_limit = 0

    async def _request_async(
        self,
        method: str,
        route: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute one generation request through the persistent connection pool."""
        aiohttp = _load_aiohttp()
        session = self._async_session
        if session is None:
            raise RuntimeError("vLLM async HTTP session is not initialized")
        url = f"{self._base_url}/{route.lstrip('/')}"
        try:
            async with session.request(method, url, json=payload) as response:
                body = await response.read()
                status = response.status
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise RuntimeError(f"vLLM HTTP {method} {route} failed: {error}") from error
        if status >= 400:
            message = body.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP {method} {route} failed with status {status}: {message}"
            )
        if not body:
            return {}
        try:
            result = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(f"vLLM HTTP {method} {route} returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError(f"vLLM HTTP {method} {route} returned a non-object response")
        return result

    @property
    def is_server_owner(self) -> bool:
        """Return whether this client owns the endpoint's server process."""
        return self._process is not None

    def _request(
        self,
        method: str,
        route: str,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        base_url: Optional[str] = None,
        request_headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if data is None else {"Content-Type": "application/json"}
        if request_headers is not None:
            headers.update(request_headers)
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
            self._raise_if_owned_process_exited()
            try:
                self._request("GET", "health", payload=None, timeout=2)
            except RuntimeError as error:
                last_error = error
                self._raise_if_owned_process_exited()
                time.sleep(1)
                continue
            self._raise_if_owned_process_exited()
            return
        raise RuntimeError(
            f"vLLM server did not become ready within {startup_timeout} seconds: {last_error}"
        )

    def _raise_if_owned_process_exited(self) -> None:
        """Fail immediately when the server process exits during startup."""
        if self._process is None:
            return
        return_code = self._process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM server exited during startup with code {return_code}")

    def _completion_payload(
        self,
        prompts: list[list[int]],
        seed: Optional[int],
        settings: Any,
        num_choices: int = 1,
    ) -> dict[str, Any]:
        """Build one OpenAI completion payload without changing sampling semantics."""
        payload = {
            "model": self._model_name,
            "prompt": prompts,
            "max_tokens": settings.max_new_tokens,
            "temperature": settings.temperature if settings.do_sample else 0.0,
            "top_p": settings.top_p,
            "top_k": settings.top_k if settings.top_k > 0 else -1,
            "n": num_choices,
            "logprobs": 1 if settings.collect_log_probs else None,
            "ignore_eos": settings.ignore_eos,
            "return_token_ids": True,
            "add_special_tokens": False,
            "stop_token_ids": (
                [] if settings.ignore_eos else list(settings.eos_token_ids)
            ),
        }
        if seed is not None:
            payload["seed"] = seed
        return payload

    def _completion_records_from_response(
        self,
        prompts: list[list[int]],
        num_choices: int,
        settings: Any,
        result: Mapping[str, Any],
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Validate one OpenAI response and restore its prompt-major child order."""
        choices = result.get("choices")
        expected_choices = len(prompts) * num_choices
        if not isinstance(choices, list) or len(choices) != expected_choices:
            raise RuntimeError(
                "vLLM completion response count mismatch: "
                f"expected={expected_choices}, received={len(choices) if isinstance(choices, list) else None}"
            )
        indices = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise RuntimeError("vLLM completion response choice must be a mapping")
            index = choice.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RuntimeError("vLLM completion response choice index must be an integer")
            indices.append(index)
        if sorted(indices) != list(range(expected_choices)):
            raise RuntimeError(
                "vLLM completion response choice indices must be a unique contiguous range: "
                f"expected=0..{expected_choices - 1}, received={indices}"
            )
        records = []
        for choice in sorted(choices, key=lambda item: item["index"]):
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

    async def _generate_completion_request(
        self,
        request: _CompletionRequest,
        settings: Any,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Execute one admitted completion parent through the persistent pool."""
        payload = self._completion_payload(
            request.prompts,
            request.seed,
            settings,
            request.num_choices,
        )
        result = await self._request_async("POST", "v1/completions", payload)
        return self._completion_records_from_response(
            request.prompts,
            request.num_choices,
            settings,
            result,
        )

    @staticmethod
    def _completion_requests(
        prompt_token_ids: list[list[int]],
        settings: Any,
        child_capacity: int,
        batch_invariant: bool,
        row_seeds: Optional[tuple[int, ...]] = None,
    ) -> list[_CompletionRequest]:
        """Build capacity-safe parents while preserving each original row seed."""
        requests = []
        if row_seeds is None:
            normalized_seeds: tuple[Optional[int], ...] = tuple(
                None if settings.seed is None else settings.seed + row
                for row in range(len(prompt_token_ids))
            )
        else:
            if len(row_seeds) != len(prompt_token_ids):
                raise ValueError(
                    "Generation row_seeds must align with prompt rows: "
                    f"seeds={len(row_seeds)}, prompts={len(prompt_token_ids)}"
                )
            normalized_seeds = tuple(row_seeds)
        if all(seed is None for seed in normalized_seeds):
            for start in range(0, len(prompt_token_ids), child_capacity):
                prompts = prompt_token_ids[start:start + child_capacity]
                requests.append(
                    _CompletionRequest(len(requests), start, prompts, None, 1)
                )
            return requests
        if any(seed is None for seed in normalized_seeds):
            raise ValueError("Generation row_seeds must be either all integers or all null")
        if (
            not batch_invariant
            or not settings.do_sample
            or float(settings.temperature) < _VLLM_SAMPLING_EPS
        ):
            return [
                _CompletionRequest(row, row, [prompt_ids], normalized_seeds[row], 1)
                for row, prompt_ids in enumerate(prompt_token_ids)
            ]
        start = 0
        while start < len(prompt_token_ids):
            end = start + 1
            while (
                end < len(prompt_token_ids)
                and end - start < child_capacity
                and prompt_token_ids[end] == prompt_token_ids[start]
                and normalized_seeds[end] == normalized_seeds[start] + end - start
            ):
                end += 1
            requests.append(
                _CompletionRequest(
                    len(requests),
                    start,
                    [prompt_token_ids[start]],
                    normalized_seeds[start],
                    end - start,
                )
            )
            start = end
        return requests

    async def _dispatch_completion_requests(
        self,
        requests: list[_CompletionRequest],
        row_count: int,
        settings: Any,
        child_capacity: int,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Run a child-bounded rolling pool and restore stable output row order."""
        results: list[Optional[tuple[list[int], Optional[list[float]]]]] = [None] * row_count
        pending = deque(requests)
        active: dict[asyncio.Task, _CompletionRequest] = {}
        inflight_children = 0
        pending_children = sum(request.child_count for request in requests)
        while pending or active:
            while pending:
                available_children = child_capacity - inflight_children
                request = next(
                    (
                        candidate
                        for candidate in pending
                        if candidate.child_count <= available_children
                    ),
                    None,
                )
                if request is None:
                    break
                pending.remove(request)
                pending_children -= request.child_count
                inflight_children += request.child_count
                task = asyncio.create_task(
                    self._generate_completion_request(request, settings)
                )
                active[task] = request
            if not active:
                raise RuntimeError(
                    "vLLM child admission made no progress: "
                    f"capacity={child_capacity}, pending_children={pending_children}"
                )
            completed, _ = await asyncio.wait(
                tuple(active),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in sorted(completed, key=lambda item: active[item].ordinal):
                request = active.pop(task)
                inflight_children -= request.child_count
                try:
                    records = task.result()
                except Exception:
                    other_tasks = tuple(active)
                    cancellable_tasks = tuple(
                        active_task for active_task in other_tasks if not active_task.done()
                    )
                    for active_task in cancellable_tasks:
                        active_task.cancel()
                    if other_tasks:
                        await asyncio.gather(*other_tasks, return_exceptions=True)
                    raise
                if len(records) != request.child_count:
                    raise RuntimeError(
                        "vLLM admitted parent returned an unexpected child count: "
                        f"expected={request.child_count}, received={len(records)}"
                    )
                for offset, record in enumerate(records):
                    results[request.start_row + offset] = record
        if any(record is None for record in results):
            raise RuntimeError("vLLM async admission completed with missing output rows")
        return [record for record in results if record is not None]

    def generate_tokens(
        self,
        prompt_token_ids: list[list[int]],
        settings: Any,
        child_capacity: int,
        batch_invariant: bool = False,
        row_seeds: Optional[tuple[int, ...]] = None,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Generate ordered records through persistent child-bounded HTTP admission."""
        if child_capacity <= 0:
            raise ValueError(f"vLLM child_capacity must be positive, got {child_capacity}")
        requests = self._completion_requests(
            prompt_token_ids,
            settings,
            child_capacity,
            batch_invariant,
            row_seeds,
        )
        loop = self._ensure_async_runtime(child_capacity)
        future = asyncio.run_coroutine_threadsafe(
            self._dispatch_completion_requests(
                requests,
                len(prompt_token_ids),
                settings,
                child_capacity,
            ),
            loop,
        )
        return future.result()

    def close(self) -> None:
        """Terminate the server and all EngineCore descendants."""
        runtime_error = None
        try:
            self._close_async_runtime()
        except Exception as error:  # pylint: disable=W0718  # Process cleanup must still run.
            runtime_error = error
        if self._process is None:
            if runtime_error is not None:
                raise runtime_error
            return
        process_group_id = self._process.pid
        parent_running = self._process.poll() is None
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError as error:
            if runtime_error is not None:
                raise runtime_error from error
            return
        if not parent_running:
            # The process-group leader can exit before its EngineCore descendants.
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._wait_process_group_exit(process_group_id)
            if runtime_error is not None:
                raise runtime_error
            return
        try:
            self._process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait(timeout=10)
        else:
            # The API server leader may exit before EngineCore descendants.
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._wait_process_group_exit(process_group_id)
        if runtime_error is not None:
            raise runtime_error

    @staticmethod
    def _has_live_process_group_members(process_group_id: int) -> bool:
        """Return whether a Linux process group contains a non-zombie member."""
        try:
            entries = tuple(Path("/proc").iterdir())
        except OSError:
            return True
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            command_end = stat.rfind(")")
            if command_end < 0:
                continue
            fields = stat[command_end + 2 :].split()
            if len(fields) < 3:
                continue
            try:
                member_process_group = int(fields[2])
            except ValueError:
                continue
            if member_process_group == process_group_id and fields[0] != "Z":
                return True
        return False

    @staticmethod
    def _wait_process_group_exit(process_group_id: int, timeout: float = 30) -> None:
        """Wait until the server process group no longer owns IPC consumers."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                return
            if not _VLLMHTTPClient._has_live_process_group_members(process_group_id):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"vLLM process group {process_group_id} remained alive after shutdown"
                )
            time.sleep(0.1)


class VLLMGenerationEngine:
    """Adapt optional vLLM generation to the shared rollout contract."""
    name = "vllm"

    def __init__(
        self,
        model: ModelRegistration,
        config: Mapping[str, Any],
        client: Optional[Any] = None,
        refitter: Optional[WeightTransfer] = None,
        rollout_model: Optional[VLLMModelRegistration] = None,
    ) -> None:
        """Initialize the lazy vLLM client and policy synchronization."""
        self._model = model
        self._config = dict(config.get("vllm", {}))
        self._deployment = str(self._config.get("deployment", "disjoint"))
        self._rollout_model = rollout_model or resolve_vllm_model(
            model,
            self._config.get("model_implementation", "native"),
        )
        self._client = client
        self._resolved_topology: Optional[VLLMRolloutTopology] = None
        self._trainer_tp_group: Optional[Any] = None
        self._trainer_tp_rank = 0
        self._trainer_tp_size = 1
        self._trainer_request_rank: Optional[int] = None
        self._trainer_request_size: Optional[int] = None
        self._request_owner_generate_count = 0
        self._weight_sync = ActorRolloutWeightSync(
            model.name,
            self._deployment,
            self._ensure_client,
            refitter,
        )

    def configure_trainer_tensor_parallel(
        self,
        *,
        group: Any,
        tp_rank: int,
        tp_size: int,
        request_rank: int,
        request_size: int,
    ) -> None:
        """Assign one HTTP request owner per Trainer TP group."""
        if tp_size <= 1:
            raise ValueError("Trainer TP request ownership requires tp_size > 1")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"Trainer TP rank must be in [0, {tp_size}), got {tp_rank}"
            )
        if request_size <= 0 or not 0 <= request_rank < request_size:
            raise ValueError(
                "Trainer logical request rank is invalid: "
                f"rank={request_rank}, size={request_size}"
            )
        if group is None:
            raise ValueError("Trainer TP request ownership requires a process group")
        self._trainer_tp_group = group
        self._trainer_tp_rank = int(tp_rank)
        self._trainer_tp_size = int(tp_size)
        self._trainer_request_rank = int(request_rank)
        self._trainer_request_size = int(request_size)

    @property
    def is_request_owner(self) -> bool:
        """Return whether this Trainer rank submits rollout HTTP requests."""
        return self._trainer_tp_rank == 0

    @property
    def request_owner_generate_count(self) -> int:
        """Return the number of generation calls owned by this rank."""
        return self._request_owner_generate_count

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
    def weight_sync_configured_strategy(self) -> Optional[str]:
        """Return the effective configured weight-transfer strategy."""
        return self._weight_sync.configured_strategy

    @property
    def weight_sync_last_strategy(self) -> Optional[str]:
        """Return the strategy that completed the latest publication."""
        return self._weight_sync.last_strategy

    @property
    def weight_sync_fallback_count(self) -> int:
        """Return the number of successful fallback publications."""
        return self._weight_sync.fallback_count

    @property
    def weight_sync_direct_success_count(self) -> int:
        """Return the number of successful direct publications."""
        return self._weight_sync.direct_success_count

    @property
    def phase(self) -> str:
        """Return the current colocated residency and publication phase."""
        return self._weight_sync.phase

    def _server_endpoint(self) -> tuple[str, str, int]:
        """Resolve the shared deployment's devices, owner, and explicit endpoint."""
        topology_environment = dict(os.environ)
        if "LOCAL_RANK" not in topology_environment:
            topology_environment["LOCAL_RANK"] = str(platform.get_rank())
        if "LOCAL_WORLD_SIZE" not in topology_environment:
            topology_environment["LOCAL_WORLD_SIZE"] = str(platform.get_world_size())
        self._resolved_topology = resolve_vllm_rollout_topology(
            self._config,
            topology_environment,
        )
        return (
            self._resolved_topology.visible_devices_csv,
            self._resolved_topology.host,
            self._resolved_topology.port,
        )

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
            "--data-parallel-size",
            str(int(self._config.get("data_parallel_size", 1))),
            "--dtype",
            str(self._config.get("dtype", "bfloat16")),
            "--weight-transfer-config",
            json.dumps({"backend": weight_transfer_backend}),
        ]
        if self._rollout_model.is_hyper:
            command.extend(
                (
                    "--hf-overrides",
                    json.dumps(
                        {
                            "architectures": [self._rollout_model.architecture]
                        }
                    ),
                )
            )
        if bool(self._config.get("trust_remote_code", True)):
            command.append("--trust-remote-code")
        if bool(self._config.get("enforce_eager", True)):
            command.append("--enforce-eager")
        for key, option in (
            ("enable_prefix_caching", "--enable-prefix-caching"),
            ("enable_chunked_prefill", "--enable-chunked-prefill"),
        ):
            if key in self._config:
                command.append(option if bool(self._config[key]) else f"--no-{option[2:]}")
        if bool(self._config.get("enable_prompt_tokens_details", False)):
            command.append("--enable-prompt-tokens-details")
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
            ("attention_backend", "--attention-backend"),
            ("gpu_memory_utilization", "--gpu-memory-utilization"),
            ("kv_cache_memory_bytes", "--kv-cache-memory-bytes"),
            ("max_model_len", "--max-model-len"),
            ("max_num_seqs", "--max-num-seqs"),
            ("max_num_batched_tokens", "--max-num-batched-tokens"),
            ("block_size", "--block-size"),
            ("logprobs_mode", "--logprobs-mode"),
        ):
            if key in self._config:
                command.extend((option, str(self._config[key])))
        profiler_config = self._config.get("profiler_config")
        if profiler_config is not None:
            command.extend(("--profiler-config", json.dumps(profiler_config)))
        return command

    def _server_environment(self, visible_devices: str) -> dict[str, str]:
        """Build an isolated environment for the owned vLLM process."""
        server_environment = os.environ.copy()
        for variable in _DISTRIBUTED_ENVIRONMENT_VARIABLES:
            server_environment.pop(variable, None)
        server_environment.pop("HYPER_RL_CONSISTENCY_PROFILE", None)
        server_environment.update(
            {
                "ASCEND_RT_VISIBLE_DEVICES": str(visible_devices),
                "HYPER_RL_ROLLOUT_VISIBLE_DEVICES": str(visible_devices),
                "VLLM_ASCEND_ENABLE_NZ": "0",
                "VLLM_BATCH_INVARIANT": "1" if bool(self._config.get("batch_invariant", False)) else "0",
                "VLLM_HOST_IP": "127.0.0.1",
                "VLLM_SERVER_DEV_MODE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            }
        )
        for config_key, environment_key in (
            ("server_hccl_if_base_port", "HCCL_IF_BASE_PORT"),
            (
                "server_hccl_npu_socket_port_range",
                "HCCL_NPU_SOCKET_PORT_RANGE",
            ),
        ):
            value = self._config.get(config_key)
            if value is not None:
                server_environment[environment_key] = str(value)
        consistency_profile = self._config.get("consistency_profile")
        if consistency_profile is not None:
            server_environment["HYPER_RL_CONSISTENCY_PROFILE"] = str(consistency_profile)
        if self._deployment == "colocated":
            server_environment["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
            server_environment.pop("PYTORCH_NPU_ALLOC_CONF", None)
        return server_environment

    def _launch_client(self, visible_devices: str, host: str, port: int) -> Any:
        """Launch the owned server process and wait for its health endpoint."""
        try:
            connection = socket.create_connection((host, port), timeout=0.2)
        except OSError:
            pass
        else:
            connection.close()
            raise RuntimeError(
                f"Configured vLLM endpoint is already in use before launch: {host}:{port}"
            )
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

    def _connect_client(self, host: str, port: int) -> Any:
        """Connect to a shared server after its owner passed the startup barrier."""
        return _VLLMHTTPClient(
            None,
            f"http://{host}:{port}",
            self._model.name,
            request_timeout=float(self._config.get("request_timeout", 600)),
        )

    def _ensure_client(self) -> Any:
        """Return the existing client or launch a process-isolated vLLM server."""
        if self._client is not None:
            return self._client
        visible_devices, host, port = self._server_endpoint()
        topology = self._resolved_topology
        if topology is None:
            raise RuntimeError("Shared vLLM topology was not resolved before startup")
        client = None
        local_error = None
        operation = "shared vLLM server startup"
        try:
            if topology.server_owner:
                try:
                    client = self._launch_client(visible_devices, host, port)
                except Exception as error:  # pylint: disable=W0718
                    local_error = error
            else:
                try:
                    client = self._connect_client(host, port)
                except Exception as error:  # pylint: disable=W0718
                    local_error = error
            synchronize_error(local_error, operation)
        except Exception:
            if client is not None:
                client.close()
            raise
        if client is None:
            raise RuntimeError("Shared vLLM startup returned no client")
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
        row_seeds: Optional[tuple[int, ...]] = None,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Generate through an injected in-process vLLM test client."""
        try:
            from vllm import SamplingParams, TokensPrompt  # pylint: disable=C0415
        except ImportError as error:
            raise RuntimeError(
                "The injected vLLM client requires the optional vLLM package"
            ) from error

        def sampling_params(seed: Optional[int]) -> Any:
            """Build sampling parameters for one deterministic output row."""
            return SamplingParams(
                n=1,
                max_tokens=settings.max_new_tokens,
                temperature=settings.temperature if settings.do_sample else 0.0,
                top_p=settings.top_p,
                top_k=settings.top_k if settings.top_k > 0 else -1,
                logprobs=1 if settings.collect_log_probs else None,
                seed=seed,
                stop_token_ids=(
                    [] if settings.ignore_eos else list(settings.eos_token_ids)
                ),
                ignore_eos=settings.ignore_eos,
            )

        if row_seeds is not None and len(row_seeds) != len(prompt_token_ids):
            raise ValueError(
                "Generation row_seeds must align with prompt rows: "
                f"seeds={len(row_seeds)}, prompts={len(prompt_token_ids)}"
            )
        sampling = (
            sampling_params(settings.seed)
            if row_seeds is None
            else [sampling_params(seed) for seed in row_seeds]
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
        row_seeds: Optional[tuple[int, ...]] = None,
    ) -> list[tuple[list[int], Optional[list[float]]]]:
        """Generate completions through HTTP or an injected local client."""
        if isinstance(client, _VLLMHTTPClient):
            return client.generate_tokens(
                prompt_token_ids,
                settings,
                child_capacity=self._local_child_capacity(),
                batch_invariant=bool(self._config.get("batch_invariant", False)),
                row_seeds=row_seeds,
            )
        return self._inprocess_completions(client, prompt_token_ids, settings, row_seeds)

    def _local_child_capacity(self) -> int:
        """Derive this Trainer rank's child admission quota from engine capacity."""
        max_num_seqs = self._config.get("max_num_seqs")
        if max_num_seqs is None:
            raise ValueError(
                "rollout.vllm.max_num_seqs is required for automatic child admission"
            )
        max_num_seqs = int(max_num_seqs)
        # Keep one bounded waiting wave so routing feedback latency cannot starve engines.
        per_engine_capacity = (
            max_num_seqs * _VLLM_CHILD_CAPACITY_OVERSUBSCRIPTION
        )
        if self._trainer_request_size is not None:
            trainer_world_size = self._trainer_request_size
            trainer_rank = self._trainer_request_rank
        else:
            try:
                trainer_world_size = platform.get_world_size()
                trainer_rank = platform.get_rank()
            except (RuntimeError, ValueError):
                trainer_world_size = 1
                trainer_rank = 0
        if trainer_rank is None:
            raise RuntimeError("Trainer logical request rank was not configured")
        engine_count = int(self._config.get("data_parallel_size", 1))
        global_capacity = engine_count * per_engine_capacity
        local_capacity, remainder = divmod(global_capacity, trainer_world_size)
        local_capacity += int(trainer_rank < remainder)
        if local_capacity <= 0:
            raise ValueError(
                "Automatic child admission assigned no capacity to this Trainer rank: "
                f"global_capacity={global_capacity}, trainer_world_size={trainer_world_size}, "
                f"trainer_rank={trainer_rank}"
            )
        return local_capacity

    @staticmethod
    def _build_generation_result(
        request: GenerationRequest,
        completion_records: list[tuple[list[int], Optional[list[float]]]],
        elapsed: float,
        worker_policy_version: Optional[int],
        worker_policy_fingerprint: Optional[str],
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
        return GenerationResult(
            sequences=sequences,
            rollout_log_probs=rollout_log_probs,
            generation_seconds=elapsed,
            response_mask=response_mask,
            worker_policy_version=worker_policy_version,
            worker_policy_fingerprint=worker_policy_fingerprint,
        )

    def _generate_request(
        self,
        request: GenerationRequest,
        client: Any,
        worker_identity: tuple[int, str],
    ) -> GenerationResult:
        """Execute one request without entering distributed error barriers."""
        worker_policy_version, worker_policy_fingerprint = worker_identity
        completion_records = None
        elapsed = None
        prompt_token_ids = [
            ids[mask.bool()].detach().cpu().tolist()
            for ids, mask in zip(request.input_ids, request.attention_mask)
        ]
        started = time.perf_counter()
        completion_records = self._completion_records(
            client,
            prompt_token_ids,
            request.settings,
            request.row_seeds,
        )
        if completion_records is None:
            raise RuntimeError("vLLM generation request returned no completions")
        elapsed = time.perf_counter() - started
        if completion_records is None or elapsed is None:
            raise RuntimeError("vLLM generation request failed without a synchronized error")
        return self._build_generation_result(
            request,
            completion_records,
            elapsed,
            worker_policy_version,
            worker_policy_fingerprint,
        )

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute one local variable-length generation request."""
        client = self._ensure_client()
        worker_identity = self._weight_sync.generation_identity(client)
        result = None
        request_error = None
        try:
            result = self._generate_request(request, client, worker_identity)
        except Exception as error:  # pylint: disable=W0718
            request_error = error
        self.synchronize_error(request_error, "generation request")
        if result is None:
            raise RuntimeError("vLLM generation request failed without a synchronized error")
        served_identity = self._weight_sync.generation_identity(client)
        if served_identity != worker_identity:
            raise RuntimeError(
                "vLLM policy identity changed while serving a generation request: "
                f"before={worker_identity}, after={served_identity}"
            )
        return result

    def _validate_tp_request(self, request: GenerationRequest) -> None:
        """Require TP siblings to submit the same logical generation request."""
        contract = (
            request.input_ids.detach().cpu().tolist(),
            request.attention_mask.detach().cpu().tolist(),
            request.row_seeds,
            repr(request.settings),
        )
        contracts: list[Any] = [None] * self._trainer_tp_size
        platform.all_gather_object(contracts, contract, self._trainer_tp_group)
        if any(candidate != contracts[0] for candidate in contracts[1:]):
            raise RuntimeError(
                "Trainer TP siblings submitted different rollout requests"
            )

    def _broadcast_tp_result(
        self,
        request: GenerationRequest,
        result: Optional[GenerationResult],
        worker_identity: tuple[int, str],
    ) -> GenerationResult:
        """Broadcast one owner result to every Trainer TP sibling."""
        response_shape = (
            request.input_ids.shape[0],
            request.settings.max_new_tokens,
        )
        sequence_shape = (
            request.input_ids.shape[0],
            request.input_ids.shape[1] + request.settings.max_new_tokens,
        )
        if self.is_request_owner:
            if result is None:
                raise RuntimeError("Trainer TP request owner produced no rollout result")
            sequences = result.sequences
            response_mask = result.response_mask
            rollout_log_probs = result.rollout_log_probs
            generation_seconds = result.generation_seconds
        else:
            sequences = request.input_ids.new_empty(sequence_shape)
            response_mask = request.attention_mask.new_empty(
                response_shape,
                dtype=platform.tensor_dtype.bool,
            )
            rollout_log_probs = (
                request.input_ids.new_empty(
                    response_shape,
                    dtype=platform.tensor_dtype.float32,
                )
                if request.settings.collect_log_probs
                else None
            )
            generation_seconds = None
        if response_mask is None:
            raise RuntimeError("Trainer TP rollout result requires response_mask")
        platform.broadcast(
            sequences,
            group=self._trainer_tp_group,
            group_src=0,
        )
        platform.broadcast(
            response_mask,
            group=self._trainer_tp_group,
            group_src=0,
        )
        if request.settings.collect_log_probs:
            if rollout_log_probs is None:
                raise RuntimeError("Trainer TP rollout result requires raw logprobs")
            platform.broadcast(
                rollout_log_probs,
                group=self._trainer_tp_group,
                group_src=0,
            )
        elapsed: list[Any] = [None] * self._trainer_tp_size
        platform.all_gather_object(
            elapsed,
            generation_seconds if self.is_request_owner else None,
            self._trainer_tp_group,
        )
        if elapsed[0] is None or any(value is not None for value in elapsed[1:]):
            raise RuntimeError(
                f"Trainer TP rollout timing ownership is invalid: {elapsed}"
            )
        return GenerationResult(
            sequences=sequences,
            rollout_log_probs=rollout_log_probs,
            generation_seconds=float(elapsed[0]),
            response_mask=response_mask,
            worker_policy_version=worker_identity[0],
            worker_policy_fingerprint=worker_identity[1],
        )

    def _generate_tp_owned(self, request: GenerationRequest) -> GenerationResult:
        """Generate once per logical DP coordinate and replicate within TP."""
        self._validate_tp_request(request)
        client = self._ensure_client()
        worker_identity = self._weight_sync.generation_identity(client)
        result = None
        request_error = None
        if self.is_request_owner:
            try:
                result = self._generate_request(request, client, worker_identity)
                self._request_owner_generate_count += 1
            except Exception as error:  # pylint: disable=W0718
                request_error = error
        self.synchronize_error(request_error, "TP-owned generation request")
        served_identity = self._weight_sync.generation_identity(client)
        if served_identity != worker_identity:
            raise RuntimeError(
                "vLLM policy identity changed while serving a TP-owned request: "
                f"before={worker_identity}, after={served_identity}"
            )
        return self._broadcast_tp_result(request, result, worker_identity)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate locally and make colocated replica failures globally fatal."""
        result = None
        local_error = None
        try:
            result = (
                self._generate_tp_owned(request)
                if getattr(self, "_trainer_tp_size", 1) > 1
                else self._generate(request)
            )
        except Exception as error:  # pylint: disable=W0718
            local_error = error
        self.synchronize_error(local_error, "generation")
        if result is None:
            raise RuntimeError("vLLM generation failed without a synchronized error")
        return result

    def synchronize_error(self, local_error: Optional[Exception], operation: str) -> None:
        """Propagate rollout and postprocessing failures across training ranks."""
        synchronize_error(local_error, operation)

    def update_weights(self, snapshot: PolicySnapshot) -> None:
        """Transfer and publish a strictly newer policy snapshot."""
        self._weight_sync.update_weights(snapshot)

    def prepare_for_rollout(self) -> None:
        """Wake rollout memory and expose a completed policy transfer."""
        self._weight_sync.prepare_for_rollout()

    def close(self) -> None:
        """Release weight-transfer resources and stop an owned vLLM server."""
        if isinstance(self._client, _VLLMHTTPClient):
            synchronized_call("shared vLLM server shutdown", self._client.close)
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
    rollout_model = resolve_vllm_model(
        model,
        vllm_config.get("model_implementation", "native")
    )
    weight_sync_config = vllm_config.get("weight_sync", {})
    if not isinstance(weight_sync_config, Mapping):
        raise ValueError("rollout.vllm.weight_sync must be a mapping")
    weight_transfer = build_weight_transfer(
        deployment,
        rollout_model,
        tensor_parallel_size=int(vllm_config.get("tensor_parallel_size", 1)),
        data_parallel_size=int(vllm_config.get("data_parallel_size", 1)),
        bucket_size_bytes=int(weight_sync_config.get("bucket_size_mb", 128)) * 2**20,
        strategy=str(weight_sync_config.get("strategy", "full_gather")),
        fallback_strategy=str(
            weight_sync_config.get("fallback_strategy", "none")
        ),
    )
    return VLLMGenerationEngine(
        model,
        config,
        refitter=weight_transfer,
        rollout_model=rollout_model,
    )
