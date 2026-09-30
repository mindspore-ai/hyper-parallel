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
"""Frozen vLLM reward service colocated with Actor rollout devices."""

import asyncio
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence
from urllib.request import Request, urlopen

import aiohttp
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
_DISTRIBUTED_ENV = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
    "HYPER_RL_CONSISTENCY_PROFILE", "HYPER_RL_ROLLOUT_VISIBLE_DEVICES", "PYTORCH_NPU_ALLOC_CONF",
    "RANK_ID", "RANK_SIZE", "DEVICE_ID", "RANK_TABLE_FILE", "VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL",
    "VLLM_DP_SIZE", "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT",
)


class RewardModelClient:
    """Own a colocated, frozen vLLM service without deciding task rewards.

    Args:
        config: Validated service configuration.
        devices: Physical devices also used by Actor rollout.
        owner: Whether this rank may start and stop the service process.
    """

    def __init__(self, config: Mapping[str, Any], devices: Sequence[str], *, owner: bool) -> None:
        """Create an idle handle; load model weights after rollout sleeps."""
        self.config = dict(config)
        self.devices = tuple(devices)
        self.owner = owner
        self.base_url = f"http://127.0.0.1:{config['port']}"
        self.model_name = str(config.get("served_model_name", "hyper-rl-reward"))
        self.state = "not_started"
        self.process: Optional[subprocess.Popen] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._log_handle = None
        self._resources = ExitStack()
        self.request_count = 0
        self.retry_count = 0
        self._tokenizer = None

    @property
    def tokenizer(self) -> Any:
        """Load the RM's own template only for scorers which need text inputs."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"], local_files_only=True)
        return self._tokenizer

    def server_command(self) -> list[str]:
        """Construct a native vLLM command without Actor refit or architecture overrides."""
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(self.config["model_path"]),
                   "--host", "127.0.0.1", "--port", str(self.config["port"]),
                   "--served-model-name", self.model_name, "--enable-sleep-mode", "--enforce-eager",
                   "--additional-config", json.dumps({"weight_nz_mode": 0})]
        defaults = {
            "tensor_parallel_size": ("--tensor-parallel-size", 1),
            "data_parallel_size": ("--data-parallel-size", 1),
            "dtype": ("--dtype", "bfloat16"),
            "gpu_memory_utilization": ("--gpu-memory-utilization", 0.35),
            "max_model_len": ("--max-model-len", 2048),
            "max_num_seqs": ("--max-num-seqs", 8),
            "kv_cache_memory_bytes": ("--kv-cache-memory-bytes", 536870912),
        }
        for key, (option, default) in defaults.items():
            command.extend((option, str(self.config.get(key, default))))
        if self.config.get("scoring") == "discriminative":
            command.extend(("--runner", "pooling"))
        return command

    def server_environment(self) -> dict[str, str]:
        """Isolate RM rank discovery and communication ports from the Trainer."""
        environment = os.environ.copy()
        for name in _DISTRIBUTED_ENV:
            environment.pop(name, None)
        for name in tuple(environment):
            if name.startswith("TORCHELASTIC_"):
                environment.pop(name)
        environment.update({
            "ASCEND_RT_VISIBLE_DEVICES": ",".join(self.devices),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_SERVER_DEV_MODE": "1", "VLLM_BATCH_INVARIANT": "0", "VLLM_ASCEND_ENABLE_NZ": "0",
            "HCCL_IF_BASE_PORT": str(self.config["server_hccl_if_base_port"]),
            "HCCL_NPU_SOCKET_PORT_RANGE": str(self.config["server_hccl_npu_socket_port_range"]),
        })
        return environment

    def control(self, method: str, endpoint: str, timeout: Optional[float] = None) -> dict[str, Any]:
        """Execute a bounded control request; only the owner invokes mutations."""
        request = Request(f"{self.base_url}/{endpoint}", method=method)
        with urlopen(request, timeout=timeout or float(self.config.get("request_timeout", 60))) as response:
            body = response.read()
        return json.loads(body) if body else {}

    def _require_sleep_state(self, expected: bool) -> None:
        sleeping = self.control("GET", "is_sleeping").get("is_sleeping")
        if not isinstance(sleeping, bool) or sleeping != expected:
            raise RuntimeError(f"RM residency mismatch: expected sleeping={expected}, got {sleeping!r}")

    def prepare(self) -> str:
        """Start lazily or wake frozen weights after rollout has yielded the devices."""
        if not self.owner or self.state not in {"not_started", "sleeping"}:
            raise RuntimeError(f"Invalid RM prepare: owner={self.owner}, state={self.state}")
        try:
            if self.state == "not_started":
                self._start()
            else:
                self.control("POST", "wake_up")
            self._require_sleep_state(False)
            self.state = "awake"
            logger.info("reward service awake: devices=%s model=%s", self.devices, self.model_name)
            return self.state
        except Exception:
            self.state = "failed"
            raise

    def _start(self) -> None:
        """Start the reward model service after checking its port is free."""
        try:
            connection = socket.create_connection(("127.0.0.1", int(self.config["port"])), timeout=0.2)
        except OSError:
            connection = None
        if connection is not None:
            connection.close()
            raise RuntimeError(f"RM port already in use: {self.config['port']}")
        log_path = self.config.get("log_path")
        if log_path:
            path = Path(str(log_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self._resources.enter_context(path.open("a", encoding="utf-8"))
        # The owned process spans scoring batches and is closed by close().
        self.process = subprocess.Popen(  # pylint: disable=consider-using-with
            self.server_command(), env=self.server_environment(), shell=False, start_new_session=True,
            stdout=self._log_handle, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + float(self.config.get("startup_timeout", 600))
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Reward vLLM exited during startup: status={self.process.returncode}")
            try:
                self.control("GET", "health", timeout=2)
                return
            except (OSError, ValueError):
                time.sleep(0.5)
        raise TimeoutError("Reward vLLM did not become healthy before startup_timeout")

    def sleep(self) -> str:
        """Drain all service requests and verify frozen RM memory has been released."""
        if not self.owner or self.state != "awake":
            raise RuntimeError(f"Invalid RM sleep: owner={self.owner}, state={self.state}")
        try:
            self.control("POST", "sleep?level=1&mode=wait")
            self._require_sleep_state(True)
            self.state = "sleeping"
            logger.info("reward service sleeping: devices=%s", self.devices)
            return self.state
        except Exception:
            self.state = "failed"
            raise

    async def request(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Send a bounded scoring request while the service is awake."""
        if self.state != "awake":
            raise RuntimeError(f"Scoring requires an awake reward service, got {self.state}")
        if self._session is None:
            timeout = float(self.config.get("request_timeout", self.config.get("timeout_seconds", 60)))
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout),
                connector=aiohttp.TCPConnector(limit=int(self.config.get("max_concurrency", 16))),
            )
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        retries = int(self.config.get("max_retries", 2))
        for attempt in range(retries + 1):
            self.request_count += 1
            try:
                async with self._session.post(url, json=dict(payload)) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if not isinstance(result, dict):
                        raise ValueError("Reward service must return a JSON object")
                    return result
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                permanent = 400 <= getattr(error, "status", 0) < 500
                if permanent or attempt == retries:
                    raise
                self.retry_count += 1
                await asyncio.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Reward request exhausted without a result")

    async def close_connection(self) -> None:
        """Close the request session on the event loop which owns it."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def close(self) -> None:
        """Terminate only the process group created by this client."""
        if self.process is not None:
            process = self.process
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            # Allow worker resource trackers to unlink shared memory after their parent exits.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
            self.process = None
        if self._log_handle is not None:
            self._resources.close()
            self._log_handle = None
        self.state = "closed"
