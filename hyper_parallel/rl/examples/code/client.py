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
"""Async client for the fixed Python-only SandboxFusion execution profile."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from urllib.parse import urlsplit

import aiohttp


@dataclass(frozen=True)
class ExecutionResult:
    """Candidate outcome; service and protocol failures never produce this value."""

    status: str
    stdout: str
    duration: float
    exit_code: int | None


def _validate_execution_details(stdout: object, duration: object, exit_code: object) -> None:
    """Reject malformed execution fields before classifying the outcome."""
    if (not isinstance(stdout, str) or not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration < 0
            or (exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)))):
        raise RuntimeError("SandboxFusion execution details violate the response contract")


def _parse_result(payload: object) -> ExecutionResult:
    """Validate the service response and classify the candidate execution."""
    if not isinstance(payload, dict):
        raise RuntimeError("SandboxFusion response must be a JSON object")
    if payload.get("status") not in {"Success", "Failed"}:
        raise RuntimeError(f"SandboxFusion service returned status={payload.get('status')!r}")
    result = payload.get("run_result")
    if payload.get("compile_result") is not None or not isinstance(result, dict):
        raise RuntimeError("SandboxFusion Python response has invalid execution details")
    status = result.get("status")
    stdout, duration, exit_code = result.get("stdout"), result.get("execution_time"), result.get("return_code")
    _validate_execution_details(stdout, duration, exit_code)
    if status == "Finished" and exit_code is not None:
        outcome = "success" if exit_code == 0 else "runtime_error"
    elif status in {"TimeLimitExceeded", "OutputLimitExceeded"}:
        outcome = "timeout" if status == "TimeLimitExceeded" else "output_limit"
    else:
        raise RuntimeError("SandboxFusion returned an unsupported or infrastructure execution status")
    if (payload["status"] == "Success") != (outcome == "success"):
        raise RuntimeError("SandboxFusion response contains inconsistent execution status")
    return ExecutionResult(outcome, stdout, float(duration), exit_code)


class SandboxFusionExecutor:
    """Execute remotely without retries; HTTP failures abort the episode."""

    def __init__(self, endpoint: str, run_timeout: float = 5.0, request_timeout: float = 300.0) -> None:
        """Configure candidate timeout and a separate HTTP deadline including queue time."""
        url = urlsplit(endpoint)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("SandboxFusion endpoint must be an HTTP URL without credentials/query/fragment")
        if (not math.isfinite(run_timeout) or run_timeout <= 0 or not math.isfinite(request_timeout)
                or request_timeout <= run_timeout + 2):
            raise ValueError("HTTP request_timeout must exceed positive run_timeout plus two-second cleanup")
        self.endpoint = endpoint.rstrip("/") + "/run_code"
        self.run_timeout = run_timeout
        self.request_timeout = request_timeout
        self._session = None
        self._closed = False

    async def run(self, code: str, stdin: str, *, request_id: str) -> ExecutionResult:
        """Submit one test; request ID is tracing only, not server deduplication."""
        if self._closed:
            raise RuntimeError("SandboxFusion executor is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout))
        try:
            async with self._session.post(
                self.endpoint, json={"language": "python", "code": code, "stdin": stdin,
                                     "run_timeout": self.run_timeout}, headers={"X-Request-ID": request_id},
            ) as response:
                response.raise_for_status()
                # JSON escaping can expand the fixed 1 MiB raw output budget sixfold.
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > 8 * 1024 * 1024:
                        raise RuntimeError("SandboxFusion response exceeds the protocol size limit")
                payload = json.loads(body)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            raise RuntimeError("SandboxFusion HTTP request or response decoding failed") from error
        return _parse_result(payload)

    async def close(self) -> None:
        """Close the transport; remote work remains bounded by its server timeout."""
        self._closed = True
        if self._session is not None:
            await self._session.close()
