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
"""Standard-library HTTP transport shared by harnesses and host executors."""

from __future__ import annotations

import http.client
import json
from typing import Any, Mapping
import urllib.error
import urllib.request


class GatewayTransportError(RuntimeError):
    """Identify a disconnected management request without treating HTTP rejection as retryable."""


def request_gateway_json(
    label: str,
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
    admin_token: str | None = None,
    max_response_bytes: int | None = None,
) -> dict[str, Any]:
    """Exchange one JSON object with an external harness gateway."""
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    open_request = (urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                    if admin_token is not None else urllib.request.urlopen)
    try:
        with open_request(request, timeout=timeout) as response:
            content = response.read() if max_response_bytes is None else response.read(max_response_bytes + 1)
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"{label} gateway HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise GatewayTransportError(f"{label} gateway request failed: {error.reason}") from error
    except (OSError, http.client.HTTPException) as error:
        raise GatewayTransportError(f"{label} gateway connection failed: {error}") from error
    if max_response_bytes is not None and len(content) > max_response_bytes:
        raise RuntimeError(f"{label} gateway response exceeded its byte limit")
    try:
        decoded = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} gateway returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{label} gateway returned a non-object response")
    return decoded
