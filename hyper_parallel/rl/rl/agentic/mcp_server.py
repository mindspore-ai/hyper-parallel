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
"""Expose an existing hyperparallel-RL ToolRegistry through MCP stdio without tool rewrites."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
from typing import Any, Mapping

from rl.agentic.tools import ToolRegistry


def _load_registry(factory_path: str, settings: Mapping[str, Any]) -> ToolRegistry:
    if ":" not in factory_path:
        raise ValueError("MCP registry factory must use 'module:function' syntax")
    module_name, attribute = factory_path.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise ValueError(f"MCP registry factory is not callable: {factory_path}")
    registry = factory(dict(settings))
    if not isinstance(registry, ToolRegistry):
        raise ValueError("MCP registry factory must return ToolRegistry")
    return registry


async def _call_tool(registry: ToolRegistry, params: Mapping[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("tools/call requires a name and object arguments")
    try:
        result = registry.get(name).handler(**dict(arguments))
        if inspect.isawaitable(result):
            result = await result
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return {"content": [{"type": "text", "text": text}], "isError": False}
    except Exception as error:  # pylint: disable=W0718
        return {
            "content": [{"type": "text", "text": f"Tool {name!r} failed: {error}"}],
            "isError": True,
        }


async def _dispatch(registry: ToolRegistry, request: Mapping[str, Any]) -> Any:
    method = request.get("method")
    if method == "initialize":
        params = request.get("params", {})
        requested_version = (
            params.get("protocolVersion") if isinstance(params, Mapping) else None
        )
        return {
            "protocolVersion": (
                requested_version
                if isinstance(requested_version, str) and requested_version
                else "2024-11-05"
            ),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "hyper-rl-tool-registry", "version": "1.0.0"},
        }
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": name,
                    "description": registry.get(name).description,
                    "inputSchema": registry.get(name).parameters,
                }
                for name in registry.names
            ]
        }
    if method == "tools/call":
        return await _call_tool(registry, request.get("params", {}))
    if method in {"notifications/initialized", "ping"}:
        return {} if method == "ping" else None
    raise ValueError(f"Unsupported MCP method: {method}")


async def _serve(registry: ToolRegistry) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
        if not line:
            return
        request: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("MCP request must be an object")
            result = await _dispatch(registry, request)
            if request.get("id") is None:
                continue
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as error:  # pylint: disable=W0718
            request_id = request.get("id") if isinstance(request, Mapping) else None
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(error)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main() -> None:
    """Start a registry-backed MCP stdio server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--settings-json", default="{}")
    arguments = parser.parse_args()
    settings = json.loads(arguments.settings_json)
    if not isinstance(settings, Mapping):
        raise ValueError("--settings-json must decode to an object")
    asyncio.run(_serve(_load_registry(arguments.factory, settings)))


if __name__ == "__main__":
    main()
