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
"""Strict OpenAI Responses to vLLM Chat Completions translation for Codex."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from typing import Any, Iterable


_SUPPORTED_TOOL_TYPES = frozenset({"function", "local_shell", "shell"})
_CHAT_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}
    )


def _reasoning_text(item: dict[str, Any]) -> str:
    for key in ("content", "summary"):
        content = item.get(key)
        if isinstance(content, list):
            text = "\n".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("text")
            )
            if text:
                return text
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted.startswith("hyper-rl:"):
        try:
            return base64.urlsafe_b64decode(encrypted[9:].encode("ascii")).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return ""
    return ""


def _encrypt_reasoning(text: str) -> str:
    return "hyper-rl:" + base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def _shell_arguments(action: Any) -> str:
    if not isinstance(action, dict):
        raise ValueError("Codex local shell action must be an object")
    commands = action.get("commands")
    if (
        not isinstance(commands, list)
        or not commands
        or not all(isinstance(command, str) for command in commands)
    ):
        raise ValueError("Codex local shell action must contain string commands")
    arguments: dict[str, Any] = {"cmd": "\n".join(commands)}
    if action.get("timeout_ms") is not None:
        arguments["timeout_ms"] = int(action["timeout_ms"])
    return json.dumps(arguments, ensure_ascii=False)


def _shell_action(arguments: Any) -> dict[str, Any]:
    try:
        payload = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("vLLM shell arguments are not valid JSON") from error
    command = payload.get("cmd", payload.get("command"))
    if not isinstance(command, str) or not command:
        raise ValueError("vLLM shell call omitted its command")
    action: dict[str, Any] = {"commands": [command]}
    if payload.get("timeout_ms") is not None:
        action["timeout_ms"] = int(payload["timeout_ms"])
    return action


class CodexResponsesProtocol:
    """Translate the protocol while rejecting lossy or unknown tool shapes."""

    def transform_request(self, body: dict[str, Any], served_model: str) -> dict[str, Any]:
        """Convert one Codex Responses request into a non-streaming vLLM request."""
        if not isinstance(body, dict):
            raise ValueError("Responses request must be a JSON object")
        tools, namespace_aliases, _ = self._tools(body.get("tools", []))
        messages: list[dict[str, Any]] = []
        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append({"role": "system", "content": instructions})
        input_data = body.get("input", "")
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            messages.extend(self._input_messages(input_data, namespace_aliases))
        else:
            raise ValueError("Responses input must be text or an item list")
        request: dict[str, Any] = {
            "model": served_model,
            "messages": self._merge_system_messages(messages),
            "stream": False,
            "logprobs": True,
            "top_logprobs": 0,
            "return_token_ids": True,
        }
        for source, target in (
            ("max_output_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("parallel_tool_calls", "parallel_tool_calls"),
        ):
            if body.get(source) is not None:
                request[target] = body[source]
        if tools:
            request["tools"] = tools
            request["tool_choice"] = self._tool_choice(
                body.get("tool_choice", "auto"),
                namespace_aliases,
            )
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort") not in {None, "none"}:
            request["chat_template_kwargs"] = {"enable_thinking": True}
        return request

    def transform_response(
        self,
        response: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert one complete vLLM chat response to a Responses object."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("vLLM response omitted its first choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("vLLM response omitted its assistant message")
        _, _, alias_namespaces = self._tools(original_request.get("tools", []))
        output: list[dict[str, Any]] = []
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        if isinstance(reasoning, str) and reasoning:
            output.append(
                {
                    "id": f"rs_{uuid.uuid4().hex[:24]}",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": reasoning}],
                    "content": [{"type": "reasoning_text", "text": reasoning}],
                    "encrypted_content": _encrypt_reasoning(reasoning),
                }
            )
        content = _content_text(message.get("content"))
        if content:
            output.append(
                {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            )
        tool_calls = message.get("tool_calls", [])
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError("vLLM assistant tool_calls must be a list")
        for tool_call in tool_calls or []:
            output.append(self._response_tool_call(tool_call, alias_namespaces))
        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0
        output_tokens = int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
        return {
            "id": str(response.get("id", f"resp_{uuid.uuid4().hex[:24]}")),
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": str(original_request.get("model", response.get("model", ""))),
            "output": output,
            "usage": {
                "input_tokens": prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": prompt_tokens + output_tokens,
            },
            "error": None,
            "incomplete_details": None,
        }

    @staticmethod
    def stream_events(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Produce a deterministic synthetic Responses SSE stream."""
        base = {**result, "status": "in_progress", "output": []}
        yield {"type": "response.created", "response": base}
        for index, item in enumerate(result.get("output", [])):
            item_type = item.get("type")
            added_item = dict(item)
            added_item["status"] = "in_progress"
            if item_type == "message":
                added_item["content"] = []
            elif item_type == "function_call":
                added_item["arguments"] = ""
            yield {
                "type": "response.output_item.added",
                "output_index": index,
                "item": added_item,
            }
            if item_type == "message":
                for content_index, part in enumerate(item.get("content", [])):
                    empty_part = {**part, "text": ""}
                    yield {
                        "type": "response.content_part.added",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item.get("id"),
                        "part": empty_part,
                    }
                    yield {
                        "type": "response.output_text.delta",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item.get("id"),
                        "delta": part.get("text", ""),
                    }
                    yield {
                        "type": "response.output_text.done",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item.get("id"),
                        "text": part.get("text", ""),
                    }
                    yield {
                        "type": "response.content_part.done",
                        "output_index": index,
                        "content_index": content_index,
                        "item_id": item.get("id"),
                        "part": part,
                    }
            elif item_type == "function_call":
                yield {
                    "type": "response.function_call_arguments.delta",
                    "output_index": index,
                    "item_id": item.get("id"),
                    "delta": item.get("arguments", ""),
                }
                done_event = {
                    "type": "response.function_call_arguments.done",
                    "output_index": index,
                    "item_id": item.get("id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments", ""),
                }
                if item.get("namespace") is not None:
                    done_event["namespace"] = item["namespace"]
                yield done_event
            yield {"type": "response.output_item.done", "output_index": index, "item": item}
        yield {"type": "response.completed", "response": result}

    def _input_messages(
        self,
        items: list[Any],
        namespace_aliases: dict[tuple[str, str], str],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_calls: list[dict[str, Any]] = []
        pending_reasoning = ""

        def flush_calls() -> None:
            nonlocal pending_calls, pending_reasoning
            if pending_calls:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": pending_calls,
                }
                if pending_reasoning:
                    message["reasoning"] = pending_reasoning
                messages.append(message)
                pending_calls = []
                pending_reasoning = ""

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Every Responses input item must be an object")
            item_type = item.get("type")
            if item_type == "reasoning":
                flush_calls()
                pending_reasoning = _reasoning_text(item)
            elif item_type == "message":
                flush_calls()
                role = str(item.get("role", "user"))
                role = "system" if role == "developer" else role
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"Unsupported Responses message role: {role}")
                message = {"role": role, "content": _content_text(item.get("content"))}
                if role == "assistant" and pending_reasoning:
                    message["reasoning"] = pending_reasoning
                    pending_reasoning = ""
                messages.append(message)
            elif item_type in {"input_text", "output_text"}:
                flush_calls()
                messages.append({"role": "user", "content": str(item.get("text", ""))})
            elif item_type == "function_call":
                pending_calls.append(
                    self._chat_tool_call(item, namespace_aliases=namespace_aliases)
                )
            elif item_type in {"local_shell_call", "shell_call"}:
                pending_calls.append(self._chat_tool_call(item, local_shell=True))
            elif item_type in {
                "function_call_output",
                "local_shell_call_output",
                "shell_call_output",
            }:
                flush_calls()
                call_id = item.get("call_id", item.get("id"))
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("Responses tool output omitted call_id")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _content_text(item.get("output")),
                })
            else:
                raise ValueError(f"Unsupported Responses input item type: {item_type!r}")
        flush_calls()
        if pending_reasoning:
            messages.append({"role": "assistant", "content": "", "reasoning": pending_reasoning})
        return messages

    @staticmethod
    def _chat_tool_call(
        item: dict[str, Any],
        local_shell: bool = False,
        namespace_aliases: dict[tuple[str, str], str] | None = None,
    ) -> dict[str, Any]:
        call_id = item.get("call_id", item.get("id"))
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Responses tool call omitted call_id")
        name = "shell" if local_shell else item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Responses function call omitted its name")
        namespace = item.get("namespace")
        if namespace is not None:
            if local_shell or not isinstance(namespace, str) or not namespace:
                raise ValueError("Responses function call contains an invalid namespace")
            key = (namespace, name)
            if namespace_aliases is None or key not in namespace_aliases:
                raise ValueError(
                    "Responses function call references an unknown namespace tool: "
                    f"{namespace}.{name}"
                )
            name = namespace_aliases[key]
        arguments = (
            _shell_arguments(item.get("action"))
            if local_shell
            else item.get("arguments", "{}")
        )
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    @staticmethod
    def _merge_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system = [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        ]
        remaining = [message for message in messages if message.get("role") != "system"]
        return ([{"role": "system", "content": "\n\n".join(system)}] if system else []) + remaining

    @classmethod
    def _tools(
        cls,
        tools: Any,
    ) -> tuple[
        list[dict[str, Any]],
        dict[tuple[str, str], str],
        dict[str, tuple[str, str]],
    ]:
        if tools is None:
            return [], {}, {}
        if not isinstance(tools, list):
            raise ValueError("Responses tools must be a list")
        converted: list[dict[str, Any]] = []
        namespace_aliases: dict[tuple[str, str], str] = {}
        alias_namespaces: dict[str, tuple[str, str]] = {}
        used_names: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError("Every Responses tool must be an object")
            tool_type = str(tool.get("type", "function"))
            if tool_type == "namespace":
                namespace = tool.get("name")
                nested_tools = tool.get("tools")
                if not isinstance(namespace, str) or not namespace:
                    raise ValueError("Responses namespace tool omitted its name")
                if not isinstance(nested_tools, list) or not nested_tools:
                    raise ValueError("Responses namespace tool requires nested tools")
                for nested_tool in nested_tools:
                    if not isinstance(nested_tool, dict):
                        raise ValueError(
                            "Every Responses namespace member must be an object"
                        )
                    if nested_tool.get("type", "function") != "function":
                        raise ValueError(
                            "Only function members are supported inside Responses namespaces"
                        )
                    nested_name = nested_tool.get("name")
                    if not isinstance(nested_name, str) or not nested_name:
                        raise ValueError("Responses namespace member omitted its name")
                    key = (namespace, nested_name)
                    if key in namespace_aliases:
                        raise ValueError(
                            "Responses namespace contains a duplicate tool: "
                            f"{namespace}.{nested_name}"
                        )
                    alias = cls._namespace_alias(namespace, nested_name)
                    if alias in used_names:
                        raise ValueError(
                            f"Responses tools produced a duplicate name: {alias}"
                        )
                    namespace_aliases[key] = alias
                    alias_namespaces[alias] = key
                    used_names.add(alias)
                    converted.append(cls._function_tool(nested_tool, alias))
                continue
            if tool_type not in _SUPPORTED_TOOL_TYPES:
                raise ValueError(f"Unsupported Responses tool type: {tool_type}")
            name = "shell" if tool_type in {"local_shell", "shell"} else tool.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Responses tool omitted its name")
            if name in used_names:
                raise ValueError(f"Responses tools contain a duplicate name: {name}")
            used_names.add(name)
            converted.append(cls._function_tool(tool, name))
        return converted, namespace_aliases, alias_namespaces

    @staticmethod
    def _function_tool(tool: dict[str, Any], name: str) -> dict[str, Any]:
        function = {
            "name": name,
            "description": str(tool.get("description", "")),
            "parameters": tool.get("parameters", {"type": "object"}),
        }
        if tool.get("strict") is not None:
            function["strict"] = bool(tool["strict"])
        return {"type": "function", "function": function}

    @staticmethod
    def _namespace_alias(namespace: str, name: str) -> str:
        """Build one deterministic Chat-Completions-safe namespace alias."""
        readable = _CHAT_TOOL_NAME_PATTERN.sub("_", f"{namespace}__{name}")
        digest = hashlib.sha256(f"{namespace}\0{name}".encode("utf-8")).hexdigest()[:10]
        prefix = readable[: 64 - len(digest) - 2].rstrip("_-") or "tool"
        return f"{prefix}__{digest}"

    @staticmethod
    def _tool_choice(
        choice: Any,
        namespace_aliases: dict[tuple[str, str], str],
    ) -> Any:
        if isinstance(choice, str):
            return choice
        if not isinstance(choice, dict):
            raise ValueError("Responses tool_choice must be text or an object")
        choice_type = choice.get("type")
        if choice_type in {"local_shell", "shell"}:
            return {"type": "function", "function": {"name": "shell"}}
        if choice_type == "function":
            name = choice.get("name")
            namespace = choice.get("namespace")
            if namespace is not None:
                key = (namespace, name)
                if key not in namespace_aliases:
                    raise ValueError(
                        "Responses tool_choice references an unknown namespace tool: "
                        f"{namespace}.{name}"
                    )
                name = namespace_aliases[key]
            return {"type": "function", "function": {"name": name}}
        raise ValueError(f"Unsupported Responses tool_choice type: {choice_type!r}")

    @staticmethod
    def _response_tool_call(
        tool_call: Any,
        alias_namespaces: dict[str, tuple[str, str]],
    ) -> dict[str, Any]:
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
            raise ValueError("vLLM returned an invalid tool call")
        function = tool_call["function"]
        call_id = str(tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"))
        name = str(function.get("name", ""))
        arguments = function.get("arguments", "{}")
        if name == "shell":
            return {
                "id": f"sh_{uuid.uuid4().hex[:24]}",
                "type": "local_shell_call",
                "call_id": call_id,
                "status": "completed",
                "action": _shell_action(arguments),
            }
        result = {
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }
        namespace_identity = alias_namespaces.get(name)
        if namespace_identity is not None:
            namespace, nested_name = namespace_identity
            result["namespace"] = namespace
            result["name"] = nested_name
        return result
