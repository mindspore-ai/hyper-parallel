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
"""Codex tool narrowing preserves default requests and actual sampled context."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rl.agentic.codex.gateway import _Handler, _State
from rl.agentic.codex.protocol import CodexResponsesProtocol


def request() -> dict:
    """Return a task with optional tools and non-task harness messages."""
    return {"instructions": "Task instructions", "input": [
        {"type": "message", "role": "developer", "content": "<skills_instructions>skills"},
        {"type": "message", "role": "user", "content": "<environment_context>environment"},
        {"type": "message", "role": "user", "content": "What is 2 + 2?"},
    ], "tools": [{"type": "function", "name": name, "parameters": {"type": "object"}}
                 for name in ("exec_command", "write_stdin", "view_image", "create_goal")]}


class TestToolAllowlist(unittest.TestCase):
    """Validate filtering before sampling and reversible retained tool calls."""

    def test_default_and_explicit_filter_preserve_original_request(self) -> None:
        protocol = CodexResponsesProtocol()
        original = request()
        untouched = deepcopy(original)
        full = protocol.transform_request(original, "qwen3")
        narrowed = protocol.transform_request(original, "qwen3", allowed_tools=["exec_command", "write_stdin"])
        self.assertEqual(len(full["tools"]), 4)
        self.assertEqual([t["function"]["name"] for t in narrowed["tools"]], ["exec_command", "write_stdin"])
        self.assertEqual(full["messages"], narrowed["messages"])
        self.assertEqual(original, untouched)

    def test_invalid_unknown_and_forced_excluded_tools_fail(self) -> None:
        """Reject invalid allowlists and choices requiring an excluded tool."""
        protocol = CodexResponsesProtocol()
        for value in ("exec_command", [""], [" x"], [1], [["x"]], ["exec_command", "exec_command"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.transform_request(request(), "qwen3", allowed_tools=value)
        with self.assertRaisesRegex(ValueError, "not advertised"):
            protocol.transform_request(request(), "qwen3", allowed_tools=["missing"])
        original = request()
        original["tool_choice"] = {"type": "function", "name": "view_image"}
        with self.assertRaisesRegex(ValueError, "excluded"):
            protocol.transform_request(original, "qwen3", allowed_tools=["exec_command"])
        original["tool_choice"] = "required"
        with self.assertRaisesRegex(ValueError, "excluded"):
            protocol.transform_request(original, "qwen3", allowed_tools=[])
        original["tool_choice"] = "auto"
        self.assertNotIn("tools", protocol.transform_request(original, "qwen3", allowed_tools=[]))

    def test_compaction_can_preserve_environment_and_permissions(self) -> None:
        """File tasks retain execution context and observations without unrelated skills."""
        original = request()
        original["input"][1]["content"] = "<environment_context><cwd>/work</cwd><shell>bash</shell>"
        original["input"].append({"type": "message", "role": "developer",
                                  "content": "<permissions instructions>No network access."})
        original["input"].extend([
            {"type": "function_call", "name": "exec_command", "call_id": "c", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "c", "output": "articles"},
        ])
        untouched = deepcopy(original)
        for preserve in (False, True):
            transformed = CodexResponsesProtocol().transform_request(
                original, "qwen3", compact_harness_context=True, preserve_environment_context=preserve)
            text = str(transformed["messages"])
            self.assertEqual("/work" in text, preserve)
            self.assertEqual("No network access" in text, preserve)
            self.assertNotIn("<skills_instructions>", text)
            self.assertEqual(transformed["messages"][-1]["content"], "articles")
            self.assertEqual(transformed["messages"][-2]["tool_calls"][0]["function"]["name"], "exec_command")
        self.assertEqual(original, untouched)

    def test_explicit_no_tools_request_does_not_require_allowlist_declarations(self) -> None:
        """Final answers and repairs may disable tools without weakening ordinary validation."""
        original = request()
        original.update(tools=[], tool_choice="none")
        transformed = CodexResponsesProtocol().transform_request(
            original, "qwen3", allowed_tools=["exec_command", "write_stdin"])
        self.assertNotIn("tools", transformed)
        for choice in ("auto", "required"):
            original["tool_choice"] = choice
            with self.subTest(choice=choice), self.assertRaisesRegex(ValueError, "not advertised"):
                CodexResponsesProtocol().transform_request(original, "qwen3", allowed_tools=["exec_command"])

    def test_namespaced_tool_round_trip(self) -> None:
        """Keep namespace identity through request and response translation."""
        protocol = CodexResponsesProtocol()
        original = request()
        original["tools"] = [{"type": "namespace", "name": "math", "tools": [
            {"type": "function", "name": "calculate"}, {"type": "function", "name": "plot"}]}]
        transformed = protocol.transform_request(original, "qwen3", allowed_tools=["math.calculate"])
        self.assertEqual(len(transformed["tools"]), 1)
        name = transformed["tools"][0]["function"]["name"]
        response = protocol.transform_response({"choices": [{"message": {"tool_calls": [
            {"id": "c", "type": "function", "function": {"name": name, "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}]}, original)
        self.assertEqual(response["output"][0]["name"], "calculate")
        self.assertEqual(response["output"][0]["namespace"], "math")

    def test_registration_and_gateway_apply_session_allowlist(self) -> None:
        """Apply the registered session allowlist to gateway model requests."""
        state = _State("http://unused", "qwen3", 2)
        state.register("s", {"policy_version": 0, "max_completions": 2, "allowed_tools": ["exec_command"],
                             "compact_harness_context": True, "preserve_environment_context": True})
        session = state.sessions["s"]
        handler = object.__new__(_Handler)
        handler.server = SimpleNamespace(state=state)
        handler._model_completion = Mock(return_value=(
            {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]},
            {"failure_origin": None, "trainable": True}, {}))
        handler._json = Mock()
        handler._respond_to_session("s", session, request())
        transformed = handler._model_completion.call_args.args[2]
        self.assertEqual([t["function"]["name"] for t in transformed["tools"]], ["exec_command"])
        self.assertIn("<environment_context>", str(transformed["messages"]))
        self.assertNotIn("<skills_instructions>", str(transformed["messages"]))
        handler._json.assert_called_once()
        with self.assertRaises(ValueError):
            state.register("bad", {"policy_version": 0, "max_completions": 2, "allowed_tools": "exec_command"})
