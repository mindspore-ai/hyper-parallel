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
"""Regression tests for bounded agent HTTP and process-output buffers."""
import asyncio
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from rl.agentic.codex import harness
from rl.agentic.core import gateway_transport as transport
from rl.agentic.ds_harness import gateway as deepseek


class TestAgentMemoryBounds(unittest.TestCase):
    """Check transport defaults and retain failure evidence without full output buffers."""

    def test_public_request_has_a_finite_default(self) -> None:
        """Unconfigured successful responses are rejected before JSON parsing."""
        with patch.object(transport, "DEFAULT_MAX_RESPONSE_BYTES", 16):
            with patch.object(transport.urllib.request, "urlopen", return_value=BytesIO(b"x" * 17)):
                with self.assertRaisesRegex(RuntimeError, "byte limit"):
                    transport.request_gateway_json("DeepSeek", "GET", "http://unused", None, 1)

    def test_codex_runtime_passes_default_and_explicit_limits(self) -> None:
        """Validation defaults reach the server even when absent from the user config."""
        for config, expected in (({}, 8 * 1024 * 1024), ({"max_request_bytes": 123}, 123)):
            with patch.object(harness.HarnessRuntime, "__init__", return_value=None) as initialize:
                harness.CodexRuntime(object(), config)
            self.assertEqual(initialize.call_args.kwargs["gateway_options"]["max_request_bytes"], expected)
            self.assertEqual(initialize.call_args.kwargs["gateway_options"]["max_response_bytes"], 32 * 1024 * 1024)

    def test_deepseek_rejects_oversized_request_before_read(self) -> None:
        """Declared oversized requests allocate no body and exact boundaries work."""
        handler = deepseek._Handler.__new__(deepseek._Handler)
        handler.server = SimpleNamespace(state=deepseek._State("http://unused", "model", 1, 16, 16))
        handler.headers = {"Content-Length": "17"}
        handler.rfile = BytesIO(b"x" * 17)
        with patch.object(handler.rfile, "read", wraps=handler.rfile.read) as read:
            with self.assertRaisesRegex(ValueError, "byte limit"):
                handler._request_json()
            read.assert_not_called()
        handler.headers = {"Content-Length": "7"}
        handler.rfile = BytesIO(b'{"a":1}')
        self.assertEqual(handler._request_json(), {"a": 1})
        for error in (False, True):
            response = (urllib.error.HTTPError("http://unused", 500, "failed", {}, BytesIO(b"x" * 17))
                        if error else BytesIO(b"x" * 17))
            options = {"side_effect": response} if error else {"return_value": response}
            with patch.object(deepseek.urllib.request, "urlopen", **options):
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    handler._backend_request({})

    def test_output_tail_is_bounded_but_early_events_are_preserved(self) -> None:
        """Large tool output cannot hide an early failure or final answer."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            answer = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}})
            failure = json.dumps({"type": "turn.failed", "message": "early failure"})
            for initial, fails in ((answer, False), (failure, True)):
                lines = iter([initial.encode() + b"\n"] + [b'{"type":"tool.output"}\n'] * 1000)

                async def readline():
                    return next(lines, b"")

                captured = harness._CapturedOutput(path)
                asyncio.run(harness._drain_codex_stream(
                    SimpleNamespace(readline=readline), path, captured, normalize_newlines=False,
                ))
                self.assertEqual(len(captured), 20)
                self.assertEqual(sum(1 for _ in captured), 1001)
                if fails:
                    with self.assertRaisesRegex(RuntimeError, "early failure"):
                        harness._interpret_codex_output(captured, [], 0, Path(directory))
                else:
                    self.assertEqual(harness._interpret_codex_output(captured, [], 0, Path(directory))[0], "done")

    def test_repeated_diagnostic_events_fail_with_bounded_evidence(self) -> None:
        """Pathological failure streams are rejected rather than buffered indefinitely."""
        events = (json.dumps({"type": "error", "message": "failed"}) for _ in range(1000))
        with self.assertRaisesRegex(RuntimeError, "event limit"):
            harness._classify_codex_events(events)
