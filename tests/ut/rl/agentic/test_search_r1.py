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
"""Real relay authorization and candidate-only workspace tests."""

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from examples.search_r1 import launcher
from examples.search_r1.agent import repair_request, validate_citations
from examples.search_r1.container_execution import (
    RelayHandler, SearchBudgetExceeded, SearchFeedback, classify_command_result, request_container_execution,
)
from examples.search_r1.launcher import training_command, verify_artifacts, verify_training
from examples.search_r1.workspace import build_search_workspace
from rl.agentic.core.program_runner import audit_training_rows


class EchoHandler(BaseHTTPRequestHandler):
    """Minimal real upstream used to verify authorized forwarding."""

    def do_POST(self):  # pylint: disable=invalid-name
        """Echo a bounded request body."""
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        """Suppress test transport logs."""


class SearchR1Tests(unittest.TestCase):
    """Verify data exposure and enforcement at the transport boundary."""

    def test_bash_syntax_error_is_a_trainable_model_command(self):
        """A malformed model command does not become an infrastructure failure."""
        output = (
            "Process exited with code 2\n"
            "/usr/bin/bash: -c: line 1: unexpected EOF while looking for matching quote\n"
            "/usr/bin/bash: -c: line 2: syntax error: unexpected end of file\n"
        )
        result = classify_command_result("rg -i -F -- 'unfinished articles", output)
        self.assertEqual(result, {
            "failure_origin": "model", "failure_reason": "invalid_command", "trainable": True,
        })

    def test_host_and_standalone_relay_are_standard_library_only(self):
        """Host launcher and mounted relay must not load the training stack."""
        script = (
            "import importlib.util, sys; "
            "from examples.search_r1 import launcher, container_execution; "
            "spec = importlib.util.spec_from_file_location('isolated_relay', container_execution.__file__); "
            "relay = importlib.util.module_from_spec(spec); spec.loader.exec_module(relay); "
            "blocked = set(sys.modules) & {'torch', 'mindspore', 'hyper_parallel', 'rl'}; "
            "print(sorted(blocked)); sys.exit(bool(blocked))"
        )
        root = Path(launcher.__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, f"Expected exit=0, got={result.returncode}: {result.stderr}")
        self.assertEqual(result.stdout.strip(), "[]", f"Expected no heavy imports, got={result.stdout!r}")

    def test_host_direct_entry_help(self):
        """The documented direct-script launch remains usable without package imports."""
        result = subprocess.run(
            [sys.executable, str(Path(launcher.__file__).resolve()), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, f"Expected exit=0, got={result.returncode}: {result.stderr}")
        self.assertIn("--devices", result.stdout)

    def test_training_entry_and_optimizer_gate(self):
        """The launcher uses the real trainer and rejects rollout-only or zero-gradient runs."""
        command = training_command(2)
        self.assertIn("train_rl.py", command)
        self.assertIn("--master_addr=127.0.0.1", command)
        self.assertNotIn("--standalone", command)
        self.assertIn("examples/search_r1/configs/qwen3_4b_search_r1.yaml", command)
        line = "step=2 | policy/version=2 train/global_step=2 train/gradient_norm=0.01 train/optimizer_steps=1"
        verify_training(line, 2)
        invalid_logs = ("rollout completed", line.replace("0.01", "0"),
                        line.replace("policy/version=2", "policy/version=1"))
        for invalid in invalid_logs:
            with self.subTest(log=invalid), self.assertRaises(RuntimeError):
                verify_training(invalid, 2)

    def test_strict_training_gate_accepts_current_validation_metrics(self):
        """Accept scored validation and reload evidence, and reject missing counts."""
        line = (
            "step=2 | policy/version=2 train/global_step=2 "
            "train/gradient_norm=0.01 train/optimizer_steps=1 "
            "training/pre_update_exact_valid=1 training/pre_update_mismatch_count=0 "
            "validation/accuracy=0.4 validation/total=8"
        )
        verify_training(line + "\nverified checkpoint reload", 2, strict=True)
        with self.assertRaisesRegex(RuntimeError, "independent evaluation"):
            verify_training(line.replace("validation/total=8", "validation/total=0")
                            + "\nverified checkpoint reload", 2, strict=True)

    def test_audit_rejects_wrong_context(self):
        """The audit fails before training when a reconstructed prefix replaces the real prompt."""
        row = SimpleNamespace(metadata={"gateway_record": {"response": {"choices": [{
            "prompt_token_ids": [1, 2], "token_ids": [3], "logprobs": {"content": [{"logprob": -0.5}]},
        }]}}}, token_ids=Mock())
        row.token_ids.tolist.return_value = [1, 9, 3]
        with self.assertRaisesRegex(ValueError, "context differs"):
            audit_training_rows((row,))

    def test_artifact_gate_requires_isolation_and_context(self):
        """A successful training log alone is insufficient for Search-R1 acceptance."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "No isolated"):
                verify_artifacts(output)
            artifact = output / "jobs" / "episode"
            artifact.mkdir(parents=True)
            (artifact / "execution-request.json").write_text("{}")
            (artifact / "isolation.json").write_text('{"hidden_data": true}')
            (artifact / "training-audit.json").write_text(json.dumps({
                "model_calls": 2, "training_context_mismatches": 0, "raw_prefix_mismatches": 1,
            }))
            (artifact / "search-outcome.json").write_text(json.dumps({
                "citation_repaired": True, "task_outcome": "completed",
            }))
            (artifact / "tool-error-summary.json").write_text(json.dumps({
                "complete": True, "infra_reward_contamination_count": 0,
            }))
            self.assertEqual(verify_artifacts(output)["citation_repairs"], 1)
            (artifact / "isolation.json").write_text('{"hidden_data": false}')
            with self.assertRaisesRegex(RuntimeError, "Isolation failed"):
                verify_artifacts(output)

    def test_artifact_gate_accepts_untrainable_infra_without_training_audit(self):
        """A proven unscored infra retry has no model context to audit."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            artifact = output / "jobs" / "infra-retry"
            artifact.mkdir(parents=True)
            (artifact / "execution-request.json").write_text("{}")
            (artifact / "isolation.json").write_text('{"hidden_data": true}')
            (artifact / "search-outcome.json").write_text(json.dumps({
                "failure_origin": "infrastructure", "failure_reason": "execution_environment",
                "trainable": False, "reward_evaluated": False, "reward": None,
                "training_rows_emitted": 0,
            }))
            (artifact / "tool-error-summary.json").write_text(json.dumps({
                "complete": True, "infra_reward_contamination_count": 0,
            }))
            result = verify_artifacts(output)
            self.assertEqual(result["infrastructure_failures"], 1)
            self.assertEqual(result["trainable_episodes"], 0)
            self.assertEqual(result["tool_protocol"]["by_phase"]["infrastructure"]["model_calls"], 0)

            (artifact / "search-outcome.json").write_text(json.dumps({
                "failure_origin": "infrastructure", "trainable": False,
                "reward_evaluated": True, "reward": 0, "training_rows_emitted": 1,
            }))
            with self.assertRaisesRegex(RuntimeError, "Invalid untrainable"):
                verify_artifacts(output)

    def test_transport_failure_is_terminal_not_budget_failure(self):
        """A failed upstream cannot be retried until it becomes a zero-reward task failure."""
        relay = ThreadingHTTPServer(("127.0.0.1", 0), RelayHandler)
        relay.session_id = "current"
        connection_factory = Mock(side_effect=ConnectionRefusedError("unavailable"))
        relay.connection_factory = connection_factory
        thread = threading.Thread(target=relay.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                relay.search_policy = SearchFeedback(Path(temporary), max_calls=2)
                for _ in range(4):
                    connection = http.client.HTTPConnection(*relay.server_address, timeout=3)
                    connection.request("POST", "/responses", b'{"input": []}',
                                       {"Authorization": "Bearer current"})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 502)
                    self.assertEqual(response.getheader("X-Search-Terminal"), "upstream_error")
                    response.read()
                    connection.close()
                self.assertEqual(connection_factory.call_count, 1)
                terminal = json.loads((Path(temporary) / "search-terminal.json").read_text())
                self.assertEqual(terminal["reason"], "upstream_error")
        finally:
            relay.shutdown()
            relay.server_close()
            thread.join()

    def test_execution_uses_episode_gateway_and_budget(self):
        """Container loopback stays fixed while the trusted upstream follows the training rank."""
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            workspace = artifact / "workspace"
            workspace.mkdir()
            (workspace / "articles").mkdir()
            home = artifact / "codex-home"
            home.mkdir()
            environment = {"OPENAI_API_KEY": "session", "HYPER_CODEX_GATEWAY_URL": "http://127.0.0.1:8413",
                           "HYPER_CODEX_MAX_CALLS": "7"}
            with patch.dict("os.environ", {"HYPER_SEARCH_CONTAINER_BROKER": "1", "PHASE_UID": "1", "PHASE_GID": "1"}), \
                    patch("examples.search_r1.container_execution.UnixServer") as server_class, \
                    patch("examples.search_r1.container_execution.threading.Thread"), \
                    patch("examples.search_r1.container_execution.os.chown"), \
                    patch("examples.search_r1.container_execution._BRIDGES", []):
                request_container_execution(["codex", "exec", "--", "question"], workspace, home, environment)
                server = server_class.return_value
                upstream = server.connection_factory()
                self.assertEqual(upstream.port, 8413)
                self.assertEqual(server.search_policy.max_calls, 6)
                request = json.loads((artifact / "execution-request.json").read_text())
                self.assertIn('model_providers.hyper_rl.base_url="http://127.0.0.1:8200"', request["command"])

    def test_repair_is_a_new_action_without_tools(self):
        """Correction keeps original history and answer; it never rewrites sampled tokens."""
        original = {"input": [{"role": "user", "content": "question"}], "tools": ["shell"],
                    "stream": True, "previous_response_id": "old"}
        result = repair_request(original, "ANSWER: X\nSOURCES: bad.txt:2", ["unknown_reference"])
        self.assertEqual(len(original["input"]), 1)
        self.assertEqual(result["input"][-2]["content"], "ANSWER: X\nSOURCES: bad.txt:2")
        self.assertEqual(result["tools"], [])
        self.assertFalse(result["stream"])
        self.assertNotIn("previous_response_id", result)
        self.assertTrue(all(item["type"] == "message" for item in result["input"][-2:]))

    def test_workspace_has_candidates_without_labels(self):
        """Only current candidate content is exported, with safe filenames."""
        prompt = SimpleNamespace(metadata={
            "context_json": json.dumps({"title": ["../outside"], "sentences": [["", "Evidence."]]}),
            "supporting_facts_json": "secret-label", "answer": "secret-answer",
        })
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            build_search_workspace(prompt, workspace)
            files = list((workspace / "articles").glob("*.txt"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "000.txt")
            self.assertEqual(files[0].read_text(),
                             "FILE: articles/000.txt\nTITLE: ../outside\n"
                             "[REF: ../outside:0] \n[REF: ../outside:1] Evidence.\n")
            rendered = "".join(path.read_text() for path in workspace.rglob("*") if path.is_file())
            self.assertNotIn("secret-label", rendered)
            self.assertNotIn("secret-answer", rendered)
            self.assertEqual({path.name for path in workspace.iterdir()}, {"articles", "articles.json"})

    def test_relay_enforces_session_and_route(self):
        """Real HTTP requests reach upstream only for the authorized session/route."""
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), EchoHandler)
        relay = ThreadingHTTPServer(("127.0.0.1", 0), RelayHandler)
        relay.session_id = "current"
        relay.connection_factory = lambda: http.client.HTTPConnection(*upstream.server_address, timeout=3)
        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (upstream, relay)]
        for thread in threads:
            thread.start()
        try:
            for method, route, token, status in (
                ("POST", "/responses", "current", 200),
                ("POST", "/responses", "other", 403),
                ("POST", "/internal/sessions", "current", 403),
                ("GET", "/internal/sessions/current", "current", 403),
                ("DELETE", "/internal/sessions/current", "current", 403),
                ("POST", "/responses?redirect=/internal/sessions", "current", 403),
            ):
                with self.subTest(method=method, route=route, token=token):
                    connection = http.client.HTTPConnection(*relay.server_address, timeout=3)
                    connection.request(method, route, b"{}", {"Authorization": "Bearer " + token})
                    response = connection.getresponse()
                    self.assertEqual(response.status, status)
                    response.read()
                    connection.close()
            with tempfile.TemporaryDirectory() as temporary:
                relay.search_policy = SearchFeedback(Path(temporary), max_calls=2)
                for expected_status in (200, 200, 409):
                    connection = http.client.HTTPConnection(*relay.server_address, timeout=3)
                    connection.request("POST", "/responses", b'{"input": []}',
                                       {"Authorization": "Bearer current"})
                    response = connection.getresponse()
                    self.assertEqual(response.status, expected_status)
                    if expected_status == 409:
                        self.assertEqual(response.getheader("X-Search-Terminal"), "budget_exhausted")
                    response.read()
                    connection.close()
        finally:
            for server in (upstream, relay):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_missing_broker_fails_closed(self):
        """A standalone training invocation cannot fall back to a local shell."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "host container broker"):
                request_container_execution([], Path("/unused"), Path("/unused"), {})

    def test_citations_require_real_visible_reference(self):
        """Filename, wrong ID and unseen references do not count as valid citations."""
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            prompt = SimpleNamespace(metadata={"context_json": [["A–B", ["Evidence."]]]})
            build_search_workspace(prompt, workspace)
            visible = ["[REF: A–B:0] Evidence."]
            self.assertTrue(validate_citations("ANSWER: X\nSOURCES: A–B:0", workspace, visible)["valid"])
            for source in ("000.txt:0", "A-B:0", "A–B:1", "A–B:0; bogus:0"):
                with self.subTest(source=source):
                    self.assertFalse(validate_citations("ANSWER: X\nSOURCES: " + source, workspace, visible)["valid"])
            self.assertFalse(validate_citations("ANSWER: X\nSOURCES: A–B:0", workspace, [])["valid"])
            self.assertFalse(validate_citations("ANSWER: UNKNOWN\nSOURCES: NONE", workspace, [])["valid"])

    def test_feedback_preserves_tool_output_and_stops_budget(self):
        """Feedback precedes inference and is separate from unchanged tool observations."""
        with tempfile.TemporaryDirectory() as temporary:
            policy = SearchFeedback(Path(temporary), max_calls=2)
            items = [
                {"type": "function_call", "call_id": "one", "arguments": json.dumps({"cmd": "rg X articles"})},
                {"type": "function_call_output", "call_id": "one",
                 "output": "Process exited with code 1\nFinal output:"},
            ]
            result = json.loads(policy.prepare(json.dumps({"input": items}).encode()))
            self.assertEqual(result["input"][:2], items)
            self.assertIn("Do not repeat", result["input"][-1]["content"])
            self.assertEqual(result["input"][-1]["type"], "message")
            result = json.loads(policy.prepare(json.dumps({"input": items}).encode()))
            self.assertIn("FINAL MODEL CALL", result["input"][-1]["content"])
            self.assertEqual(result["tools"], [])
            self.assertEqual(result["tool_choice"], "none")
            self.assertEqual(policy.failed_commands[("/work", "rg X articles")], 1)
            with self.assertRaisesRegex(SearchBudgetExceeded, "budget_exhausted"):
                policy.prepare(b'{"input": []}')
            self.assertTrue((Path(temporary) / "search-terminal.json").exists())

    def test_recovery_survives_requests_without_new_output(self):
        """Intermediate model messages cannot erase the pending recovery instruction."""
        with tempfile.TemporaryDirectory() as temporary:
            policy = SearchFeedback(Path(temporary))
            items = [
                {"type": "function_call", "call_id": "1",
                 "arguments": json.dumps({"cmd": "rg -F -- 'Alpha–Beta Commission' articles"})},
                {"type": "function_call_output", "call_id": "1", "output": "Process exited with code 1"},
            ]
            first = json.loads(policy.prepare(json.dumps({"input": items}).encode()))
            second = json.loads(policy.prepare(json.dumps({"input": items}).encode()))
            self.assertEqual(first["input"][-1], second["input"][-1])
            self.assertIn("-- Alpha articles", second["input"][-1]["content"])
            items.extend([
                {"type": "function_call_output", "call_id": "2", "output": "Process exited with code 0\nEvidence"},
            ])
            third = json.loads(policy.prepare(json.dumps({"input": items}).encode()))
            self.assertEqual(third["input"], items)

    def test_repeated_failed_commands_stop(self):
        """Distinct observations of the same failed command terminate bounded retries."""
        with tempfile.TemporaryDirectory() as temporary:
            policy = SearchFeedback(Path(temporary))
            for index in range(3):
                items = [
                    {"type": "function_call", "call_id": str(index), "arguments": '{"cmd":"rg X articles"}'},
                    {"type": "function_call_output", "call_id": str(index), "output": "Process exited with code 1"},
                ]
                if index < 2:
                    policy.prepare(json.dumps({"input": items}).encode())
                else:
                    with self.assertRaisesRegex(SearchBudgetExceeded, "repeated_tool_failure"):
                        policy.prepare(json.dumps({"input": items}).encode())
