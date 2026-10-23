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
"""Call admission and cancellation must settle before weight publication."""
import asyncio
import json
import threading
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import AsyncMock, Mock, patch

import torch

from rl.agentic.core.program_runner import ProgramAgentRunner
from rl.async_training.vllm_server import CallAdmissionMiddleware, PartialRequestService, install_async_routes
from rl.roles.rollout.base import GenerationRequest, GenerationSettings
from rl.roles.rollout.client import HTTPGenerationEngine


class TestCallPublication(unittest.IsolatedAsyncioTestCase):
    """Publication waits for active calls, never for tool/episode completion."""

    def setUp(self) -> None:
        """Provide an asynchronous engine boundary without a device."""
        self.engine = SimpleNamespace(
            collective_rpc=AsyncMock(return_value=[{"version": 1}, {"version": 1}]),
            pause_generation=AsyncMock(), resume_generation=AsyncMock(),
        )
        self.service = PartialRequestService(self.engine, Mock())

    async def test_publication_closes_admission_and_waits_only_for_active_call(self):
        """Test publication closes admission and waits only for active call."""
        version = await self.service.enter_call("before")
        pause = asyncio.create_task(self.service.pause(drain=True))
        await asyncio.sleep(0)
        following = asyncio.create_task(self.service.enter_call("after"))
        await asyncio.sleep(0)
        self.assertFalse(pause.done())
        self.assertFalse(following.done())
        self.engine.pause_generation.assert_not_awaited()
        self.service.leave_call("before", version)
        self.assertEqual(await pause, {"status": "paused"})
        self.assertFalse(following.done())
        self.engine.collective_rpc.return_value = [{"version": 2}, {"version": 2}]
        await self.service.resume({"policy_version": 2})
        self.assertEqual(await following, 2)
        self.service.leave_call("after", 2)
        self.assertTrue(self.service.idle.is_set())

    async def test_wrong_publication_keeps_admission_closed(self):
        await self.service.pause(drain=True)
        with self.assertRaises(ValueError):
            await self.service.resume({"policy_version": 2})
        self.assertFalse(self.service.admission.is_set())
        self.assertTrue(self.service.closed)

    async def test_disagreeing_workers_cannot_admit_call(self):
        self.engine.collective_rpc.return_value = [{"version": 1}, {"version": 2}]
        with self.assertRaises(ValueError):
            await self.service.enter_call("bad")
        self.assertEqual(self.service.active, 0)

    async def test_uncertain_cancellation_prevents_weight_write(self):
        """Test uncertain cancellation prevents weight write."""
        async def failed_application(scope: Any, receive: Any, send: Any) -> None:
            """Model cancellation with unknown engine settlement."""
            raise asyncio.CancelledError()

        middleware = CallAdmissionMiddleware(failed_application, lambda _: self.service, 1)
        with self.assertRaises(asyncio.CancelledError):
            await middleware({"type": "http", "path": "/v1/chat/completions", "app": object()},
                             AsyncMock(), AsyncMock())
        self.assertEqual(self.service.active, 1)
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(self.service.pause(drain=True), timeout=.02)
        self.engine.pause_generation.assert_not_awaited()

    async def test_producer_shutdown_waits_for_program_cleanup(self):
        """Test producer shutdown waits for program cleanup."""
        entered, cleaned = threading.Event(), threading.Event()

        async def run_program() -> None:
            """Simulate an external tool with asynchronous cancellation cleanup."""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.01)
                cleaned.set()

        engine = SimpleNamespace(asynchronous=True, closed=False)
        settings = GenerationSettings(8, 1., 1., 0, True, 0, 2)
        runner = ProgramAgentRunner(lambda *_: SimpleNamespace(run=run_program), 1, settings, engine)
        running = asyncio.create_task(asyncio.to_thread(runner.rollout, [SimpleNamespace(prompt_id="0")], 1))
        started = await asyncio.to_thread(entered.wait, 1)
        engine.closed = True
        self.assertTrue(started)
        with self.assertRaisesRegex(RuntimeError, "draining"):
            await asyncio.wait_for(running, timeout=1)
        self.assertTrue(cleaned.is_set())

    async def test_installed_routes_guard_native_chat_and_stamp_resumed_version(self):
        """Test installed routes guard native chat and stamp resumed version."""
        app = SimpleNamespace(state=SimpleNamespace(engine_client=self.engine),
                              add_api_route=Mock(), add_middleware=Mock())

        async def native_chat() -> dict:
            """Represent a completed native model invocation."""
            return {"choices": []}

        def register(application: Any) -> None:
            """Represent the pinned runtime's native chat route registration."""
            application.add_api_route("/v1/chat/completions", native_chat, methods=["POST"])

        vllm = ModuleType("vllm")
        vllm.SamplingParams = Mock()
        entrypoints = ModuleType("vllm.entrypoints")
        serve = ModuleType("vllm.entrypoints.serve")
        serve.register_vllm_serve_api_routers = register
        entrypoints.serve = serve
        sampling = ModuleType("vllm.sampling_params")
        sampling.RequestOutputKind = SimpleNamespace(CUMULATIVE="cumulative")
        fastapi = ModuleType("fastapi")
        fastapi.Request = object  # Only the endpoint annotation belongs to the optional framework.
        modules = {"vllm": vllm, "vllm.entrypoints": entrypoints,
                   "vllm.entrypoints.serve": serve, "vllm.sampling_params": sampling, "fastapi": fastapi}
        with patch.dict("sys.modules", modules), patch(
                "rl.async_training.vllm_server.package_version",
                side_effect=lambda name: "0.23.0" if name == "vllm" else "0.23.0.post1"):
            install_async_routes()
            serve.register_vllm_serve_api_routers(app)
        routes = {call.args[0]: call.args[1] for call in app.add_api_route.call_args_list}
        self.assertEqual(set(routes), {"/v1/chat/completions", "/hyper_async/generate",
                                       "/hyper_async/pause", "/hyper_async/resume"})
        self.assertTrue(all(call.kwargs["methods"] == ["POST"] for call in app.add_api_route.call_args_list))

        async def native_application(scope: Any, receive: Any, send: Any) -> None:
            """Adapt the registered native endpoint to the ASGI response boundary."""
            del receive
            body = await routes[scope["path"]]()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": json.dumps(body).encode()})

        registration = app.add_middleware.call_args
        middleware = registration.args[0](native_application, **registration.kwargs)

        async def post(path: str, payload: dict | None = None) -> Any:
            """Execute real control endpoints or capture real middleware ASGI messages."""
            if path.startswith("/hyper_async/"):
                request = SimpleNamespace(app=app, url=SimpleNamespace(path=path),
                                          json=AsyncMock(return_value=payload))
                return await routes[path](request)
            messages = []
            await middleware({"type": "http", "path": path, "app": app},
                             AsyncMock(), AsyncMock(side_effect=messages.append))
            return messages[0]

        first = await post("/v1/chat/completions")
        self.assertEqual(first["status"], 200)
        self.assertEqual(dict(first["headers"])[b"x-hyper-policy-version"], b"1")
        self.assertEqual(await post("/hyper_async/pause", {"drain": True}), {"status": "paused"})
        following = asyncio.create_task(post("/v1/chat/completions"))
        await asyncio.sleep(.01)
        self.assertFalse(following.done())
        self.engine.collective_rpc.return_value = [{"version": 2}]
        self.assertEqual(await post("/hyper_async/resume", {"policy_version": 2}), {"status": "resumed"})
        self.assertEqual(dict((await following)["headers"])[b"x-hyper-policy-version"], b"2")

    async def generate_outputs(self, completions: list[tuple]) -> dict:
        """Feed cumulative engine output through the public generation and cleanup boundary."""
        outputs = []
        for index, (tokens, scores, reason) in enumerate(completions):
            probabilities = (None if scores is None else
                             [{token: SimpleNamespace(logprob=score)} for token, score in zip(tokens, scores)])
            completion = SimpleNamespace(token_ids=tokens, logprobs=probabilities, finish_reason=reason)
            outputs.append(SimpleNamespace(finished=index == len(completions) - 1, outputs=[completion]))
        collector = Mock()
        collector.get_nowait.side_effect = outputs
        self.engine.add_request = AsyncMock(return_value=collector)
        result = await self.service.generate({"request_id": "test", "prompt_token_ids": [4], "max_tokens": 3})
        collector.close.assert_called_once()
        return result

    async def test_empty_abort_keeps_already_emitted_prefix(self):
        result = await self.generate_outputs([([7, 8], [-.7, -.8], None), ([], None, "abort")])
        self.assertEqual((result["token_ids"], result["log_probs"], result["finish_reason"]),
                         ([7, 8], [-.7, -.8], "abort"))

    async def test_engine_must_not_rewrite_an_emitted_prefix(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            await self.generate_outputs([([7], [-.7], None), ([8], [-.8], "stop")])


class TestPartialGeneration(unittest.TestCase):
    """Check actual preserved prefixes, raw probabilities and cumulative budgets."""

    def test_repeated_interruptions_keep_original_probabilities(self):
        """Test repeated interruptions keep original probabilities."""
        for prefix_length in (1, 2048):
            with self.subTest(prefix_length=prefix_length):
                prefix = list(range(7, 7 + prefix_length))
                probabilities = [-.1 - index / 10000 for index in range(prefix_length)]
                client = Mock(operation_timeout=30)
                client.generate_segment.side_effect = [
                    {"token_ids": prefix, "log_probs": probabilities,
                     "finish_reason": "abort", "policy_version": 1},
                    {"token_ids": [], "log_probs": [], "finish_reason": "abort", "policy_version": 2},
                    {"token_ids": [8], "log_probs": [-.8], "finish_reason": "abort", "policy_version": 3},
                    {"token_ids": [9], "log_probs": [-.9], "finish_reason": "abort", "policy_version": 4},
                ]
                engine = HTTPGenerationEngine(client, 1)
                settings = GenerationSettings(prefix_length + 2, 1., 1., 0, True, 0, 2, collect_log_probs=True)
                result = engine.generate(GenerationRequest(torch.tensor([[4, 5]]), torch.ones(1, 2), settings))
                self.assertEqual(result.sequences.tolist(), [[4, 5] + prefix + [8, 9]])
                torch.testing.assert_close(result.rollout_log_probs, torch.tensor([probabilities + [-.8, -.9]]),
                                           rtol=0., atol=0.)
                calls = [call.args[0] for call in client.generate_segment.call_args_list]
                self.assertEqual([call["prompt_token_ids"] for call in calls],
                                 [[4, 5], [4, 5] + prefix, [4, 5] + prefix, [4, 5] + prefix + [8]])
                self.assertEqual([call["max_tokens"] for call in calls], [prefix_length + 2, 2, 2, 1])
                self.assertEqual(engine.segments, [
                    {"start": 0, "end": prefix_length, "policy_version": 1, "source_update": 0},
                    {"start": prefix_length, "end": prefix_length + 1, "policy_version": 3, "source_update": 2},
                    {"start": prefix_length + 1, "end": prefix_length + 2, "policy_version": 4, "source_update": 3},
                ])
                self.assertEqual(engine.interruptions, 4)
                self.assertEqual(result.finish_reasons, ("length",))

    def test_misaligned_partial_probabilities_fail(self):
        client = Mock(operation_timeout=30)
        client.generate_segment.return_value = {
            "token_ids": [7], "log_probs": [], "finish_reason": "stop", "policy_version": 1,
        }
        engine = HTTPGenerationEngine(client, 1)
        settings = GenerationSettings(3, 1., 1., 0, True, 0, 2, collect_log_probs=True)
        with self.assertRaisesRegex(ValueError, "payload"):
            engine.generate(GenerationRequest(torch.tensor([[4]]), torch.ones(1, 1), settings))

    def test_stop_at_budget_is_not_reported_as_length_truncation(self):
        """Test stop at budget is not reported as length truncation."""
        for reason, expected in (("stop", "stop"), ("abort", "length"), ("length", "length")):
            with self.subTest(reason=reason):
                client = Mock(operation_timeout=30)
                client.generate_segment.return_value = {
                    "token_ids": [7], "log_probs": [-.7], "finish_reason": reason, "policy_version": 1,
                }
                engine = HTTPGenerationEngine(client, 1)
                settings = GenerationSettings(1, 1., 1., 0, True, 0, 2, collect_log_probs=True)
                result = engine.generate(GenerationRequest(torch.tensor([[4]]), torch.ones(1, 1), settings))
                self.assertEqual(result.finish_reasons, (expected,))
