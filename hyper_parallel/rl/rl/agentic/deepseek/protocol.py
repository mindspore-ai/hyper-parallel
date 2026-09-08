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
"""DeepSeek Harness Chat Completions protocol adapter for the shared vLLM."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from typing import Any


class DeepSeekChatProtocol:
    """Preserve DeepSeek chat/tool semantics while collecting exact token evidence."""

    def transform_request(
        self, body: dict[str, Any], served_model: str
    ) -> dict[str, Any]:
        """Convert a streaming Harness request into one non-streaming vLLM request."""
        if not isinstance(body, dict):
            raise ValueError("DeepSeek request must be a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("DeepSeek request requires a non-empty messages list")
        request = dict(body)
        request.update(
            {
                "model": served_model,
                "stream": False,
                "logprobs": True,
                "top_logprobs": 0,
                "return_token_ids": True,
            }
        )
        request.pop("stream_options", None)
        # vLLM's OpenAI server does not define DeepSeek's provider-only effort field.
        effort = request.pop("reasoning_effort", None)
        if effort == "off":
            chat_template_kwargs = request.get("chat_template_kwargs", {})
            if not isinstance(chat_template_kwargs, dict):
                raise ValueError("DeepSeek chat_template_kwargs must be a mapping")
            request["chat_template_kwargs"] = {
                **chat_template_kwargs,
                "enable_thinking": False,
            }
        elif effort in {"high", "max"}:
            chat_template_kwargs = request.get("chat_template_kwargs", {})
            if not isinstance(chat_template_kwargs, dict):
                raise ValueError("DeepSeek chat_template_kwargs must be a mapping")
            request["chat_template_kwargs"] = {
                **chat_template_kwargs,
                "enable_thinking": True,
            }
        elif effort is not None:
            raise ValueError(f"Unsupported DeepSeek reasoning_effort: {effort}")
        return request

    def stream_events(self, response: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Synthesize the OpenAI-compatible SSE chunks consumed by DeepSeek Harness."""
        choices = response.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise ValueError("vLLM response omitted its first choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("vLLM response omitted its assistant message")
        response_id = str(response.get("id") or f"chatcmpl-{uuid.uuid4().hex}")
        model = str(response.get("model") or "policy")
        created = int(response.get("created") or time.time())
        delta: dict[str, Any] = {"role": "assistant"}
        for name in ("content", "reasoning_content"):
            value = message.get(name)
            if value is not None:
                delta[name] = value
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            delta["tool_calls"] = [
                {**tool_call, "index": index}
                if isinstance(tool_call, dict)
                else tool_call
                for index, tool_call in enumerate(tool_calls)
            ]
        yield self._chunk(response_id, model, created, delta, None)
        finish_reason = choice.get("finish_reason")
        if finish_reason is None:
            finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        final = self._chunk(response_id, model, created, {}, str(finish_reason))
        usage = response.get("usage")
        if isinstance(usage, dict):
            final["usage"] = dict(usage)
        yield final

    @staticmethod
    def _chunk(
        response_id: str,
        model: str,
        created: int,
        delta: dict[str, Any],
        finish_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
