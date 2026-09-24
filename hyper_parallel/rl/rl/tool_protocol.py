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
"""Read-only Hermes evidence and fail-closed rollout attribution."""

from contextvars import ContextVar
from functools import wraps
import json
import re
from typing import Any, Iterable, Mapping


_CAPTURE = ContextVar("hyper_tool_protocol", default=None)
_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)


def validate_trainability(trajectories: Iterable[Any]) -> None:
    """Reject an entire update, never filter members of a GRPO group."""
    for trajectory in trajectories:
        metadata = trajectory.metadata
        if not isinstance(metadata, Mapping):
            raise ValueError("Trajectory metadata must be a mapping")
        history = metadata.get("gateway_records", [])
        if not isinstance(history, (list, tuple)):
            raise ValueError("Gateway history must be a sequence of completion records")
        records = [metadata.get("gateway_record", {}), *history]
        if not all(isinstance(record, Mapping) for record in records):
            raise ValueError("Gateway records must be mappings")
        evidence = [metadata, *(record.get("metadata", {}) for record in records)]
        if any(not isinstance(item, Mapping) or item.get("trainable", True) is not True
               or item.get("failure_origin") not in (None, "model") for item in evidence):
            raise ValueError("Untrainable trajectory: " + str({
                key: metadata.get(key) for key in (
                    "episode_id", "failure_origin", "failure_reason", "trainable",
                )
            }))


def _response_message(response: dict) -> tuple[dict, dict]:
    """Require the single-choice response shape used by one harness model call."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Tool attribution requires exactly one completion choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Tool attribution requires an assistant message")
    return choice, message


def inspect_tool_response(response: dict) -> dict:
    """Attribute only failures supported by immutable engine/parser evidence."""
    choice, message = _response_message(response)
    evidence = response.get("hyper_tool_protocol")
    suspicious = "<tool_call>" in (message.get("content") or "") or bool(message.get("tool_calls"))
    result = {"failure_origin": None, "failure_reason": None, "trainable": True}
    if not evidence:
        if suspicious:
            result.update(failure_origin="unknown", failure_reason="missing_parser_evidence", trainable=False)
        return result
    if not isinstance(evidence, list) or len(evidence) != 1 or not isinstance(evidence[0], dict):
        return {"failure_origin": "unknown", "failure_reason": "ambiguous_parser_evidence", "trainable": False}
    item = evidence[0]
    error = _validate_source_evidence(item, choice)
    if error is not None:
        return error
    blocks = _BLOCK.findall(item["parser_input"])
    if not blocks:
        if message.get("tool_calls"):
            return {"failure_origin": "unknown", "failure_reason": "missing_raw_tool_call", "trainable": False}
        return result
    calls, error = _raw_calls(blocks)
    if error is not None:
        return error
    parsed_result = item.get("parser_result")
    if not isinstance(parsed_result, dict):
        return {"failure_origin": "unknown", "failure_reason": "missing_parser_result", "trainable": False}
    if _parsed_calls(message.get("tool_calls")) != calls or _parsed_calls(parsed_result.get("tool_calls")) != calls:
        result.update(failure_origin="infrastructure", failure_reason="parser_result_mismatch", trainable=False)
    return result


def _validate_source_evidence(item: dict, choice: dict) -> Any:
    """Require agreement between immutable action IDs and each recorded tool text."""
    if not all(isinstance(item.get(key), str) for key in ("parser_input", "decoded_tokens", "engine_text")):
        return {"failure_origin": "unknown", "failure_reason": "incomplete_parser_evidence", "trainable": False}
    blocks = _BLOCK.findall(item["parser_input"])
    if (not isinstance(item.get("token_ids"), list) or not item["token_ids"]
            or item["token_ids"] != choice.get("token_ids")
            or blocks != _BLOCK.findall(item["decoded_tokens"])
            or blocks != _BLOCK.findall(item["engine_text"])):
        return {"failure_origin": "unknown", "failure_reason": "parser_input_mismatch", "trainable": False}
    return None


def _raw_calls(blocks: list[str]) -> tuple[list, Any]:
    """Classify invalid model JSON/schema only after its source evidence is verified."""
    calls = []
    try:
        for block in blocks:
            call = json.loads(block)
            if (not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]
                    or not isinstance(call.get("arguments"), dict)):
                return [], {"failure_origin": "model", "failure_reason": "invalid_tool_schema", "trainable": True}
            calls.append({"name": call["name"], "arguments": call["arguments"]})
    except json.JSONDecodeError as error:
        return [], {"failure_origin": "model", "failure_reason": "invalid_json", "trainable": True,
                    "parser_error": {"message": error.msg, "position": error.pos,
                                     "line": error.lineno, "column": error.colno}}
    return calls, None


def _parsed_calls(parsed: Any) -> list:
    """Normalize parser output without repairing malformed function arguments."""
    try:
        return [{"name": call["function"]["name"],
                 "arguments": json.loads(call["function"]["arguments"])} for call in (parsed or [])]
    except (KeyError, TypeError, ValueError):
        return []


def install_tool_evidence() -> None:
    """Instrument the pinned vLLM non-streaming path without changing its output tokens."""
    # Optional server dependency: this module is also used by CPU-only training code.
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # pylint: disable=C0415
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser  # pylint: disable=C0415

    original = OpenAIServingChat.chat_completion_full_generator
    if getattr(original, "hyper_tool_evidence", False):
        return
    parse = Hermes2ProToolParser.extract_tool_calls

    @wraps(parse)
    def capture_parser(self: Any, model_output: str, request: Any) -> Any:
        """Record Hermes input/result without replacing sampled tokens or parser output."""
        result = parse(self, model_output, request)
        capture = _CAPTURE.get()
        if capture is not None:
            capture.append({"parser_input": model_output, "parser_result": result.model_dump(),
                            "source": "Hermes2ProToolParser.extract_tool_calls"})
        return result

    @wraps(original)
    async def capture_response(self: Any, request: Any, result_generator: Any, request_id: str, model_name: str,
                               conversation: Any, tokenizer: Any, request_metadata: Any,
                               reasoning_parser: Any = None) -> Any:
        """Attach one-call evidence only when both engine output and parser capture are unambiguous."""
        capture = []
        outputs = []

        async def observe() -> Any:
            """Keep the final engine output while preserving the original async iterator."""
            async for result in result_generator:
                outputs[:] = result.outputs
                yield result

        token = _CAPTURE.set(capture if request.return_token_ids else None)
        try:
            response = await original(self, request, observe(), request_id, model_name,
                                      conversation, tokenizer, request_metadata, reasoning_parser)
            if capture and len(outputs) == len(capture) == 1:
                output = outputs[0]
                capture[0].update(
                    engine_text=output.text, token_ids=list(output.token_ids), request_id=request_id,
                    decoded_tokens=tokenizer.decode(output.token_ids, skip_special_tokens=False),
                )
                response.__pydantic_extra__ = {**(response.model_extra or {}), "hyper_tool_protocol": capture}
            return response
        finally:
            _CAPTURE.reset(token)

    capture_response.hyper_tool_evidence = True
    Hermes2ProToolParser.extract_tool_calls = capture_parser
    OpenAIServingChat.chat_completion_full_generator = capture_response
