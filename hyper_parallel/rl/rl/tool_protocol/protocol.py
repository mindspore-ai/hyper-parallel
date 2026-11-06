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
"""Read-only tool parser evidence and fail-closed rollout attribution."""

from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import re
from typing import Any, Iterable, Mapping

from rl.tool_protocol.xml import inspect_xml_calls


class InfrastructureRolloutError(RuntimeError):
    """A rollout failure that must not enter training as a task reward."""


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


def _json_error_type(error_message: str) -> str:
    """Classify a JSON parser failure for stable protocol diagnostics."""
    message = error_message.lower()
    if "invalid \\escape" in message:
        return "invalid_escape"
    if "unterminated string" in message:
        return "unterminated_string"
    if "invalid control character" in message:
        return "invalid_control_character"
    if "expecting ',' delimiter" in message or "expecting ':' delimiter" in message:
        return "missing_delimiter"
    if "extra data" in message:
        return "extra_data"
    return "other_json_error"


def inspect_tool_response(response: dict) -> dict:
    """Include stable source and parser diagnostics for every protocol failure."""
    outcome = _inspect_tool_response(response)
    reason = outcome.get("failure_reason")
    if reason is None:
        return outcome
    origin = outcome.get("failure_origin")
    if origin == "model":
        category = "model_invalid_json" if reason == "invalid_json" else "model_invalid_tool_call"
        stage = "generation"
    elif reason in {"parser_result_mismatch", "missing_parser_result"}:
        evidence = response.get("hyper_tool_protocol")
        xml_parser = (isinstance(evidence, list) and len(evidence) == 1
                      and isinstance(evidence[0], dict)
                      and evidence[0].get("source") == "Qwen3CoderToolParser.extract_tool_calls")
        category, stage = "parser_error", "qwen3_coder_parse" if xml_parser else "hermes_parse"
    elif reason in {"parser_input_mismatch", "incomplete_parser_evidence"}:
        category, stage = "unknown_json_error", "serialization"
    elif reason == "missing_parser_evidence":
        category, stage = "unknown_json_error", "hermes_parse"
    else:
        category, stage = "serialization_error", "serialization"
    outcome = {**outcome, "error_category": category, "failure_stage": stage}
    if reason == "invalid_json":
        parser_error = outcome.get("parser_error") or {}
        outcome["json_error_type"] = _json_error_type(str(parser_error.get("message", "")))
    evidence = response.get("hyper_tool_protocol")
    if isinstance(evidence, list) and len(evidence) == 1 and isinstance(evidence[0], dict):
        blocks = _BLOCK.findall(evidence[0].get("parser_input", ""))
        if blocks:
            outcome["raw_tool_call_json"] = blocks[0]
            outcome["raw_tool_call_sha256"] = hashlib.sha256(blocks[0].encode("utf-8")).hexdigest()
            outcome["tool_call_block_index"] = 0
    return outcome


def _inspect_tool_response(response: dict) -> dict:
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
    calls, error, xml_format = _evidence_calls(item, blocks)
    if error is not None:
        return error
    parsed_result = item.get("parser_result")
    if not isinstance(parsed_result, dict):
        return {"failure_origin": "unknown", "failure_reason": "missing_parser_result", "trainable": False}
    parsed_message = _parsed_calls(message.get("tool_calls"))
    captured_calls = _parsed_calls(parsed_result.get("tool_calls"))
    if not _calls_equal(parsed_message, calls, xml_format) or not _calls_equal(captured_calls, calls, xml_format):
        result.update(failure_origin="infrastructure", failure_reason="parser_result_mismatch", trainable=False)
    return result


def _evidence_calls(item: dict, blocks: list[str]) -> tuple[list, Any, bool]:
    """Select the captured parser format without repairing generated arguments."""
    source = item.get("source")
    xml_format = source == "Qwen3CoderToolParser.extract_tool_calls"
    if xml_format:
        if item.get("parser_format") != "qwen3_xml":
            return [], {"failure_origin": "unknown", "failure_reason": "unsupported_parser_format",
                        "trainable": False}, xml_format
        calls, error = inspect_xml_calls(item["parser_input"], item.get("request_tools"))
    elif source not in (None, "Hermes2ProToolParser.extract_tool_calls"):
        return [], {"failure_origin": "unknown", "failure_reason": "unsupported_parser_source",
                    "trainable": False}, xml_format
    elif any(block.lstrip().startswith("<function=") for block in blocks):
        return [], {"failure_origin": "infrastructure", "failure_reason": "parser_format_mismatch",
                    "trainable": False}, xml_format
    else:
        calls, error = _raw_calls(blocks)
    return calls, error, xml_format


def _validate_source_evidence(item: dict, choice: dict) -> Any:
    """Require agreement between immutable action IDs and each recorded tool text."""
    if not all(isinstance(item.get(key), str) for key in ("parser_input", "decoded_tokens", "engine_text")):
        return {"failure_origin": "unknown", "failure_reason": "incomplete_parser_evidence", "trainable": False}
    blocks = _BLOCK.findall(item["parser_input"])
    engine_blocks = _BLOCK.findall(item["engine_text"])
    decoded = item["decoded_tokens"]
    if any(decoded == item["engine_text"] + ending for ending in ("<|im_end|>", "<|endoftext|>")):
        decoded = item["engine_text"]
    if (not isinstance(item.get("token_ids"), list) or not item["token_ids"]
            or item["token_ids"] != choice.get("token_ids")
            or engine_blocks != _BLOCK.findall(decoded)):
        return {"failure_origin": "unknown", "failure_reason": "parser_input_mismatch", "trainable": False}
    if "reasoning_parser" in item:
        return _validate_reasoning_evidence(item, choice)
    if blocks != engine_blocks:
        return {"failure_origin": "unknown", "failure_reason": "parser_input_mismatch", "trainable": False}
    return None


def _validate_reasoning_evidence(item: dict, choice: dict) -> Any:
    """Match reasoning-only output and source text to immutable parser evidence."""
    evidence = item["reasoning_parser"]
    message = choice.get("message", {})
    # The unified vLLM parser feeds empty text to tools for reasoning-only output.
    content = evidence.get("content") if isinstance(evidence, dict) else None
    if (not isinstance(evidence, dict)
            or evidence.get("source") != "Qwen3ReasoningParser.extract_reasoning"
            or not isinstance(evidence.get("reasoning"), str)
            or evidence.get("parser_input") != item["engine_text"]
            or "content" not in evidence
            or ("" if content is None else content) != item["parser_input"]
            or any(message.get(key) is not None and message[key] != evidence["reasoning"]
                   for key in ("reasoning", "reasoning_content"))):
        return {"failure_origin": "unknown", "failure_reason": "reasoning_parser_mismatch", "trainable": False}
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


def _calls_equal(parsed: list, expected: list, typed: bool) -> bool:
    """Retain XML argument type identity rather than accepting Python bool/int equality."""
    if not typed:
        return parsed == expected
    try:
        parsed_json = json.dumps(parsed, sort_keys=True, allow_nan=False)
        return parsed_json == json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _tool_parser_capture(parser_cls: type, source: str, *, xml_format: bool = False) -> Any:
    """Build one instrumented delegate with its original parser and schema provenance."""
    original = parser_cls.extract_tool_calls

    @wraps(original)
    def capture_parser(self: Any, model_output: str, request: Any) -> Any:
        """Record input/result without replacing sampled tokens or parser output."""
        original_tools = None
        context = _CAPTURE.get()
        if xml_format and context is not None and request is context["request"]:
            tools = getattr(request, "tools", None)
            if isinstance(tools, list):
                try:
                    original_tools = json.loads(json.dumps([
                        tool.model_dump(mode="json") if hasattr(tool, "model_dump") else tool for tool in tools
                    ], allow_nan=False))
                except (TypeError, ValueError):
                    pass
        result = original(self, model_output, request)
        capture = _CAPTURE.get()
        if capture is not None:
            evidence = {"invalid": "cross_request_tool_capture"}
            if request is capture["request"]:
                evidence = {"parser_input": model_output, "parser_result": result.model_dump(), "source": source}
                if xml_format:
                    evidence["parser_format"] = "qwen3_xml"
                    evidence["request_tools"] = original_tools
            capture["tools"].append(evidence)
        return result

    return capture_parser


def install_tool_evidence() -> None:
    """Instrument the pinned vLLM non-streaming path without changing its output tokens."""
    # Optional server dependency: this module is also used by CPU-only training code.
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat  # pylint: disable=C0415
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser  # pylint: disable=C0415
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser  # pylint: disable=C0415

    original = OpenAIServingChat.chat_completion_full_generator
    if getattr(original, "hyper_tool_evidence", False):
        return
    generator_signature = inspect.signature(original)
    if "parser" in generator_signature.parameters:
        parser_parameter = "parser"
    elif "reasoning_parser" in generator_signature.parameters:
        parser_parameter = "reasoning_parser"
    else:
        raise ValueError("vLLM full-generator interface has no supported parser parameter")
    parsers = [(Hermes2ProToolParser, False, "Hermes2ProToolParser.extract_tool_calls")]
    if parser_parameter == "parser":
        # Qwen3Coder is a reviewed 0.23 delegate; legacy 0.22 Hermes behavior stays unchanged.
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser  # pylint: disable=C0415
        parsers.append((Qwen3CoderToolParser, True, "Qwen3CoderToolParser.extract_tool_calls"))
    extract_reasoning = Qwen3ReasoningParser.extract_reasoning

    @wraps(extract_reasoning)
    def capture_reasoning(self: Any, model_output: str, request: Any) -> Any:
        """Record the actual Qwen3 split, never derive a new split from output strings."""
        result = extract_reasoning(self, model_output, request)
        capture = _CAPTURE.get()
        if capture is not None:
            evidence = {"invalid": "cross_request_reasoning_capture"}
            if request is capture["request"]:
                evidence = {"source": "Qwen3ReasoningParser.extract_reasoning", "parser_input": model_output,
                            "reasoning": result[0], "content": result[1]}
            capture["reasoning"].append(evidence)
        return result

    @wraps(original)
    async def capture_response(self: Any, request: Any, result_generator: Any, request_id: str, model_name: str,
                               conversation: Any, tokenizer: Any, request_metadata: Any,
                               *parser_args: Any, **parser_kwargs: Any) -> Any:
        """Attach one-call evidence only when both engine output and parser capture are unambiguous."""
        bound = generator_signature.bind(self, request, result_generator, request_id, model_name,
                                         conversation, tokenizer, request_metadata, *parser_args, **parser_kwargs)
        selected_parser = bound.arguments.get(parser_parameter)
        reasoning_parser = (getattr(selected_parser, "reasoning_parser", None)
                            if parser_parameter == "parser" else selected_parser)
        capture = []
        reasoning_capture = []
        outputs = []

        async def observe() -> Any:
            """Keep the final engine output while preserving the original async iterator."""
            async for result in result_generator:
                outputs[:] = result.outputs
                yield result

        context = {"request": request, "tools": capture, "reasoning": reasoning_capture}
        token = _CAPTURE.set(context if request.return_token_ids else None)
        try:
            response = await original(self, request, observe(), request_id, model_name,
                                      conversation, tokenizer, request_metadata, *parser_args, **parser_kwargs)
            if capture and len(outputs) == len(capture) == 1:
                output = outputs[0]
                capture[0].update(
                    engine_text=output.text, token_ids=list(output.token_ids), request_id=request_id,
                    decoded_tokens=tokenizer.decode(output.token_ids, skip_special_tokens=False),
                )
                if reasoning_parser is not None or reasoning_capture:
                    evidence = (reasoning_capture[0] if len(reasoning_capture) == 1
                                else {"invalid": "missing_or_ambiguous_reasoning_capture"})
                    if evidence.get("reasoning") is not None or "invalid" in evidence:
                        capture[0]["reasoning_parser"] = evidence
                response.__pydantic_extra__ = {**(response.model_extra or {}), "hyper_tool_protocol": capture}
            return response
        finally:
            _CAPTURE.reset(token)

    capture_response.hyper_tool_evidence = True
    for parser_cls, xml_format, source in parsers:
        parser_cls.extract_tool_calls = _tool_parser_capture(parser_cls, source, xml_format=xml_format)
    Qwen3ReasoningParser.extract_reasoning = capture_reasoning
    OpenAIServingChat.chat_completion_full_generator = capture_response
