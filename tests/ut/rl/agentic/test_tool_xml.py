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
"""Independent Qwen3 XML attribution with immutable source and typed request schemas."""

from copy import deepcopy
import hashlib
import json
from typing import Any
import unittest

from rl.tool_protocol.protocol import inspect_tool_response
from rl.tool_protocol.xml import inspect_xml_calls


TOOLS = [{"type": "function", "function": {"name": "add_numbers", "parameters": {
    "type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}, "label": {"type": "string"},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "flag": {"type": "boolean"},
        "number": {"type": "number"}, "data": {"type": "object", "properties": {"value": {"type": "integer"}},
                                              "required": ["value"], "additionalProperties": False},
        "values": {"type": "array", "items": {"type": "integer"}},
    }, "required": ["a", "b", "label"], "additionalProperties": False,
}}}]
_USE_DEFAULT_TOOLS = object()


def xml_call(parameters: str) -> str:
    """Emit exact model text without fixing any parameter markup."""
    return f"<tool_call>\n<function=add_numbers>{parameters}</function>\n</tool_call>"


def response(raw: str, arguments: dict, tools: Any = _USE_DEFAULT_TOOLS) -> dict:
    """Build unchanged engine, parser and public API evidence for a selected call."""
    calls = [{"type": "function", "function": {"name": "add_numbers", "arguments": json.dumps(arguments)}}]
    return {"choices": [{"token_ids": [17, 18], "message": {"tool_calls": deepcopy(calls), "content": None}}],
            "hyper_tool_protocol": [{"source": "Qwen3CoderToolParser.extract_tool_calls", "parser_format": "qwen3_xml",
                                     "request_tools": deepcopy(TOOLS if tools is _USE_DEFAULT_TOOLS else tools),
                                     "parser_input": raw, "engine_text": raw,
                                     "decoded_tokens": raw, "token_ids": [17, 18],
                                     "parser_result": {"tool_calls": deepcopy(calls)}}]}


class TestToolXML(unittest.TestCase):
    """Never relabel valid XML as invalid JSON or derive arguments from parser output."""

    def _assert_failure(self, outcome: dict, expected: dict, raw: str, category: str, stage: str) -> None:
        """Retain exact attribution assertions and also check the mainline diagnostic evidence."""
        body = raw.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        # The mainline end-of-text boundary excludes a final newline on an unclosed block.
        if "</tool_call>" not in raw and body.endswith("\n"):
            body = body[:-1]
        expected = {**expected, "error_category": category, "failure_stage": stage,
                    "raw_tool_call_json": body,
                    "raw_tool_call_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "tool_call_block_index": 0}
        self.assertEqual(outcome, expected)

    def setUp(self) -> None:
        """Freeze the typed arithmetic arguments used by the native interface smoke."""
        self.parameters = "<parameter=a>1</parameter><parameter=b>1</parameter><parameter=label>\nresult\n</parameter>"
        self.arguments = {"a": 1, "b": 1, "label": "result"}

    def test_complete_typed_xml_preserves_original_evidence(self) -> None:
        """A complete typed call agrees with both parser and public response arguments."""
        value = response(xml_call(self.parameters), self.arguments)
        original = deepcopy(value)
        self.assertEqual(inspect_tool_response(value),
                         {"failure_origin": None, "failure_reason": None, "trainable": True})
        self.assertEqual(value, original)

    def test_pinned_single_boundary_newline_and_nullable_types(self) -> None:
        """Boundary formatting and nullable unions use pinned semantics without stripping payload spaces."""
        raw = self.parameters.replace("\nresult\n", "\n\n result \n\n")
        for note, expected in (("null", None), ("NULL", None), ("text", "text")):
            with self.subTest(note=note):
                parameters = raw + f"<parameter=note>{note}</parameter><parameter=flag>false</parameter>"
                arguments = {**self.arguments, "label": "\n result \n", "note": expected, "flag": False}
                self.assertIsNone(inspect_tool_response(response(xml_call(parameters), arguments))["failure_origin"])

    def test_typed_containers_and_numeric_values_use_original_schema(self) -> None:
        """Recursive declared values and finite numeric conversions match the actual call."""
        parameters = self.parameters + ("<parameter=data>{\"value\":2}</parameter>"
                                        "<parameter=values>[1,2]</parameter><parameter=number>1.5</parameter>")
        arguments = {**self.arguments, "data": {"value": 2}, "values": [1, 2], "number": 1.5}
        self.assertIsNone(inspect_tool_response(response(xml_call(parameters), arguments))["failure_origin"])

    def test_bad_model_markup_or_typed_values_are_not_repaired(self) -> None:
        """Partial tags, duplicates, omitted required arguments and bad types remain model failures."""
        invalid = [xml_call(self.parameters)[:-12], xml_call(self.parameters + "<parameter=a>2</parameter>"),
                   xml_call(self.parameters.replace("<parameter=b>1</parameter>", "")),
                   xml_call(self.parameters.replace("<parameter=a>1</parameter>", "<parameter=a>bad</parameter>")),
                   xml_call(self.parameters + "<parameter=flag>yes</parameter>"),
                   xml_call(self.parameters + "<parameter=data>{\"value\":1,\"value\":2}</parameter>"),
                   xml_call(self.parameters + "<parameter=number>1e999</parameter>")]
        for raw in invalid:
            with self.subTest(raw=raw):
                outcome = inspect_tool_response(response(raw, self.arguments))
                self._assert_failure(outcome,
                                     {"failure_origin": "model", "failure_reason": "invalid_tool_schema",
                                      "trainable": True}, raw, "model_invalid_tool_call", "generation")

    def test_missing_or_ambiguous_schema_remains_unknown(self) -> None:
        """Unsupported request typing cannot be blamed on the generated text."""
        tools = deepcopy(TOOLS)
        tools[0]["function"]["parameters"]["properties"]["a"] = {
            "anyOf": [{"type": "integer"}, {"type": "string"}]}
        for declaration, reason in ((None, "missing_tool_schema"), (tools, "unsupported_tool_schema")):
            with self.subTest(reason=reason):
                outcome = inspect_tool_response(response(xml_call(self.parameters), self.arguments, declaration))
                self._assert_failure(outcome,
                                     {"failure_origin": "unknown", "failure_reason": reason, "trainable": False},
                                     xml_call(self.parameters), "serialization_error", "serialization")

    def test_wrong_parser_and_wrong_typed_result_fail_closed(self) -> None:
        """XML under Hermes is a format error; bool/int Python equality cannot hide parser corruption."""
        value = response(xml_call(self.parameters), self.arguments)
        value["hyper_tool_protocol"][0]["source"] = "Hermes2ProToolParser.extract_tool_calls"
        self.assertEqual(inspect_tool_response(value)["failure_reason"], "parser_format_mismatch")
        value = response(xml_call(self.parameters), self.arguments)
        value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {**self.arguments, "a": True})
        self._assert_failure(inspect_tool_response(value),
                             {"failure_origin": "infrastructure", "failure_reason": "parser_result_mismatch",
                              "trainable": False}, xml_call(self.parameters), "parser_error", "qwen3_coder_parse")

    def test_unknown_source_and_changed_token_identity_remain_untrainable(self) -> None:
        """Source dispatch cannot bypass immutable generated IDs."""
        value = response(xml_call(self.parameters), self.arguments)
        value["hyper_tool_protocol"][0]["source"] = "Other.extract_tool_calls"
        self.assertEqual(inspect_tool_response(value)["failure_reason"], "unsupported_parser_source")
        value["hyper_tool_protocol"][0]["source"] = "Qwen3CoderToolParser.extract_tool_calls"
        value["choices"][0]["token_ids"] = [17, 19]
        self.assertEqual(inspect_tool_response(value)["failure_reason"], "parser_input_mismatch")

    def test_incomplete_model_xml_with_exact_terminal_eos_is_a_model_error(self) -> None:
        """A tokenizer-retained EOS cannot turn a verified unclosed model call into unknown evidence."""
        raw = "<tool_call><function=add_numbers><parameter=a>1</parameter><parameter=b>1"
        for ending in ("<|im_end|>", "<|endoftext|>"):
            with self.subTest(ending=ending):
                value = response(raw, self.arguments)
                value["hyper_tool_protocol"][0]["decoded_tokens"] = raw + ending
                outcome = inspect_tool_response(value)
                self.assertEqual(outcome["failure_origin"], "model")
                self.assertEqual(outcome["failure_reason"], "invalid_tool_schema")
                self.assertTrue(outcome["trainable"])
                value["hyper_tool_protocol"][0]["decoded_tokens"] = raw + "changed" + ending
                self.assertEqual(inspect_tool_response(value)["failure_reason"], "parser_input_mismatch")

    def test_direct_parser_ignores_plain_text_without_fabricating_calls(self) -> None:
        """A final assistant answer contains no artificial tool invocation."""
        self.assertEqual(inspect_xml_calls("2", TOOLS), ([], None))

    def test_top_level_type_and_literal_types_constrain_nullable_unions(self) -> None:
        """A nullable branch cannot override an intersecting type or bool/int enum distinction."""
        tools = deepcopy(TOOLS)
        properties = tools[0]["function"]["parameters"]["properties"]
        properties["note"]["type"] = "string"
        properties["flag"]["enum"] = [1]
        for suffix, extra in (("<parameter=note>null</parameter>", {"note": None}),
                              ("<parameter=flag>true</parameter>", {"flag": True})):
            with self.subTest(suffix=suffix):
                value = response(xml_call(self.parameters + suffix), {**self.arguments, **extra}, tools)
                self.assertEqual(inspect_tool_response(value)["failure_origin"], "model")

    def test_simple_enum_type_inference_is_not_misattributed_to_model(self) -> None:
        """An enum without type still declares numeric literals under pinned XML semantics."""
        tools = deepcopy(TOOLS)
        tools[0]["function"]["parameters"]["properties"]["a"] = {"enum": [1, None]}
        value = response(xml_call(self.parameters), self.arguments, tools)
        self.assertIsNone(inspect_tool_response(value)["failure_origin"])
