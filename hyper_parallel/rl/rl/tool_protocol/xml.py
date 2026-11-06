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
"""Independent, closed-call Qwen3 XML interpretation against immutable tool schemas."""

from __future__ import annotations

import json
import math
import re
from typing import Any


_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"\s*<function=([^<>\s=]+)>(.*?)</function>\s*", re.DOTALL)
_PARAMETER = re.compile(r"\s*<parameter=([^<>\s=]+)>(.*?)</parameter>", re.DOTALL)
_ANNOTATIONS = {"description", "title", "default", "examples", "$comment", "$schema"}
_SUPPORTED = _ANNOTATIONS | {"type", "anyOf", "properties", "required", "additionalProperties", "items",
                             "enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                             "minLength", "maxLength", "pattern", "minItems", "maxItems"}
_TYPES = {"string", "integer", "number", "boolean", "object", "array", "null"}


class _UnsupportedSchema(ValueError):
    """Distinguish an unreviewed request schema from invalid generated arguments."""


def _declared_types(schema: dict) -> list[str]:
    """Read explicit, inferred or nullable types before validating ambiguity."""
    if "anyOf" in schema:
        choices = schema["anyOf"]
        if (not isinstance(choices, list) or len(choices) != 2
                or any(not isinstance(choice, dict) for choice in choices)):
            raise _UnsupportedSchema("Only a single nullable union is reviewed")
        candidates = [_types(choice) for choice in choices]
        if any(len(candidate) != 1 for candidate in candidates):
            raise _UnsupportedSchema("Ambiguous nested union")
        types = [candidate[0] for candidate in candidates]
    else:
        value = schema.get("type")
        if value is None and "enum" in schema:
            value = _enum_types(schema["enum"])
        if value is None:
            value = "object" if "properties" in schema else "string"
        types = value if isinstance(value, list) else [value]
    return types


def _types(schema: dict) -> tuple[str, ...]:
    """Resolve one standard type, including the common nullable anyOf form."""
    if not isinstance(schema, dict) or set(schema) - _SUPPORTED:
        raise _UnsupportedSchema("Unsupported tool schema keywords")
    types = _declared_types(schema)
    if (not types or any(not isinstance(value, str) or value not in _TYPES for value in types)
            or len(set(types)) != len(types) or len(types) > 2
            or len(types) == 2 and "null" not in types):
        raise _UnsupportedSchema("Unsupported or ambiguous tool schema type")
    return tuple(types)


def _enum_types(values: Any) -> list[str]:
    """Infer the same simple literal types as pinned Qwen3Coder schema conversion."""
    if not isinstance(values, list) or not values:
        raise _UnsupportedSchema("Invalid enum schema")
    kinds = set()
    for value in values:
        if value is None:
            kinds.add("null")
        elif isinstance(value, bool):
            kinds.add("boolean")
        elif isinstance(value, int):
            kinds.add("integer")
        elif isinstance(value, float) and math.isfinite(value):
            kinds.add("number")
        elif isinstance(value, str):
            kinds.add("string")
        elif isinstance(value, dict):
            kinds.add("object")
        elif isinstance(value, list):
            kinds.add("array")
        else:
            raise _UnsupportedSchema("Invalid enum literal")
    if {"integer", "number"}.issubset(kinds):
        kinds.remove("integer")
    return sorted(kinds)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    """Keep duplicate JSON object keys from silently replacing model arguments."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON argument key")
        result[key] = value
    return result


def _json_value(raw: str) -> Any:
    """Decode a complete JSON value without duplicate-key or nonfinite repair."""
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _reject_constant(value: str) -> Any:
    """Reject JSON parser extensions that are not finite argument values."""
    raise ValueError(f"Nonfinite {value}")


def _parameter_value(raw: str, schema: dict) -> Any:
    """Interpret pinned XML boundary newlines and schema types without changing raw evidence."""
    # Qwen3Coder removes exactly one formatting newline at each parameter boundary.
    value = raw[1:] if raw.startswith("\n") else raw
    value = value[:-1] if value.endswith("\n") else value
    types = _types(schema)
    if "null" in types and value.lower() == "null":
        return None
    if "integer" in types:
        try:
            return int(value)
        except ValueError:
            pass
    if "number" in types:
        try:
            number = float(value)
            if math.isfinite(number):
                return int(number) if number.is_integer() else number
        except ValueError:
            pass
    if "boolean" in types and value.strip().lower() in ("true", "false", "1", "0"):
        return value.strip().lower() in ("true", "1")
    if "string" in types:
        return value
    return _json_value(value)


def _type_matches(value: Any, kind: str) -> bool:
    """Use JSON Schema types, including integral finite floats for integer values."""
    integer = isinstance(value, int) and not isinstance(value, bool)
    finite_float = isinstance(value, float) and math.isfinite(value)
    number = integer or finite_float
    matches = {"string": isinstance(value, str), "integer": integer or finite_float and value.is_integer(),
               "number": number, "boolean": isinstance(value, bool), "object": isinstance(value, dict),
               "array": isinstance(value, list), "null": value is None}
    return matches[kind]


def _numeric_constraints(value: int | float, schema: dict) -> bool:
    """Validate finite numeric bounds without coercing schema literals."""
    checks = {"minimum": lambda bound: value >= bound, "maximum": lambda bound: value <= bound,
              "exclusiveMinimum": lambda bound: value > bound, "exclusiveMaximum": lambda bound: value < bound}
    for key, check in checks.items():
        if key in schema:
            bound = schema[key]
            if (not isinstance(bound, (int, float)) or isinstance(bound, bool)
                    or isinstance(bound, float) and not math.isfinite(bound)):
                raise _UnsupportedSchema("Invalid numeric bound")
            if not check(bound):
                return False
    return True


def _scalar_constraints(value: Any, schema: dict) -> bool:
    """Validate the reviewed literal, numeric and string constraints."""
    if "enum" in schema:
        if not isinstance(schema["enum"], list):
            raise _UnsupportedSchema("Invalid enum schema")
        if not any(_same_json_value(value, allowed) for allowed in schema["enum"]):
            return False
    if "const" in schema and not _same_json_value(value, schema["const"]):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not _numeric_constraints(value, schema):
            return False
    if isinstance(value, str):
        if not _length_constraints(len(value), schema, "minLength", "maxLength"):
            return False
        if "pattern" in schema:
            if not isinstance(schema["pattern"], str):
                raise _UnsupportedSchema("Invalid string pattern")
            try:
                if re.search(schema["pattern"], value) is None:
                    return False
            except re.error as error:
                raise _UnsupportedSchema("Invalid string pattern") from error
    return True


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare JSON literals without treating booleans as numeric enum values."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json_value(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json_value(a, b) for a, b in zip(left, right))
    return left == right


def _length_constraints(length: int, schema: dict, minimum: str, maximum: str) -> bool:
    """Reject invalid schema lengths and check applicable collection bounds."""
    for key in (minimum, maximum):
        if key in schema and (not isinstance(schema[key], int) or isinstance(schema[key], bool) or schema[key] < 0):
            raise _UnsupportedSchema("Invalid length bound")
    return schema.get(minimum, 0) <= length <= schema.get(maximum, length)


def _valid_object(value: dict, schema: dict) -> bool:
    """Validate required, declared and additional object properties recursively."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    additional = schema.get("additionalProperties", True)
    if (not isinstance(properties, dict) or not isinstance(required, list)
            or any(not isinstance(name, str) for name in required)
            or not isinstance(additional, (bool, dict))):
        raise _UnsupportedSchema("Invalid object schema")
    if not set(required).issubset(value):
        return False
    for name, argument in value.items():
        if name in properties:
            if not _valid_value(argument, properties[name]):
                return False
        elif additional is False:
            return False
        elif isinstance(additional, dict) and not _valid_value(argument, additional):
            return False
    return True


def _valid_value(value: Any, schema: dict) -> bool:
    """Validate supplied JSON values against the supported schema without coercing them."""
    if schema == {}:
        try:
            json.dumps(value, allow_nan=False)
            return True
        except (TypeError, ValueError):
            return False
    kinds = _types(schema)
    if not any(_type_matches(value, kind) for kind in kinds) or not _scalar_constraints(value, schema):
        return False
    if "anyOf" in schema and "type" in schema:
        if not any(_type_matches(value, kind) for kind in _types({"type": schema["type"]})):
            return False
    if "anyOf" in schema and not any(_valid_value(value, choice) for choice in schema["anyOf"]):
        return False
    if isinstance(value, dict):
        return _valid_object(value, schema)
    if isinstance(value, list):
        if not _length_constraints(len(value), schema, "minItems", "maxItems"):
            return False
        if "items" in schema and any(not _valid_value(item, schema["items"]) for item in value):
            return False
    return True


def _function_call(block: str, tools: list[dict]) -> dict:
    """Read one complete function and its declared parameters without parser backoff."""
    function = _FUNCTION.fullmatch(block)
    if function is None:
        raise ValueError("Malformed XML function")
    name, body = function.groups()
    declarations = [tool.get("function") for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"
                    and isinstance(tool.get("function"), dict) and tool["function"].get("name") == name]
    if len(declarations) > 1:
        raise _UnsupportedSchema("Duplicate declared tool name")
    if not declarations:
        raise ValueError("Undeclared XML function")
    schema = declarations[0].get("parameters", {"type": "object"})
    if "object" not in _types(schema):
        raise _UnsupportedSchema("Tool parameters must describe an object")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise _UnsupportedSchema("Invalid properties schema")
    arguments = {}
    cursor = 0
    while cursor < len(body) and body[cursor:].strip():
        parameter = _PARAMETER.match(body, cursor)
        if parameter is None:
            raise ValueError("Malformed XML parameter")
        parameter_name, raw = parameter.groups()
        if parameter_name in arguments:
            raise ValueError("Duplicate XML parameter")
        arguments[parameter_name] = _parameter_value(raw, properties.get(parameter_name, {}))
        cursor = parameter.end()
    if not _valid_value(arguments, schema):
        raise ValueError("Invalid XML arguments")
    return {"name": name, "arguments": arguments}


def inspect_xml_calls(parser_input: str, request_tools: Any) -> tuple[list[dict], dict | None]:
    """Interpret complete Qwen3 XML calls using the captured original request tool schemas.

    Unknown or unsupported schemas remain untrainable. Invalid generated syntax and
    arguments are trainable model errors; no incomplete tag or argument is repaired.
    """
    if not isinstance(request_tools, list) or not request_tools:
        return [], {"failure_origin": "unknown", "failure_reason": "missing_tool_schema", "trainable": False}
    if any(not isinstance(tool, dict) or tool.get("type") != "function"
           or not isinstance(tool.get("function"), dict)
           or not isinstance(tool["function"].get("name"), str) for tool in request_tools):
        return [], {"failure_origin": "unknown", "failure_reason": "unsupported_tool_schema", "trainable": False}
    blocks = _CALL.findall(parser_input)
    if len(blocks) != parser_input.count("<tool_call>") or len(blocks) != parser_input.count("</tool_call>"):
        return [], {"failure_origin": "model", "failure_reason": "invalid_tool_schema", "trainable": True}
    try:
        return [_function_call(block, request_tools) for block in blocks], None
    except _UnsupportedSchema:
        return [], {"failure_origin": "unknown", "failure_reason": "unsupported_tool_schema", "trainable": False}
    except (ValueError, TypeError, OverflowError):
        return [], {"failure_origin": "model", "failure_reason": "invalid_tool_schema", "trainable": True}
