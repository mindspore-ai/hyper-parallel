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
"""Inline the method and companion-state results of native preparation."""

from __future__ import annotations

import ast
import textwrap

from hyper_parallel.codegen.astkit.edits import TextEdit, apply_edits
from hyper_parallel.codegen.astkit.index import build_source_index
from hyper_parallel.codegen.inline.ir import ImportPatch, InlinePatchSet
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set


def verify_observed_methods(source: str, records: dict) -> None:
    """Reject a local-compute body whose methods still need runtime installation."""
    classes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    for path, record in records.items():
        name = record["type"].split(":", 1)[-1]
        host = classes.get(name)
        methods = {} if host is None else {
            node.name: node for node in host.body if isinstance(node, ast.FunctionDef)
        }
        for method_name, expected in record["methods"].items():
            actual = methods.get(method_name)
            if method_name == "forward" and "_forward_impl" in methods:
                actual = methods["_forward_impl"]
            wanted = ast.parse(expected).body[0]
            if actual is None or ast.dump(ast.Module(body=actual.body, type_ignores=[])) != ast.dump(
                ast.Module(body=wanted.body, type_ignores=[])
            ):
                raise ValueError(f"Codegen native method {path}.{method_name} was not materialized")


def materialize_observed_methods(source: str, records: dict) -> str:
    """Write observed methods into the classes actually used by the model."""
    grouped = {}
    for path, record in records.items():
        identity = record["type"]
        if identity in grouped and grouped[identity] != record:
            raise ValueError(f"Codegen native methods have different per-instance contracts at {path!r}")
        grouped[identity] = record
    imports = []
    for identity, record in grouped.items():
        module, name = identity.split(":", 1)
        index = build_source_index(source)
        if name not in index.classes:
            alias = f"_Native{name}"
            imports.append(ImportPatch(module, (), raw=f"from {module} import {name} as {alias}"))
            source += f"\n\nclass {name}({alias}):\n    pass\n"
            index = build_source_index(source)
        host = index.classes[name]
        additions = []
        edits = []
        for method_name, body in record["methods"].items():
            old = host.methods.get(method_name)
            text = textwrap.indent(body, "    ") + "\n"
            if old is None:
                additions.append(text)
            else:
                edits.append(TextEdit(old.def_offset, old.body_end, text))
        fields = record["fields"]
        if fields:
            statements = "\n".join(f"        self.{key} = {value}" for key, value in fields.items())
            initializer = host.methods.get("__init__")
            if initializer is None:
                additions.insert(0, "    def __init__(self, *args, **kwargs):\n"
                                  "        super().__init__(*args, **kwargs)\n" + statements + "\n")
            else:
                edits.append(TextEdit(initializer.body_end, initializer.body_end, statements + "\n"))
        if record["local_compute"]:
            lines = source.splitlines(keepends=True)
            offsets = [0]
            for line in lines:
                offsets.append(offsets[-1] + len(line))
            tree = ast.parse(source)
            class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            flags = [node for node in class_node.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "ep_enable" for target in node.targets)]
            if len(flags) != 1:
                raise ValueError(f"Codegen native local-compute class {name!r} has no unique source dispatch")
            flag = flags[0]
            edits.append(TextEdit(offsets[flag.lineno - 1], offsets[flag.end_lineno], "    ep_enable = True\n"))
        if additions:
            edits.append(TextEdit(host.end_offset, host.end_offset, "\n" + "\n".join(additions)))
        source = apply_edits(source, edits)
        imports.extend(ImportPatch("", (), raw=value) for value in record["imports"])
    return apply_patch_set(source, InlinePatchSet(imports=imports))
