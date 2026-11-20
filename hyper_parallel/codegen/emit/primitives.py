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
"""Materialize communication boundaries around imported Torch primitives."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from typing import Any

from hyper_parallel.codegen.inline.ir import ConstructorPatch, ImportPatch, InlinePatchSet
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.distributed._builder.forward_rewriter import (
    _is_vocab_parallel_embed,
    _wrap_vocab_parallel_embedding,
)


def observe_primitive_boundaries(capture: Any) -> list[dict]:
    """Read native primitive boundaries and their actual source construction owners."""
    plan = capture.plan
    if plan is None:
        return []
    source_types = {path: item.source_type.legacy_path()
                    for item in capture.manifest.replacements for path in item.aliases}
    owners = {path: source_types.get(path, f"{type(module).__module__}:{type(module).__qualname__}")
              for path, module in capture.model.named_modules()}
    result = []
    for path, spec in sorted(plan.modules.items()):
        if not spec.is_boundary:
            continue
        module = capture.model.get_submodule(path)
        identity = f"{type(module).__module__}:{type(module).__qualname__}"
        in_src, in_dst = getattr(spec, 'in_src', None), getattr(spec, 'in_dst', None)
        out_src, out_dst = getattr(spec, 'out_src', None), getattr(spec, 'out_dst', None)
        unchanged = in_src == in_dst and (out_dst is None or out_src == out_dst)
        if unchanged or not type(module).__module__.startswith('torch.nn.'):
            continue
        dense = capture.mesh.device_mesh
        tp = dense['tp'] if 'tp' in dense.mesh_dim_names and dense['tp'].size() > 1 else None
        arguments = list(inspect.signature(module.forward).parameters.values())
        if len(arguments) != 1 or arguments[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(f"Codegen imported boundary {path!r} requires a single-input primitive forward")
        parent, _, attribute = path.rpartition('.')
        owner = capture.model.get_submodule(parent)
        owner_type = source_types.get(parent, f"{type(owner).__module__}:{type(owner).__qualname__}")
        result.append({
            'fqn': path, 'type': identity, 'owner': owner_type, 'attribute': attribute,
            'vocab': _is_vocab_parallel_embed(module, spec, tp),
            'inputs': [str(getattr(key, 'value', key)) for key in spec.in_src],
            'argument': arguments[0].name,
            'owner_instances': [key for key, value in owners.items() if value == owner_type],
        })
    return result


def _vocab_forward(argument: str) -> str:
    """Copy native masking statements; only group/range acquisition becomes live resource access."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(_wrap_vocab_parallel_embedding)))
    native = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == 'masked_embedding_forward')
    body = list(native.body)
    if ast.get_docstring(native) is not None:
        body.pop(0)
    preface = (
        f"input_ids = {argument}\nargs = ()\nkwargs = {{}}\noriginal_forward = super().forward\n"
        "to_local = getattr(self.weight, 'to_local', None)\n"
        "local_weight = to_local() if callable(to_local) else self.weight\n"
        "v_local = local_weight.shape[0]\n"
        "lo = self._hp_tp_mesh.get_local_rank() * v_local\nhi = lo + v_local\n"
    )
    return preface + ast.unparse(ast.Module(body=body, type_ignores=[]))


def materialize_primitive_boundaries(source: str, meta: Any) -> str:
    """Extend ordinary inline classes and constructor edits, then use the existing lowerer."""
    records = getattr(meta, 'primitive_boundaries', ())
    if not records:
        return source
    patches = InlinePatchSet()
    names = {}
    constructors = {}
    classes = dict(getattr(meta, 'boundary_classes', {}))
    for record in records:
        path, identity = record['fqn'], record['type']
        if len(record['inputs']) != 1 or not record['inputs'][0].isidentifier():
            raise ValueError(f"Codegen imported boundary {path!r} requires an explicit single-input source contract")
        argument = record['argument']
        original_argument = record['inputs'][0]
        entry = dict(meta.param_plan[path])
        for field in ('in_src', 'in_dst'):
            if entry.get(field):
                entry[field] = {argument if key == original_argument else key: value
                                for key, value in entry[field].items()}
        meta.param_plan[path] = entry
        contract = {key: value for key, value in meta.param_plan[path].items() if key != 'params'}
        key = (identity, argument, record['vocab'], json.dumps(contract, sort_keys=True))
        if key not in names:
            module, original = identity.split(':', 1)
            name = f"Parallel{original}_{len(names)}"
            names[key] = name
            base = f"_Native{original}_{len(names)}"
            patches.imports.append(ImportPatch(module, (), raw=f"from {module} import {original} as {base}"))
            body = _vocab_forward(argument) if record['vocab'] else f"return super().forward({argument})"
            if record['vocab']:
                patches.imports.append(ImportPatch('', (), raw='import torch'))
            source += (f"\n\nclass {name}({base}):\n"
                       f"    def forward(self, {argument}):\n{textwrap.indent(body, '        ')}\n")
        name = names[key]
        site = (record['owner'], record['attribute'])
        selected = {item['fqn'].rpartition('.')[0] for item in records
                    if (item['owner'], item['attribute']) == site}
        if selected != set(record['owner_instances']):
            raise ValueError(f"Codegen imported boundary {path!r} partially selects a shared source constructor")
        patch = ConstructorPatch(record['owner'].split(':')[-1], record['attribute'], identity.split(':')[-1], name)
        if site in constructors and constructors[site] != patch:
            raise ValueError(f"Codegen imported boundaries have differing contracts at {site!r}")
        constructors[site] = patch
        classes[path] = name
    patches.constructors.extend(constructors.values())
    meta.boundary_classes = classes
    return apply_patch_set(source, patches)
