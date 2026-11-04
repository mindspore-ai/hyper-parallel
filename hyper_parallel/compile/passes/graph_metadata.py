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
# Adapted from torchtitan-npu's graph_trainer ACLGraph implementation.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# (c) Meta Platforms, Inc. and affiliates.
# The original BSD-3-Clause license is retained in
# Third_Party_Open_Source_Software_Notice.
"""Shared FX storage, host-boundary and fake-value contracts for execution passes."""

from __future__ import annotations

from collections import defaultdict
import operator
from typing import Any

import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_leaves

COMPILED_REGION_META = "hyper_parallel_compiled_region"
COMPILED_MUTATIONS_META = "hyper_parallel_compiled_mutations"


def tensor_storages(node: fx.Node) -> set[StorageWeakRef]:
    """Return storage identities, including tensor leaves of structured values."""
    return {StorageWeakRef(value.untyped_storage()) for value in tree_leaves(node.meta.get("val"))
            if isinstance(value, torch.Tensor)}


def written_nodes(node: fx.Node) -> list[fx.Node]:
    """Resolve writable arguments of native ops and opaque compiled regions."""
    inputs = node.all_input_nodes
    result = [inputs[index] for index in node.meta.get(COMPILED_MUTATIONS_META, ())]
    if isinstance(node.target, torch._ops.OpOverload):
        for index, argument in enumerate(node.target._schema.arguments):
            if argument.alias_info and argument.alias_info.is_write:
                value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
                result.extend(leaf for leaf in tree_leaves(value) if isinstance(leaf, fx.Node))
    return result


def mutated_input_indices(graph_module: fx.GraphModule) -> set[int]:
    """Find input mutations through views, structured values or compiled calls."""
    storage_inputs = defaultdict(set)
    for index, node in enumerate(graph_module.graph.find_nodes(op="placeholder")):
        for storage in tensor_storages(node):
            storage_inputs[storage].add(index)
    mutated = set()
    for node in graph_module.graph.nodes:
        for written in written_nodes(node):
            for storage in tensor_storages(written):
                mutated.update(storage_inputs[storage])
    return mutated


def is_host_node(node: fx.Node) -> bool:
    """Keep CPU work and arbitrary Python helpers outside device recording."""
    opaque = not isinstance(node.target, torch._ops.OpOverload) and node.target is not operator.getitem
    return (
        node.op in ("call_module", "call_method")
        or (opaque and not node.meta.get(COMPILED_REGION_META, False))
        or node.target == torch.ops.aten._local_scalar_dense.default
        or any(isinstance(value, torch.Tensor) and value.device.type == "cpu"
               for parent in (node, *node.all_input_nodes) for value in tree_leaves(parent.meta.get("val")))
    )


def propagate_fake_values(
    graph_module: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    model: torch.nn.Module | None = None,
) -> fx.GraphModule:
    """Refresh metadata before lowering, using physically sharded model state.

    Compiled graphs retain the region output metadata emitted by the compiler;
    running an opaque artifact on FakeTensors would execute its runtime wrapper.
    """
    if graph_module.meta.get("inductor_compiled") in ("regional", "full"):
        return graph_module
    placeholders = list(graph_module.graph.find_nodes(op="placeholder"))
    values = [node.meta.get("val") for node in placeholders]
    if example_inputs is not None:
        values = [value if value is not None else example_inputs[index] for index, value in enumerate(values)]
    mode = next((value.fake_mode for value in values if isinstance(value, FakeTensor)), None)
    if mode is None:
        return graph_module
    if model is not None:
        state = dict(model.named_parameters(remove_duplicate=False))
        state.update(model.named_buffers(remove_duplicate=False))
        for index, fqn in enumerate(getattr(graph_module, "state_fqns", ())):
            values[index] = mode.from_tensor(state[fqn].detach(), static_shapes=True)
    FakeTensorProp(graph_module, mode).propagate_dont_convert_inputs(*values)
    return graph_module
