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
"""Partition a joint FX graph into eager host work and NPU capture regions."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Iterable

import torch
from torch import fx
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_leaves

from ..pass_config import PassConfig
from .base import GraphPass
from .graph_metadata import (
    is_host_node as _host_boundary,
    mutated_input_indices as _mutated_input_indices,
    propagate_fake_values,
    tensor_storages as _storages,
)
from .aclgraph_runtime import (
    ACLGraphWrapper,
    _PartitionInterpreter,
    _RecaptureCoordinator,
    prepare_aclgraph_runtime,
)

logger = logging.getLogger(__name__)


def _is_npu_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "npu"


def _same_layout(value, first):
    if value.dtype != first.dtype or len(value.shape) != len(first.shape):
        return False
    layout = (*value.shape, *value.stride(), value.storage_offset())
    other = (*first.shape, *first.stride(), first.storage_offset())
    return all(statically_known_true(left == right) for left, right in zip(layout, other, strict=True))


def _staging_aliases(placeholders, input_storages, staged_input_indices):
    storage_inputs = defaultdict(list)
    for i, storages in enumerate(input_storages):
        for storage in storages:
            storage_inputs[storage].append(i)
    input_aliases = {}
    for indices in storage_inputs.values():
        if len(indices) < 2:
            continue
        first = placeholders[indices[0]].meta["val"]
        if all(i in staged_input_indices and _same_layout(placeholders[i].meta["val"], first) for i in indices):
            input_aliases.update((i, indices[0]) for i in indices[1:])
        else:
            staged_input_indices.difference_update(indices)
    return input_aliases


def _collect_inputs(gm, static_input_indices):
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    input_storages = [_storages(node) for node in placeholders]
    static_storages = set().union(*(input_storages[i] for i in static_input_indices))
    mutated_storages = set().union(*(input_storages[i] for i in _mutated_input_indices(gm)))
    staged_input_indices = {
        i
        for i, node in enumerate(placeholders)
        if i not in static_input_indices
        and isinstance(value := node.meta.get("val"), torch.Tensor)
        and _is_npu_tensor(value)
        and value.is_contiguous()
        and len(input_storages[i]) == 1
        and not input_storages[i] & (mutated_storages | static_storages)
    }
    input_aliases = _staging_aliases(placeholders, input_storages, staged_input_indices)
    static_storages.update(set().union(*(input_storages[i] for i in staged_input_indices)))
    return placeholders, static_storages, staged_input_indices, input_aliases


def _partitions(gm):
    partitions, host_partitions = {}, set()
    partition_id, previous_host = -1, None
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        host = _host_boundary(node)
        if host != previous_host:
            partition_id += 1
            previous_host = host
        partitions[node] = partition_id
        if host:
            host_partitions.add(partition_id)
    return partitions, host_partitions


def _output_escapes(nodes, escaping_storages, unknown_returns):
    return unknown_returns or any("val" not in node.meta or bool(_storages(node) & escaping_storages) for node in nodes)


def _capture_ownership(result, host_partitions, escaping_storages, unknown_returns, placeholders):
    captured_storages = set()
    cloned_regions = set()
    for name, region in result.named_children():
        if not isinstance(region, torch.fx.GraphModule):
            continue
        if int(name.rsplit("_", 1)[1]) not in host_partitions:
            output = next(node for node in region.graph.nodes if node.op == "output")
            output_nodes = [node for node in tree_leaves(output.args) if isinstance(node, torch.fx.Node)]
            if _output_escapes(output_nodes, escaping_storages, unknown_returns):
                cloned_regions.add(int(name.rsplit("_", 1)[1]))
            region_input_storages = set().union(
                *(_storages(node) for node in region.graph.nodes if node.op == "placeholder")
            )
            for node in tree_leaves(output.args):
                if isinstance(node, torch.fx.Node):
                    # Views of eager inputs inherit their producer's changing
                    # addresses; only allocations made here are capture-owned.
                    captured_storages.update(_storages(node) - region_input_storages)
    for node in placeholders:
        captured_storages.difference_update(_storages(node))
    return captured_storages, cloned_regions


def _region_inputs(region, producers, eager_producers, static_storages, captured_storages):
    region_inputs = [node for node in region.graph.nodes if node.op == "placeholder"]
    # SAR can reuse forward FakeTensor storage metadata on eager
    # recomputations. Trust the actual FX producer, not storage alone.
    captured_inputs = {
        i
        for i, node in enumerate(region_inputs)
        if (producer := producers.get(node.name)) is not None and producer not in eager_producers
    }
    mutated = _mutated_input_indices(region)
    written_storages = set().union(*(_storages(region_inputs[i]) for i in mutated))
    static_indices = (
        mutated
        | captured_inputs
        | {i for i, node in enumerate(region_inputs) if _storages(node) & (static_storages | written_storages)}
    )
    # A mixed container may also own eager/state buffers. Borrow it only
    # when every tensor storage comes from a captured producer.
    borrowed_indices = {
        i
        for i, node in enumerate(region_inputs)
        if i in captured_inputs and (storages := _storages(node)) and storages <= captured_storages
    }
    return static_indices, borrowed_indices


def _wrap_regions(result, partitions, host_partitions, cloned_regions, static_storages, captured_storages, coordinator):
    producers = {node.name: partition for node, partition in partitions.items()}
    summaries = []
    for name, region in result.named_children():
        if not isinstance(region, torch.fx.GraphModule):
            continue
        host = int(name.rsplit("_", 1)[1]) in host_partitions
        summaries.append(
            {
                "name": name,
                "host": host,
                "capture": not host,
            }
        )
        if host:
            continue
        clone_outputs = int(name.rsplit("_", 1)[1]) in cloned_regions
        static_indices, borrowed_indices = _region_inputs(
            region, producers, host_partitions | cloned_regions, static_storages, captured_storages
        )
        wrapper = ACLGraphWrapper(
            region.forward,
            static_input_indices=static_indices,
            capture_runnable=torch.fx.Interpreter(region).boxed_run,
            label=name,
            clone_outputs=clone_outputs,
            recapture_coordinator=coordinator,
            borrowed_input_indices=borrowed_indices,
            capture_boxed=True,
        )
        object.__setattr__(region, "_aclgraph_wrapper", wrapper)
        region.forward = wrapper
        coordinator.wrappers.append(wrapper)
    return summaries


def _partition_graph(gm, *, static_input_indices, coordinate_recapture):
    """Capture contiguous device work, retaining host boundaries in eager."""
    placeholders, static_storages, staged, aliases = _collect_inputs(gm, static_input_indices)
    returned = next(node for node in gm.graph.nodes if node.op == "output")
    returned_nodes = [node for node in tree_leaves(returned.args) if isinstance(node, fx.Node)]
    escaping_storages = set().union(*(_storages(node) for node in returned_nodes))
    unknown_returns = any("val" not in node.meta for node in returned_nodes)
    partitions, host_partitions = _partitions(gm)
    result = split_module(gm, gm, partitions.__getitem__, keep_original_order=True, keep_original_node_name=True)
    captured_storages, cloned_regions = _capture_ownership(
        result, host_partitions, escaping_storages, unknown_returns, placeholders
    )
    coordinator = _RecaptureCoordinator(coordinate_recapture=coordinate_recapture)
    summaries = _wrap_regions(
        result, partitions, host_partitions, cloned_regions, static_storages, captured_storages, coordinator
    )
    result.meta.update(gm.meta)
    result.meta["aclgraph_partitions"] = summaries
    object.__setattr__(result, "_aclgraph_wrappers", coordinator.wrappers)
    object.__setattr__(result, "_aclgraph_recapture_coordinator", coordinator)
    result.forward = _PartitionInterpreter(result, staged_input_indices=staged, input_aliases=aliases).run
    logger.info("Applied ACLGraph pass: %d capture regions, %d eager regions", len(coordinator.wrappers),
                len(host_partitions))
    return result


def _propagate_metadata(graph_module, example_inputs, model):
    if example_inputs is None and model is None:
        return
    propagate_fake_values(graph_module, example_inputs, model)


def _has_npu_work(graph_module):
    return any(_is_npu_tensor(value) for node in graph_module.graph.nodes
               for value in tree_leaves(node.meta.get("val")) if isinstance(value, torch.Tensor))


def _validate_static_indices(indices, count):
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count for index in indices):
        raise ValueError("ACLGraph static_input_indices must index graph placeholders")


class ACLGraphPass(GraphPass):
    """Capture device regions after FSDP, bucketing and communication scheduling.

    Parameters, buffers and mutated inputs retain their original storage.
    Host reads stay eager. Static mode rejects changed replay signatures;
    dynamic mode coordinates recapture across ranks using a CPU process group.
    """

    name = "aclgraph"

    def run(
        self,
        graph_module: fx.GraphModule,
        pass_config: PassConfig,
        example_inputs: tuple[Any, ...] | None = None,
        static_input_indices: Iterable[int] | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> fx.GraphModule:
        """Install NPU capture wrappers, returning the partitioned graph.

        Args:
            graph_module: Traced joint forward/backward graph.
            pass_config: Selects off, static, or dynamic ACLGraph replay.
            example_inputs: Fake inputs used to fill metadata on inserted nodes.
            static_input_indices: Lifted state indices. Defaults to the tracer's
                ``num_state_inputs`` leading placeholders.
            model: Live model, whose parameters may already have been sharded.
            **kwargs: Additional pipeline context, unused here.

        Returns:
            A graph whose device regions warm up once before capture and replay.
            CPU-only graphs and disabled passes are returned unchanged.
        """
        if pass_config.aclgraph_mode == "off":
            return graph_module
        pass_config.validate()
        placeholders = list(graph_module.graph.find_nodes(op="placeholder"))
        _propagate_metadata(graph_module, example_inputs, model)
        if not _has_npu_work(graph_module):
            return graph_module
        if static_input_indices is None:
            static_input_indices = range(getattr(graph_module, "num_state_inputs", 0))
        static_input_indices = set(static_input_indices)
        _validate_static_indices(static_input_indices, len(placeholders))
        prepare_aclgraph_runtime()
        result = _partition_graph(
            graph_module,
            static_input_indices=static_input_indices,
            coordinate_recapture=pass_config.aclgraph_mode == "dynamic",
        )
        # The tracer's runtime contract is stored as attributes, outside meta.
        for name in ("state_fqns", "state_is_param", "num_state_inputs", "user_inputs_spec", "loss_dict_keys"):
            if hasattr(graph_module, name):
                setattr(result, name, getattr(graph_module, name))
        return result
