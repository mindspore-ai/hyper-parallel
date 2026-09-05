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
"""Validate that an already-parallelized AutoModels EP region entered FX."""

from collections import Counter
from typing import Any

from ..base import GraphPass


def _is_functional_all_to_all_single(target: Any) -> bool:
    """Return whether a target owns the canonical functional collective schema."""
    schema = getattr(target, "_schema", None)
    return getattr(schema, "name", None) == "_c10d_functional::all_to_all_single"


def _uses_static_ep_splits(node: Any, ep_degree: int) -> bool:
    """Return whether an All-to-All node has the fixed-capacity EP contract."""
    if len(node.args) < 4:
        return False
    output_splits, input_splits = node.args[1:3]
    if not isinstance(output_splits, (list, tuple)):
        return False
    if not isinstance(input_splits, (list, tuple)):
        return False
    if len(output_splits) != ep_degree or len(input_splits) != ep_degree:
        return False
    output_split_keys = tuple(str(split) for split in output_splits)
    input_split_keys = tuple(str(split) for split in input_splits)
    return (
        len(set(output_split_keys)) == 1
        and output_split_keys == input_split_keys
    )


def _get_collective_group_name(node: Any) -> Any:
    """Return the registered group name from a functional collective node."""
    if len(node.args) >= 4:
        return node.args[3]
    return node.kwargs.get("group_name")


def _is_static_ep_collective(
    node: Any, group_names: set[str], ep_degree: int
) -> bool:
    """Return whether a node is a captured fixed-capacity EP collective."""
    if node.op != "call_function":
        return False
    if not _is_functional_all_to_all_single(node.target):
        return False
    if _get_collective_group_name(node) not in group_names:
        return False
    return _uses_static_ep_splits(node, ep_degree)


class ExpertParallelPass(GraphPass):
    """Evidence pass for capture-first EP.

    Dynamic AutoModels owns expert partitioning.  Consequently this pass does
    not synthesize a second EP implementation; it rejects a graph that lost
    the dynamic EP collectives during capture.
    """

    name = "expert_parallel"

    @property
    def mesh_dim(self) -> str:
        """Return the logical mesh dimension validated by this pass."""
        return "ep"

    def run(
        self,
        graph_module: Any,
        pass_config: Any,
        **kwargs: Any,
    ) -> Any:
        """Validate exact EP All-to-All provenance and capture completeness."""
        del kwargs
        metadata = getattr(graph_module, "ep_capture_metadata", None)
        group_names = set(metadata.get("group_names", ())) if metadata else set()
        expected_count = (
            metadata.get("expected_collective_count", 0) if metadata else 0
        )
        collective_nodes = []
        for submodule in graph_module.modules():
            graph = getattr(submodule, "graph", None)
            if graph is None:
                continue
            collective_nodes.extend(
                node
                for node in graph.nodes
                if _is_static_ep_collective(
                    node, group_names, pass_config.ep_degree
                )
            )
        collective_count = len(collective_nodes)
        graph_module.ep_collective_count = collective_count
        if not getattr(pass_config, "require_ep_collectives", True):
            return graph_module
        if not group_names or expected_count < 1:
            raise RuntimeError(
                "EP is enabled, but the graph has no routed-EP capture metadata. "
                "Apply the AutoModels dynamic EP plan before constructing "
                "GraphTrainer."
            )
        if collective_count != expected_count:
            raise RuntimeError(
                "EP capture is incomplete: expected "
                f"{expected_count} fixed-capacity All-to-All nodes on groups "
                f"{sorted(group_names)}, found {collective_count}."
            )
        expected_by_group = metadata.get("collective_counts_by_group")
        actual_by_group = Counter(
            _get_collective_group_name(node) for node in collective_nodes
        )
        if expected_by_group is not None and actual_by_group != expected_by_group:
            raise RuntimeError(
                "EP capture group mismatch: expected "
                f"{expected_by_group}, found {dict(actual_by_group)}."
            )
        return graph_module


__all__ = ["ExpertParallelPass"]
