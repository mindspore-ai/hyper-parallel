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
"""Policy tagging and in-place activation rematerialization for Hyper graphs.

Uses the tag-then-duplicate design of TorchTitan GraphTrainer, implemented
against Hyper's module-stack and backward metadata. No partitioner private
APIs or TorchTitan runtime dependencies are required.
"""

# Torch-only graph passes follow the repository backend policy.
# pylint: disable=forbidden-backend-import

from collections import Counter, defaultdict
import fnmatch
import logging
import operator
from typing import Any

import torch
from torch import fx
from torch.utils.checkpoint import CheckpointPolicy as TorchCheckpointPolicy
from torch.utils._pytree import tree_leaves

from hyper_parallel.compile.pass_config import PassConfig
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy
from ..base import GraphPass
from ...recompute_config import RecomputeConfig

_LOG = logging.getLogger(__name__)
_RECOMPUTE = {CheckpointPolicy.MUST_RECOMPUTE, CheckpointPolicy.PREFER_RECOMPUTE}
_MATMUL = {"mm", "matmul", "linear", "_grouped_mm", "_scaled_grouped_mm"}
_EXPENSIVE = _MATMUL | {"addmm", "bmm", "_scaled_mm", "max", "convolution"}
_COMMUNICATION_OPS = {
    "all_to_all_single", "all_gather", "all_gather_into_tensor", "all_reduce", "allreduce_",
    "reduce_scatter", "reduce_scatter_tensor", "wait_tensor", "send", "recv",
}


def _module_paths(node: fx.Node) -> list[str]:
    return [entry[0] for entry in node.meta.get("nn_module_stack", {}).values()]


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _regions(nodes: list[fx.Node], config: RecomputeConfig) -> dict[fx.Node, str]:
    result = {}
    matched = set()
    for node in nodes:
        if node.meta.get("autograd_backward") or node.op != "call_function":
            continue
        paths = _module_paths(node)
        for pattern in config.module_patterns:
            if any(fnmatch.fnmatchcase(path, pattern) for path in paths):
                matched.add(pattern)
        if any(_matches(path, config.exclude_module_patterns) for path in paths):
            continue
        region = next((path for path in paths if _matches(path, config.module_patterns)), None)
        if region is not None:
            result[node] = region
    missing = set(config.module_patterns) - matched
    if missing:
        raise ValueError(f"Recompute module patterns matched no traced module: {sorted(missing)}")
    return result


def _schema_inputs(node: fx.Node, schema: Any) -> list[tuple[Any, list[fx.Node]]]:
    result = []
    for index, argument in enumerate(schema.arguments):
        value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
        inputs = []
        fx.map_arg(value, inputs.append)
        result.append((argument, inputs))
    return result


def _mutation_barriers(nodes: list[fx.Node]) -> set[fx.Node]:
    """Protect mutated storage, aliases and reads that could observe a later value.

    Inspect both phases: a backward write also invalidates a forward value.
    Tuple aliases conservatively share one group, sacrificing precision for safety.
    """
    parent = {node: node for node in nodes}

    def _root(node):
        while parent[node] is not node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def _merge(left, right):
        parent[_root(left)] = _root(right)

    writes = set()
    for node in nodes:
        if node.target is operator.getitem and node.all_input_nodes:
            _merge(node, node.all_input_nodes[0])
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        aliases_output = any(result.alias_info is not None for result in schema.returns)
        for argument, inputs in _schema_inputs(node, schema):
            alias = argument.alias_info
            if alias is None:
                continue
            for source in inputs:
                if aliases_output:
                    _merge(node, source)
                if alias.is_write:
                    writes.add(source)
                    writes.add(node)
    groups = {_root(node) for node in writes}
    aliases = {node for node in nodes if _root(node) in groups}
    return aliases | {user for node in aliases for user in node.users}


def _safety_reason(node: fx.Node, barriers: set[fx.Node]) -> str:
    if node in barriers:
        return "mutation_or_alias"
    if node.target is operator.getitem:
        return ""
    if not isinstance(node.target, torch._ops.OpOverload):
        return "opaque_or_higher_order_op"
    tags = node.target.tags
    if torch.Tag.nondeterministic_seeded in tags:
        return "rng"
    if torch.Tag.nondeterministic_bitwise in tags:
        return "nondeterministic"
    namespace = node.target.namespace
    op_name = str(node.target).split(".")[1]
    if namespace not in ("aten", "prims") or op_name in _COMMUNICATION_OPS:
        return "communication_or_custom_op"
    if node.is_impure():
        return "side_effect"
    value = node.meta.get("val")
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool)):
        return "scalar"
    return ""


def _normalize_policy(value: Any) -> CheckpointPolicy:
    if not isinstance(value, (CheckpointPolicy, TorchCheckpointPolicy)):
        raise ValueError("recompute.policy_fn must return a Hyper or Torch CheckpointPolicy")
    if value.name not in ("MUST_SAVE", "PREFER_SAVE", "MUST_RECOMPUTE", "PREFER_RECOMPUTE"):
        raise ValueError(f"Graph recompute does not support policy {value.name}; swap requires a graph offload pass")
    return CheckpointPolicy[value.name]


def _default_policy(node: fx.Node, config: RecomputeConfig, count: Counter, region: str) -> CheckpointPolicy:
    if str(node.target) in config.save_ops:
        return CheckpointPolicy.MUST_SAVE
    if config.mode == "full":
        return CheckpointPolicy.PREFER_RECOMPUTE
    name = str(node.target).split(".")
    op_name = name[1] if len(name) > 1 else ""
    if op_name in _MATMUL:
        count[region] += 1
        if config.selective_policy == "eager" and count[region] % 2 == 0:
            return CheckpointPolicy.PREFER_RECOMPUTE
    if op_name in _EXPENSIVE or "attention" in op_name:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _tag_policy(nodes: list[fx.Node], config: RecomputeConfig) -> tuple[dict, dict]:
    regions = _regions(nodes, config)
    barriers = _mutation_barriers(nodes)
    decisions, reasons = {}, {}
    counts = Counter()
    seen_ops = {str(node.target) for node in regions}
    missing_ops = set(config.save_ops) - seen_ops
    if missing_ops:
        raise ValueError(f"recompute.save_ops matched no selected operator: {sorted(missing_ops)}")
    for node, region in regions.items():
        policy = (_normalize_policy(config.policy_fn(node)) if config.policy_fn else
                  _default_policy(node, config, counts, region))
        reason = _safety_reason(node, barriers)
        if not reason and any(
            not user.meta.get("autograd_backward") and regions.get(user) != region
            for user in node.users
        ):
            reason = "region_boundary"
        if reason:
            if config.policy_fn and policy == CheckpointPolicy.MUST_RECOMPUTE:
                raise ValueError(f"Cannot MUST_RECOMPUTE {node.name}: {reason}")
            policy = CheckpointPolicy.MUST_SAVE
            reasons[node] = reason
        decisions[node] = policy
    return decisions, reasons


def _rematerialize(graph_module: fx.GraphModule, decisions: dict) -> int:
    graph = graph_module.graph
    originals = list(graph.nodes)
    targets = {}
    for consumer in originals:
        if not consumer.meta.get("autograd_backward"):
            continue
        pending = list(consumer.all_input_nodes)
        while pending:
            node = pending.pop()
            if decisions.get(node) not in _RECOMPUTE or node in targets:
                continue
            targets[node] = consumer
            pending.extend(node.all_input_nodes)

    copies = {}
    for node in originals:
        if node not in targets:
            continue
        with graph.inserting_before(targets[node]):
            duplicate = graph.node_copy(node, lambda arg: copies.get(arg, arg))
        duplicate.name = node.name + "_recomputed"
        duplicate.meta = dict(node.meta)
        if "custom" in duplicate.meta:
            duplicate.meta["custom"] = dict(duplicate.meta["custom"])
        duplicate.meta["autograd_backward"] = True
        duplicate.meta["recompute_source"] = node.name
        duplicate.meta.pop("recompute", None)
        copies[node] = duplicate
    # Only rewrite original backward nodes; copied forward ops already use the
    # dependency mapping, and rewriting their own inputs can introduce cycles.
    for node in originals:
        if node.meta.get("autograd_backward"):
            node.args = fx.map_arg(node.args, lambda arg: copies.get(arg, arg))
            node.kwargs = fx.map_arg(node.kwargs, lambda arg: copies.get(arg, arg))
    for node in reversed(originals):
        if node in copies and not node.users and not node.is_impure():
            graph.erase_node(node)
    return len(copies)


def _saved_activation_bytes(nodes: list[fx.Node]) -> int:
    """Estimate unique non-input storage crossing forward/backward edges.

    This is a static retained-activation estimate, not measured peak memory.
    Views and tuple elements sharing storage count once. Tracing is static-shape.
    """
    external = set()
    retained = {}
    for node in nodes:
        tensors = [value for value in tree_leaves(node.meta.get("val")) if isinstance(value, torch.Tensor)]
        if node.op in ("placeholder", "get_attr"):
            external.update(value.untyped_storage()._cdata for value in tensors)
        if node.op in ("placeholder", "get_attr") or node.meta.get("autograd_backward"):
            continue
        if not any(user.meta.get("autograd_backward") for user in node.users):
            continue
        for value in tensors:
            storage = value.untyped_storage()
            retained[storage._cdata] = storage.nbytes()
    return sum(size for key, size in retained.items() if key not in external)


class RecomputePass(GraphPass):
    """Insert activation recomputation before FSDP parameter lifetime planning."""

    name = "recompute"

    def run(self, graph_module: fx.GraphModule, pass_config: PassConfig, **kwargs: Any) -> fx.GraphModule:
        """Tag selected regions and duplicate only backward-required dependencies.

        Args:
            graph_module: Joint graph with module stacks and backward metadata.
            pass_config: Compiler config containing a RecomputeConfig.
            **kwargs: Shared pipeline arguments (unused).

        Returns:
            The same graph module, with ``meta['recompute_report']`` diagnostics.
        """
        del kwargs
        pass_config.validate()
        config = pass_config.recompute
        if config.mode == "none":
            return graph_module
        if graph_module.meta.get("recompute_report") is not None:
            raise ValueError("RecomputePass has already run; retrace to change its configuration")
        nodes = list(graph_module.graph.nodes)
        if not any(node.meta.get("autograd_backward") for node in nodes):
            raise ValueError("RecomputePass requires a joint graph with autograd_backward metadata")
        decisions, reasons = _tag_policy(nodes, config)
        for node, policy in decisions.items():
            node.meta["recompute"] = policy
            if node in reasons:
                node.meta["recompute_save_reason"] = reasons[node]
        retained_before = _saved_activation_bytes(nodes)
        count = _rematerialize(graph_module, decisions)
        graph_module.graph.lint()
        graph_module.recompile()
        by_reason = defaultdict(list)
        for node, reason in reasons.items():
            by_reason[reason].append(node.name)
        report = {"mode": config.mode, "recomputed_nodes": count,
                  "saved_nodes": sum(policy not in _RECOMPUTE for policy in decisions.values()),
                  "save_reasons": dict(by_reason),
                  "retained_activation_bytes_before": retained_before,
                  "retained_activation_bytes_after": _saved_activation_bytes(list(graph_module.graph.nodes))}
        graph_module.meta["recompute_report"] = report
        _LOG.info("Graph recompute: %s", report)
        if not count:
            _LOG.warning("Recompute selection produced no rematerialized nodes; inspect recompute_report")
        return graph_module
