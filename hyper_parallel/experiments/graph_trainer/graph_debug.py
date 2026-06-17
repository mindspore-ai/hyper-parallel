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
"""Debug helpers for inspecting traced train-step graphs."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_COLLECTIVE_KEYWORDS = (
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "all_reduce",
    "allreduce",
    "all_to_all",
    "alltoall",
    "_c10d",
    "c10d",
)


@dataclass
class CollectiveGraphSummary:
    """Collective-related node summary for an FX graph."""

    total_nodes: int
    collective_nodes: list[str]

    @property
    def collective_node_count(self) -> int:
        """Return the number of nodes whose target looks collective-related."""
        return len(self.collective_nodes)

    def contains(self, keyword: str) -> bool:
        """Return whether any collective node contains ``keyword``."""
        lowered = keyword.lower()
        compact = lowered.replace("_", "")
        return any(
            lowered in node.lower() or compact in node.lower().replace("_", "")
            for node in self.collective_nodes
        )


def _target_to_text(target: Any) -> str:
    """Convert an FX node target to stable debug text."""
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        return f"{target.__module__}.{target.__name__}"
    return str(target)


def summarize_collectives(
    graph_module: Any,
    keywords: tuple[str, ...] = _DEFAULT_COLLECTIVE_KEYWORDS,
) -> CollectiveGraphSummary:
    """Summarize collective-looking nodes in an FX graph.

    Args:
        graph_module: FX ``GraphModule`` returned by ``make_fx``.
        keywords: Lowercase substrings used to identify collective nodes.

    Returns:
        Collective node summary.
    """
    total_nodes = 0
    collective_nodes = []
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    for node in graph_module.graph.nodes:
        total_nodes += 1
        target_text = _target_to_text(node.target)
        target_lower = target_text.lower()
        if any(keyword in target_lower for keyword in lowered_keywords):
            collective_nodes.append(f"{node.op}:{target_text}")
    return CollectiveGraphSummary(total_nodes=total_nodes, collective_nodes=collective_nodes)


def dump_graph_debug(
    graph_module: Any,
    dump_dir: str | Path,
    name: str,
    summary: CollectiveGraphSummary | None = None,
) -> None:
    """Dump FX graph code, node targets, and optional collective summary."""
    output_dir = Path(dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{name}_code.py").write_text(graph_module.code, encoding="utf-8")
    node_lines = []
    for node in graph_module.graph.nodes:
        node_lines.append(f"{node.op}\t{node.name}\t{_target_to_text(node.target)}")
    (output_dir / f"{name}_nodes.txt").write_text("\n".join(node_lines), encoding="utf-8")

    if summary is not None:
        summary_lines = [
            f"total_nodes={summary.total_nodes}",
            f"collective_node_count={summary.collective_node_count}",
            *summary.collective_nodes,
        ]
        (output_dir / f"{name}_collectives.txt").write_text(
            "\n".join(summary_lines),
            encoding="utf-8",
        )
