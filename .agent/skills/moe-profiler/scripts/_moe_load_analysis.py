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
"""Extract expert-load observations with standard-library statistics."""

import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from _moe_log_records import log_records


def _load_timeline(samples: dict) -> list:
    return [
        [step, layer, expert, statistics.mean(column), None]
        for (step, layer), vectors in sorted(samples.items())
        for expert, column in enumerate(zip(*vectors))
    ]


class LoadAnalyzer:
    """Summarize observed layers, including sparse/nonzero layer IDs."""

    def __init__(self, steps: Optional[int] = None, ranks: Optional[list] = None):
        self.steps = steps
        self.ranks = ranks

    def _collect_samples(self, log_files: List[str]) -> tuple:
        samples = defaultdict(list)
        observed_steps = set()
        for line, step, _ in log_records(log_files, self.steps, self.ranks):
            layer = re.search(r"\blayer[\s_=:]+(\d+)", line, re.I)
            counts = re.search(r"tokens_per_expert\s*=\s*(?:tensor\()?\[([^]]*)\]", line)
            if not layer or not counts or step is None:
                continue
            values = [int(value.strip()) for value in counts.group(1).split(",")]
            if not values or min(values) < 0:
                raise ValueError("Expert token counts must be nonnegative")
            samples[(step, int(layer.group(1)))].append(values)
            observed_steps.add(step)
        return samples, observed_steps


    def analyze(self, log_files: List[str]) -> Dict[str, Any]:
        """Return load tables; never infer unlogged routing scores."""
        samples, observed_steps = self._collect_samples(log_files)
        layers = sorted({layer for _, layer in samples})
        per_layer, summary = [], []
        expert_counts = set()
        for layer in layers:
            vectors = [vector for (_, item_layer), group in samples.items() if item_layer == layer for vector in group]
            widths = {len(vector) for vector in vectors}
            if len(widths) != 1:
                raise ValueError(f"Inconsistent expert counts for layer {layer}")
            width = widths.pop()
            expert_counts.add(width)
            columns = list(zip(*vectors))
            averages = [statistics.mean(column) for column in columns]
            mean_load = statistics.mean(averages)
            for expert, column in enumerate(columns):
                per_layer.append([layer, expert, statistics.mean(column), statistics.pstdev(column),
                                  min(column), max(column), statistics.mean(column) / mean_load if mean_load else None])
            flat = [value for vector in vectors for value in vector]
            gini = (sum((2 * index - width - 1) * value for index, value in enumerate(sorted(averages), 1))
                    / (width * sum(averages))) if sum(averages) else None
            summary.append([layer, statistics.mean([sum(vector) for vector in vectors]), width,
                            statistics.mean(flat), statistics.pstdev(flat),
                            max(averages) / mean_load if mean_load else None, gini])
        return {"num_steps": len(observed_steps), "num_layers": len(layers),
                "num_experts": next(iter(expert_counts)) if len(expert_counts) == 1 else None,
                "per_layer": per_layer, "timeline": _load_timeline(samples), "summary": summary}
