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
"""Configuration for the experimental graph trainer."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class GraphTrainerConfig:
    """Configuration for tracing and optionally compiling a train step.

    Args:
        tracing_mode: ``make_fx`` tracing mode. ``"real"`` keeps the first
            experiment close to real NPU execution and collective behavior.
        compile_backend: Optional backend used to compile the traced
            ``GraphModule``. ``None`` executes the FX graph directly.
        fullgraph: Whether to request full-graph compile when
            ``compile_backend`` is set.
        dump_graph: Whether rank 0 should dump graph code and node targets.
        dump_dir: Directory used for debug dumps.
    """

    tracing_mode: Literal["real", "fake", "symbolic"] = "real"
    compile_backend: str | None = None
    fullgraph: bool = True
    dump_graph: bool = False
    dump_dir: Path = Path("logs/simple_fsdp_graph_trainer")
