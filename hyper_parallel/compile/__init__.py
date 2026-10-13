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
"""
HyperParallel Graph Mode - Graph-mode architecture for HyperParallel

Graph-mode architecture for automatic parallelization.

Core Features:
- Users write model code + parallel configuration
- Framework automatically handles all parallel logic
- Graph capture → Parallel partitioning → Communication-compute overlap → Execution

Usage Example:
    from hyper_parallel.compile import (
        GraphCompiler,
        GraphParallelPlan,
        GraphTrainer,
        PassConfig,
    )

    # Create model
    model = Llama3ForCausalLM(config)

    # Configure parallelism
    pass_config = PassConfig(enable_overlap=True)

    # Build parallel plan
    parallel_plan = GraphParallelPlan()
    parallel_plan.fsdp_mark_pattern("layers.*")

    # Compile + forward_backward only (no optimizer / training loop); model
    # inputs travel as kwargs, forwarded to train_fn(model, **inputs):
    compiler = GraphCompiler(model, train_fn, pass_config, parallel_plan)
    compiler.compile(input_ids=input_ids, labels=labels)
    loss = compiler.forward_backward(input_ids=input_ids, labels=labels)  # grads -> param.grad

    # Or drive the whole train/optimize loop -- the trainer composes a
    # GraphCompiler, compiles on the first batch, moves each input dict onto
    # its device, and owns the optimizer. data_iterable yields input dicts:
    trainer = GraphTrainer(model, train_fn, pass_config, parallel_plan)
    trainer.train(dataloader, max_steps=100, log_interval=10)
"""

from .compiler import GraphCompiler

from .graph_parallel_plan import (
    GraphParallelPlan,
    create_all_fsdp_plan,
    create_plan_from_yaml,
)

from .pass_config import PassConfig
from .recompute_config import RecomputeConfig

from .trainer import GraphTrainer

__all__ = [
    # Parallel plan
    "GraphParallelPlan",
    "create_plan_from_yaml",
    "create_all_fsdp_plan",
    # Config
    "PassConfig",
    "RecomputeConfig",
    # Compiler (compile + forward_backward only)
    "GraphCompiler",
    # Trainer
    "GraphTrainer",
]
