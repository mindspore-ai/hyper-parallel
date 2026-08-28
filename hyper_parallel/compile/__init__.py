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
        GraphTrainer,
        ParallelConfig,
        ShardingPlan,
    )

    # Create model
    model = Llama3ForCausalLM(config)

    # Configure parallelism
    parallel_config = ParallelConfig(enable_overlap=True)

    # Configure sharding plan
    sharding_plan = ShardingPlan()
    sharding_plan.fsdp_wrap_pattern("layers.*")

    # Create trainer
    trainer = GraphTrainer(model, train_fn, parallel_config, sharding_plan)

    # Training -- train compiles on the first batch, moves batches onto the
    # trainer's device, and drives the whole train/optimize loop.
    trainer.train(dataloader, max_steps=100, log_interval=10)
"""

from .sharding_config import (
    ShardingPlan,
    FSDPModuleConfig,
    create_sharding_plan_from_yaml,
    create_simple_sharding_plan,
)

from .parallel_config import (
    ParallelConfig,
    parallel_config,
)

from .trainer import GraphTrainer

__all__ = [
    # Sharding
    "ShardingPlan",
    "FSDPModuleConfig",
    "create_sharding_plan_from_yaml",
    "create_simple_sharding_plan",
    # Config
    "ParallelConfig",
    "parallel_config",
    # Trainer
    "GraphTrainer",
]
