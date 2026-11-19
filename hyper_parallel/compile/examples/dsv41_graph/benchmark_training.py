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
"""Measure native training and synchronized optimizer timing on each NPU rank."""

import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


class BenchmarkTrainer(TextTrainer):
    """Record allocator peaks separately from expensive numerical references."""

    def __init__(self, config: TrainerConfig) -> None:
        """Build the native trainer and start fresh per-rank metric files."""
        super().__init__(config)
        self.optimizer_seconds = 0.0
        directory = Path(os.environ.get("METRICS_DIR", "output/dsv41_graph/muon_metrics"))
        self.metrics_path = directory / f"rank{self.base.global_rank}.jsonl"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text("", encoding="utf-8")
        self.native_step = self.base.step_optimizers_and_schedulers
        self.base.step_optimizers_and_schedulers = self.timed_optimizer_step
        torch.npu.reset_peak_memory_stats()

    def timed_optimizer_step(self) -> None:
        """Include queued NPU work and optimizer communication in wall timing."""
        torch.npu.synchronize()
        start = time.perf_counter()
        self.native_step()
        torch.npu.synchronize()
        self.optimizer_seconds = time.perf_counter() - start

    def on_step_end(self, **kwargs: Any) -> None:
        """Retain native logging and append one allocator/timing observation."""
        super().on_step_end(**kwargs)
        row = {
            "rank": self.base.global_rank,
            "step": self.base.state.global_step,
            "optimizer_seconds": self.optimizer_seconds,
            "allocated_bytes": torch.npu.memory_allocated(),
            "reserved_bytes": torch.npu.memory_reserved(),
            "max_allocated_bytes": torch.npu.max_memory_allocated(),
            "max_reserved_bytes": torch.npu.max_memory_reserved(),
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    BenchmarkTrainer(parse_training_args()).train()
