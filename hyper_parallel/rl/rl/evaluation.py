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
"""Distributed evaluation for the synchronous RL trainer."""
import logging
from typing import Any, Callable, Optional
from rl.dataset.data_source import (
    PromptDataset,
    build_padded_evaluation_batches,
    build_prompt_records,
)
from rl.roles.rollout.worker import RolloutManager
from rl.utils.monitoring.metrics import select_round_robin_samples
from hyper_parallel import get_platform
platform = get_platform()
logger = logging.getLogger(__name__)
class Evaluator:
    """Run synchronized evaluation using an independently configured rollout manager."""
    def __init__(
        self,
        *,
        dataset: PromptDataset,
        collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
        rollout_manager: RolloutManager,
        device: Any,
        batch_size: int,
        max_samples: Optional[int],
        log_samples: int,
        progress_steps: int,
    ) -> None:
        """Initialize evaluation data, rollout runtime, and progress settings."""
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.rollout_manager = rollout_manager
        self.device = device
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.log_samples = log_samples
        self.progress_steps = progress_steps
        self.last_step = -1
    def run(self, step: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Generate the evaluation split and return rank-zero metrics and samples."""
        rank = platform.get_rank()
        world_size = platform.get_world_size()
        batches = build_padded_evaluation_batches(
            dataset_size=len(self.dataset),
            num_replicas=world_size,
            rank=rank,
            batch_size=self.batch_size,
            max_samples=self.max_samples,
        )
        requested = len(self.dataset) if self.max_samples is None else min(
            len(self.dataset), self.max_samples
        )
        if rank == 0:
            logger.info(
                "step=%d validation started: samples=%d, batches_per_rank=%d, "
                "batch_size_per_rank=%d",
                step,
                requested,
                len(batches),
                self.batch_size,
            )
        local = self._collect_local(step, rank, batches)
        gathered: list[Optional[dict[str, Any]]] = [None] * world_size
        platform.all_gather_object(gathered, local)
        self.last_step = step
        if rank != 0:
            return {}, []
        records = [record for record in gathered if record is not None]
        return self._summarize(step, records)
    def _collect_local(
        self,
        step: int,
        rank: int,
        batches: list[list[tuple[int, bool]]],
    ) -> dict[str, Any]:
        """Collect rank-local evaluation samples and reward moments."""
        record: dict[str, Any] = {
            "correct": 0.0,
            "total": 0,
            "generated_tokens": 0,
            "response_length": 0,
            "generation_seconds": 0.0,
            "samples": [],
        }
        for batch_index, entries in enumerate(batches, start=1):
            batch_record = self._evaluate_batch(
                step,
                rank,
                entries,
                self.log_samples - len(record["samples"]),
            )
            for key in (
                "correct",
                "total",
                "generated_tokens",
                "response_length",
                "generation_seconds",
            ):
                record[key] += batch_record[key]
            record["samples"].extend(batch_record["samples"])
            should_log = self.progress_steps > 0 and (
                batch_index % self.progress_steps == 0 or batch_index == len(batches)
            )
            if rank == 0 and should_log:
                logger.info(
                    "step=%d validation progress: %d/%d batches per rank",
                    step,
                    batch_index,
                    len(batches),
                )
        return record
    def _evaluate_batch(
        self,
        step: int,
        rank: int,
        entries: list[tuple[int, bool]],
        sample_limit: int,
    ) -> dict[str, Any]:
        """Generate and score one padded evaluation batch."""
        samples = [self.dataset[sample_index] for sample_index, _ in entries]
        batch = self.collate_fn(samples)
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        rollout = self.rollout_manager.generate(
            prompt_records=build_prompt_records(batch, input_ids, attention_mask),
            policy_version=step,
        )
        record: dict[str, Any] = {
            "correct": 0.0,
            "total": 0,
            "generated_tokens": 0,
            "response_length": 0,
            "generation_seconds": rollout.generation_seconds,
            "samples": [],
        }
        for local_index, ((_, valid), response) in enumerate(
            zip(entries, rollout.responses)
        ):
            if not valid:
                continue
            reward = float(rollout.rewards[local_index].item())
            response_length = int(rollout.action_mask[local_index].sum(dim=0).item())
            record["correct"] += reward
            record["total"] += 1
            record["generated_tokens"] += response_length
            record["response_length"] += response_length
            if len(record["samples"]) < sample_limit:
                record["samples"].append(
                    {
                        "step": step,
                        "rank": rank,
                        "prompt": batch["prompts"][local_index],
                        "response": response,
                        "ground_truth": batch["ground_truths"][local_index],
                        "extracted_answer": rollout.trajectories[
                            local_index
                        ].metadata.get("extracted_answer"),
                        "reward": reward,
                    }
                )
        return record
    def _summarize(
        self,
        step: int,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Merge distributed evaluation metrics and bounded samples."""
        correct = sum(float(record["correct"]) for record in records)
        total = sum(int(record["total"]) for record in records)
        generated_tokens = sum(int(record["generated_tokens"]) for record in records)
        response_length = sum(int(record["response_length"]) for record in records)
        generation_seconds = max(
            float(record["generation_seconds"]) for record in records
        )
        accuracy = correct / max(total, 1)
        metrics = {
            "validation/accuracy": accuracy,
            "validation/correct": correct,
            "validation/total": float(total),
            "validation/response_length_mean": response_length / max(total, 1),
            "validation/generated_tokens": float(generated_tokens),
            "validation/generation_seconds": generation_seconds,
            "validation/tokens_per_second": (
                generated_tokens / max(generation_seconds, 1.0e-9)
            ),
        }
        logger.info(
            "step=%d validation completed: accuracy=%.6f (%d/%d)",
            step,
            accuracy,
            int(correct),
            total,
        )
        return metrics, select_round_robin_samples(records, self.log_samples)
__all__ = ["Evaluator"]
