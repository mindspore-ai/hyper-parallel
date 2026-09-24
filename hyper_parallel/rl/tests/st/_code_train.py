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
"""Observe production code training, real parameter updates and TP-safe evaluation."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from rl.trainer import SyncTrainer


class CodeObservedTrainer(SyncTrainer):
    """Add evidence probes without replacing data, generated actions or rewards."""

    def __init__(self, config: dict, output: Path) -> None:
        """Bind evidence output before constructing the production trainer."""
        self.output = output
        self.records: list[dict] = []
        self.before: dict[str, torch.Tensor] = {}
        super().__init__(config)

    def _samples(self) -> dict[str, torch.Tensor]:
        """Capture finite parameter samples to detect actual training updates."""
        samples = {}
        for name, parameter in self.actor.actor_model.named_parameters():
            if not parameter.requires_grad:
                continue
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            flat = local.detach().reshape(-1)
            if flat.numel():
                samples[name] = flat[::max(1, flat.numel() // 1024)][:1024].float().cpu().clone()
        if not samples or any(not torch.isfinite(value).all() for value in samples.values()):
            raise RuntimeError("Missing or non-finite parameter samples")
        return samples

    def _write(self, name: str, evidence: Any) -> None:
        if dist.get_rank() == 0:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / name).write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def _build_runtime(self) -> None:
        """Attach evidence collection to the production evaluator."""
        super()._build_runtime()
        if self.evaluator is None:
            raise ValueError("Code acceptance requires enabled evaluation")
        original = self.evaluator.run

        def evaluate(step: int) -> tuple[dict, list]:
            """Observe production evaluation without altering its samples or scores."""
            metrics, samples = original(step)

            def record() -> None:
                """Validate aggregate counts and persist rank-zero evaluation evidence."""
                if dist.get_rank() != 0:
                    return
                count = len(self.evaluator.dataset)
                if self.evaluator.max_samples is not None:
                    count = min(count, self.evaluator.max_samples)
                if (metrics["validation/total"] != count or count <= 0
                        or not all(math.isfinite(value) for value in metrics.values())
                        or metrics["validation/generated_tokens"] <= 0
                        or not math.isclose(metrics["validation/accuracy"], metrics["validation/correct"] / count)):
                    raise RuntimeError("Evaluation counts or metrics violate the code acceptance contract")
                if any("ground_truth" in sample for sample in samples):
                    raise RuntimeError("Structured private tests appeared in evaluation logs")
                self._write("evaluation.json", {"step": step, "metrics": metrics, "samples": samples})

            self._run_rank_synchronized("code evaluation evidence", record)
            return metrics, samples

        self.evaluator.run = evaluate

    def _train_step(self, batch: dict) -> None:
        self.before = self._run_rank_synchronized("before-update parameter samples", self._samples)
        super()._train_step(batch)

    @staticmethod
    def _outcome(trajectory: Any, step: int) -> dict:
        """Validate one complete code judgement while retaining no private tests."""
        components, metadata = trajectory.reward_components, trajectory.metadata
        success = float(components["success"])
        if (trajectory.policy_version != step - 1 or components["total"] <= 0
                or success != float(components["passed"] == components["total"])
                or trajectory.reward != success):
            raise RuntimeError("Code reward does not represent all declared tests")
        for key in ("status", "candidate_id", "runtime_version", "test_version", "judge_version", "finish_reason"):
            if not isinstance(metadata.get(key), str) or not metadata[key]:
                raise RuntimeError(f"Missing code outcome evidence: {key}")
        return {"prompt_id": trajectory.prompt_id, "components": dict(components),
                "status": metadata["status"], "finish_reason": metadata["finish_reason"],
                "candidate_id": metadata["candidate_id"]}

    def _evidence(self, values: dict) -> dict:
        """Validate policy versions and collect code reward and update evidence."""
        step, rollout, update = values["step"], values["rollout"], values["actor_update"]
        loss, norm = float(update.total_loss), float(update.gradient_norm)
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise RuntimeError("Non-finite training loss or gradient norm")
        valid = rollout.old_log_probs[rollout.loss_action_mask.bool()]
        if not valid.numel() or not torch.isfinite(valid).all():
            raise RuntimeError("Invalid sampled action log probabilities")
        if rollout.worker_policy_version != step - 1 or self.rollout_engine.policy_version != step:
            raise RuntimeError("Sampling or published policy version mismatch")
        outcomes = [self._outcome(trajectory, step) for trajectory in rollout.trajectories]
        after = self._samples()
        if after.keys() != self.before.keys():
            raise RuntimeError("Parameter identity changed during training")
        deltas = {name: float((after[name] - value).abs().max()) for name, value in self.before.items()}
        return {"rank": dist.get_rank(), "dp_rank": int(self.parallel_dims.dp_rank),
                "tp_rank": int(self.parallel_dims.tp_rank), "loss": loss, "gradient_norm": norm, "outcomes": outcomes,
                "sampled_version": step - 1, "published_version": step, "parameter_samples": deltas}

    def _complete_step(self, **values: Any) -> None:
        """Observe updates and explicitly evaluate the final policy without requiring a checkpoint."""
        evidence = self._run_rank_synchronized("code step evidence", lambda: self._evidence(values))
        ranks: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, evidence)
        self.records.append({"step": values["step"], "ranks": ranks})

        def record() -> None:
            """Check TP outcome equality and persist completed-step update evidence."""
            groups: dict[int, list] = {}
            for rank in ranks:
                groups.setdefault(rank["dp_rank"], []).append(rank["outcomes"])
            if any(any(outcome != group[0] for outcome in group[1:]) for group in groups.values()):
                raise RuntimeError("TP siblings received different candidate identities or judge results")
            changed = any(delta > 0 for row in self.records for rank in row["ranks"]
                          for delta in rank["parameter_samples"].values())
            self._write("training.json", {"config": self.resolved_config, "steps": self.records,
                                          "parameter_sample_changed": changed})
            if values["step"] == self.state.max_steps and (len(self.records) < 2 or not changed):
                raise RuntimeError("Acceptance requires two steps and a real parameter sample change")

        self._run_rank_synchronized("write code training evidence", record)
        super()._complete_step(**values)
        step = values["step"]
        if step == self.state.max_steps and self.evaluator.last_step != step:
            self.evaluator.run(step)


def main(argv: list[str] | None = None) -> None:
    """Run one complete YAML; RL_ST_RESULT_DIR selects persistent evidence output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(os.environ["RL_ST_RESULT_DIR"])
    trainer = CodeObservedTrainer(config, output)
    trainer.train()
    if trainer.evaluator.last_step != trainer.state.max_steps:
        raise RuntimeError("Final policy was not evaluated")
    if int(os.environ.get("RANK", "0")) == 0:
        (output / "completed.json").write_text(json.dumps({"status": "passed", "steps": len(trainer.records)}) + "\n",
                                               encoding="utf-8")


def test_training() -> None:
    """Run the configured production recipe with update and publication checks."""
    main([os.environ["RL_ST_CONFIG"]])


if __name__ == "__main__":
    main()
