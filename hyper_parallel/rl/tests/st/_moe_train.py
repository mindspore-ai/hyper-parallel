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
"""Explicit torchrun worker observing the production MoE training pipeline."""

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


class ObservedTrainer(SyncTrainer):
    """Keep training unchanged while recording local expert and policy evidence."""

    def __init__(self, config: dict, result_dir: Path, build_only: bool) -> None:
        """Bind observation output before constructing the production runtime."""
        self.result_dir = result_dir
        self.build_only = build_only
        self.records: list[dict[str, Any]] = []
        self.before: dict[str, torch.Tensor] = {}
        super().__init__(config)

    def _expert_samples(self) -> dict[str, torch.Tensor]:
        """Sample every expert parameter's local storage without gathering weights."""
        samples = {}
        for name, parameter in self.actor.actor_model.named_parameters():
            if not name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                continue
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            flat = local.detach().reshape(-1)
            if flat.numel():
                stride = max(1, flat.numel() // 1024)
                samples[name] = flat[::stride][:1024].float().cpu().clone()
        if not samples:
            raise RuntimeError("No local GroupedExperts storage was available to observe")
        if any(not torch.isfinite(value).all() for value in samples.values()):
            raise RuntimeError("Non-finite expert parameter sample")
        return samples

    def _write_result(self, filename: str, result: dict[str, Any]) -> None:
        if dist.get_rank() == 0:
            self.result_dir.mkdir(parents=True, exist_ok=True)
            (self.result_dir / filename).write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )

    def _build_runtime(self) -> None:
        if self.build_only:
            self._build_models_and_optimizers()
            samples = self._run_rank_synchronized("expert build samples", self._expert_samples)
            self._run_rank_synchronized("build evidence", lambda: self._write_result("build.json", {
                "status": "built", "world_size": dist.get_world_size(),
                "expert_parameter_names": sorted(samples), "config": self.resolved_config,
            }))
        else:
            super()._build_runtime()

    def _train_step(self, batch: dict[str, Any]) -> None:
        self.before = self._run_rank_synchronized("pre-update expert samples", self._expert_samples)
        super()._train_step(batch)

    def _step_evidence(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validate rollout metrics and capture expert parameter changes."""
        step = values["step"]
        update = values["actor_update"]
        rollout = values["rollout"]
        loss, norm = float(update.total_loss), float(update.gradient_norm)
        if not math.isfinite(loss) or not math.isfinite(norm):
            raise RuntimeError(f"Non-finite training metrics: loss={loss}, gradient_norm={norm}")
        sampled = {trajectory.policy_version for trajectory in rollout.trajectories}
        if sampled != {step - 1} or rollout.worker_policy_version != step - 1:
            raise RuntimeError(f"Unexpected sampled versions: {sampled}, worker={rollout.worker_policy_version}")
        published = self.rollout_engine.policy_version
        if published != step:
            raise RuntimeError(f"Policy publication failed: expected={step}, actual={published}")
        after = self._expert_samples()
        if after.keys() != self.before.keys():
            raise RuntimeError("Expert parameter identity changed during training")
        deltas = {name: float((after[name] - before).abs().max()) for name, before in self.before.items()}
        valid = rollout.old_log_probs[rollout.loss_action_mask.bool()]
        if not valid.numel() or not torch.isfinite(valid).all():
            raise RuntimeError("Rollout action log probabilities are empty or non-finite")
        return {
            "rank": dist.get_rank(), "step": step, "loss": loss, "gradient_norm": norm,
            "sampled_version": step - 1, "published_version": published,
            "action_tokens": valid.numel(), "reward_sum": float(rollout.rewards.sum()),
            "expert_sample_max_deltas": deltas,
        }

    def _complete_step(self, **values: Any) -> None:
        """Gather and persist evidence for the completed MoE update."""
        evidence = self._run_rank_synchronized("step evidence", lambda: self._step_evidence(values))
        ranks: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, evidence)
        self.records.append({"step": values["step"], "ranks": ranks})

        def validate_and_write() -> None:
            """Persist evidence and require an actual expert update at completion."""
            final = values["step"] == self.state.max_steps
            changed = any(delta > 0 for record in self.records for rank in record["ranks"]
                          for delta in rank["expert_sample_max_deltas"].values())
            result = {"status": "observed", "expert_sample_changed": changed,
                      "steps": self.records, "config": self.resolved_config}
            self._write_result("training.json", result)
            if final and (len(self.records) < 2 or not changed):
                raise RuntimeError("Acceptance requires two completed steps and an observed expert update")

        self._run_rank_synchronized("write training evidence", validate_and_write)
        super()._complete_step(**values)


def main(argv: list[str] | None = None) -> None:
    """Run a supplied recipe; RL_ST_RESULT_DIR selects the evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result_dir = Path(os.environ["RL_ST_RESULT_DIR"])
    trainer = ObservedTrainer(config, result_dir, args.build_only)
    if args.build_only:
        trainer._cleanup()  # pylint: disable=protected-access
    else:
        trainer.train()
        if int(os.environ.get("RANK", "0")) == 0:
            (result_dir / "completed.json").write_text(
                json.dumps({"status": "passed", "steps": len(trainer.records)}) + "\n", encoding="utf-8",
            )


def test_training() -> None:
    """Run the configured production recipe with update and publication checks."""
    main([os.environ["RL_ST_CONFIG"]])


if __name__ == "__main__":
    main()
