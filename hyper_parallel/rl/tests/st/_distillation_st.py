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
"""Real teacher and optimizer acceptance worker, launched explicitly."""

# Tests inspect package-owned state and stage boundaries without public runtime hooks.
# pylint: disable=protected-access

import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist
import yaml

from rl.trainer.build import build_trainer


def main() -> None:
    """Reject mocked resources and verify every published update."""
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    distillation = config.get("distillation", {})
    if not distillation.get("enabled", False) or config["train"]["max_steps"] < 2:
        raise ValueError("OPD ST requires enabled real teachers and at least two steps")
    config["train"]["learning_gate"] = {"enabled": True, "min_gradient_norm": 1.0e-12, "require_mixed_rewards": False}
    completed = []
    teacher_samples = dict.fromkeys(distillation["teachers"], 0.0)
    teacher_paths = [Path(entry["model_path"]).resolve() for entry in distillation["teachers"].values()]
    if len(set(teacher_paths)) != len(teacher_paths):
        raise ValueError("MOPD ST requires distinct teacher checkpoints, not replicas of one model")
    evidence = []
    mean_limit = float(os.environ.get("RL_OPD_ST_MAX_MEAN_LOGPROB_DIFF", "0.1"))
    max_limit = float(os.environ.get("RL_OPD_ST_MAX_LOGPROB_DIFF", "1.0"))
    if not all(math.isfinite(value) and value > 0 for value in (mean_limit, max_limit)):
        raise ValueError("OPD ST probability tolerances must be finite and positive")

    # Apply the same numerical limits in the production gate before any update.
    distillation.setdefault("probability_gate", {}).update(
        max_mean_logprob_diff=mean_limit, max_logprob_diff=max_limit,
    )
    trainer = build_trainer(config)
    original = trainer._complete_step

    def verify_step(**kwargs: Any) -> None:
        """Inspect actual update outputs before accepting the training step."""
        update = kwargs["actor_update"]
        diagnostics = kwargs["diagnostic_metrics"]
        if update.gradient_norm <= 0 or diagnostics.get("distillation/supervised_tokens", 0) <= 0:
            raise RuntimeError("OPD ST requires nonzero student gradients and real teacher tokens")
        def validate_update_diagnostics() -> None:
            """Verify global probability and task diagnostics on rank zero."""
            if dist.get_rank() == 0:
                if trainer.distillation_config.use_task_rewards:
                    advantage_min = diagnostics.get("train/advantage_min")
                    advantage_max = diagnostics.get("train/advantage_max")
                    if (advantage_min is None or advantage_max is None
                            or not all(math.isfinite(value) for value in (advantage_min, advantage_max))
                            or max(abs(advantage_min), abs(advantage_max)) == 0):
                        raise RuntimeError("Joint RL+OPD ST requires nonzero task advantages on valid action tokens")
                    if not math.isfinite(update.policy_loss):
                        raise RuntimeError("Joint RL+OPD ST requires a finite task policy loss")
                probability_diff = diagnostics.get("training/rollout_log_probs_diff_max")
                if probability_diff is None or not math.isfinite(probability_diff):
                    raise RuntimeError("OPD ST requires real rollout-versus-trainer probability diagnostics")
                mean_diff = diagnostics.get("training/rollout_log_probs_diff_mean")
                if (mean_diff is None or not math.isfinite(mean_diff)
                        or mean_diff > mean_limit or probability_diff > max_limit):
                    raise RuntimeError(
                        f"OPD ST probability mismatch: mean={mean_diff} (limit {mean_limit}), "
                        f"max={probability_diff} (limit {max_limit})"
                    )

        trainer._run_rank_synchronized("OPD update diagnostic verification", validate_update_diagnostics)
        if trainer.rollout_engine.policy_version != kwargs["step"]:
            raise RuntimeError("OPD ST requires the updated policy to be published")
        main_params = bool(config["train"]["optimizer"].get("fp32_main_params", False))
        optimizer = trainer.actor.optimizer
        optimizer_dtypes = sorted({str(parameter.dtype) for group in optimizer.param_groups
                                   for parameter in group["params"]})
        if main_params:
            state_dtypes = {str(value.dtype) for leaf in optimizer.chained_optimizers
                            for state in leaf.state.values() for value in state.values()
                            if isinstance(value, torch.Tensor) and value.is_floating_point()}
            if optimizer_dtypes != ["torch.float32"] or state_dtypes != {"torch.float32"}:
                raise RuntimeError("OPD ST requires actual FP32 main parameters and FP32 optimizer moments")
        counts = {key: diagnostics.get(f"distillation/teacher_samples/{key}", 0.0) for key in teacher_samples}
        for key, count in counts.items():
            teacher_samples[key] += count
        evidence.append({
            "step": kwargs["step"], "gradient_norm": float(update.gradient_norm),
            "supervised_tokens": diagnostics["distillation/supervised_tokens"],
            "teacher_samples": counts, "policy_version": trainer.rollout_engine.policy_version,
            "optimizer_steps": update.optimizer_steps,
            "task_policy_loss": float(update.policy_loss),
            "task_advantage_min": diagnostics.get("train/advantage_min"),
            "task_advantage_max": diagnostics.get("train/advantage_max"),
            "optimizer_fp32_main_params": main_params, "optimizer_parameter_dtypes": optimizer_dtypes,
            "mean_logprob_difference": diagnostics.get("training/rollout_log_probs_diff_mean"),
            "max_logprob_difference": diagnostics.get("training/rollout_log_probs_diff_max"),
            "distillation_metrics": {key: value for key, value in diagnostics.items()
                                     if key.startswith("distillation/")},
        })
        original(**kwargs)
        completed.append(kwargs["step"])

    trainer._complete_step = verify_step
    trainer.train()
    if len(completed) < 2:
        raise RuntimeError("OPD ST did not complete two real updates")
    if any(count <= 0 for count in teacher_samples.values()):
        raise RuntimeError(f"Every configured real teacher must supervise training samples: {teacher_samples}")
    print(json.dumps({"status": "passed", "steps": completed, "teachers": len(distillation["teachers"]),
                      "teacher_samples": teacher_samples, "updates": evidence}))


if __name__ == "__main__":
    main()
