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
"""Run the normal RL loop with a strict pre-update train/rollout probability gate."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml

from rl.config.loader import load_config
from rl.trainer.build import build_trainer
from rl.trainer.distributed import run_synchronized


def check_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Require nonempty, finite and exactly equal sampled-token log probabilities."""
    keys = ("rollout_probs_diff_valid", "log_probs_valid_tokens", "rollout_log_probs_diff_max",
            "rollout_log_probs_diff_mean", "actor_log_probs_mean", "rollout_log_probs_mean")
    observed = {key: float(metrics[f"training/{key}"]) for key in keys}
    if (not all(math.isfinite(value) for value in observed.values())
            or observed["rollout_probs_diff_valid"] != 1
            or observed["log_probs_valid_tokens"] <= 0
            or observed["rollout_log_probs_diff_max"] != 0
            or observed["rollout_log_probs_diff_mean"] != 0):
        raise RuntimeError(f"Train/rollout consistency failed before optimizer update: {observed}")
    return observed


def check_metrics_on_all_ranks(metrics: dict[str, float]) -> dict[str, float]:
    """Validate rank-zero global diagnostics and propagate failures to every rank."""
    rank = int(os.environ.get("RANK", "0"))
    return run_synchronized(
        "train/rollout consistency",
        lambda: check_metrics(metrics) if rank == 0 else {},
    )


def run(config: dict[str, Any], report_path: Path) -> None:
    """Observe production diagnostics without changing tokens, rewards or numerical paths."""
    if config["train"]["max_steps"] != 2 or config["logging"]["log_steps"] != 1:
        raise ValueError("This example requires max_steps=2 and log_steps=1")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0 and report_path.exists():
        report_path.unlink()
    if int(os.environ.get("RANK", "0")) == 0:
        report_path.with_name("config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    trainer = build_trainer(config)
    prepare = trainer.targets.prepare
    checks = []

    def _checked_prepare(rollout, collect_diagnostics, timings):
        if rollout.rollout_log_probs is None:
            raise RuntimeError("Independent rollout sampling log probabilities are required")
        experience, metrics = prepare(rollout, collect_diagnostics, timings)
        observed = check_metrics_on_all_ranks(metrics)
        checks.append({"step": trainer.state.global_step + 1, **observed})
        if int(os.environ.get("RANK", "0")) == 0:
            logging.info("CONSISTENCY_CHECK %s", json.dumps(checks[-1]))
        return experience, metrics

    # This example adds an observation gate at the existing pre-update boundary.
    trainer.targets.prepare = _checked_prepare
    trainer.train()
    if trainer.state.global_step != 2 or [check["step"] for check in checks] != [1, 2]:
        raise RuntimeError("The normal RL loop did not complete and verify both updates")
    if int(os.environ.get("RANK", "0")) == 0:
        report = {"passed": True, "completed_steps": 2, "checks": checks,
                  "training": config["train"]["accelerator"],
                  "rollout": config["rollout"]["vllm"],
                  "scope": "sampled-token log probabilities before each optimizer update"}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logging.info("CONSISTENCY_PASS report=%s", report_path)


def main() -> None:
    """Load the explicit recipe and launch the ordinary synchronous trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--report", type=Path, default=Path("/results/run/consistency.json"))
    args, overrides = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(load_config(args.config, overrides), args.report)


if __name__ == "__main__":
    main()
