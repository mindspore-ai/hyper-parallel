#!/usr/bin/env python3
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
"""Emit deterministic synthetic losses for the accuracy-runner demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=20, type=int)
    return parser.parse_args()


def main() -> int:
    """Write one globally normalized loss record per synthetic optimizer step."""
    args = _parse_args()
    if args.steps < 20:
        raise ValueError("the trajectory demonstration requires at least 20 steps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for step in range(1, args.steps + 1):
            baseline_loss = 2.0 - 0.015 * (step - 1)
            delta = 0.0
            if args.role == "candidate":
                delta = 8e-4 if step % 2 == 0 else -8e-4
            valid_tokens = 128 + step % 3
            loss = baseline_loss + delta
            record = {
                "step": step,
                "loss": loss,
                "loss_sum": loss * valid_tokens,
                "valid_tokens": valid_tokens,
                "batch_fingerprint": f"synthetic-batch-{step:04d}",
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
