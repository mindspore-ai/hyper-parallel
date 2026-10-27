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
"""Frozen dual-arm codegen precision launcher for Qwen3-VL-MoE.

This launcher stays framework-free (no ``torch`` / ``hyper_parallel``
imports): it drives the frozen end-to-end flow
``examples/training_demo/codegen_validation/run_codegen_validation.sh``
with ``RUN_QVL=1``, which renders the two arms, runs them and asserts
per-step loss alignment. Full contract: ``TEST_REPORT.md`` in this
directory.
"""

import os
import subprocess
import sys
from pathlib import Path

from tests.common.mark_utils import arg_mark

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples" / "training_demo" / "codegen_validation"
    / "run_codegen_validation.sh"
)

# Assets the flow needs on the host (see TEST_REPORT.md § Prerequisites).
_REQUIRED_ENV = ("QVL_CONFIG",)


def _skip_reason() -> str | None:
    """Return why the flow cannot run, or ``None`` when it can."""
    if not _SCRIPT.is_file():
        return f"frozen flow script missing: {_SCRIPT}"
    if sys.platform == "win32":
        return "bash not assumed available on this platform; run on the NPU host"
    for name in _REQUIRED_ENV:
        if not os.environ.get(name):
            return (
                f"env {name} not set (point it at the cropped Qwen3-VL-MoE "
                "config dir, see TEST_REPORT.md § Prerequisites)"
            )
    return None


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_qwen3_vl_moe_codegen_dual_arm_precision() -> None:
    """Verify the Qwen3-VL-MoE codegen bundle reproduces native-arm losses.

    Feature: codegen dual-arm precision validation for qwen3_vl_moe.
    Description: Drive the frozen flow with RUN_QVL=1 — freshly generate
        the codegen bundle from the 8-NPU YAML, run a 10-step codegen-arm
        training and a 10-step native-arm training (seed 42 + full
        determinism), then compare per-step ``training/total_loss``.
    Expectation: The flow exits 0 and prints ``RESULT: PASSED`` — every
        arm completes 10 steps and per-step abs loss diff <= 5e-4.
    """
    reason = _skip_reason()
    if reason is not None:
        print(f"SKIP: {reason}")
        return
    env = os.environ.copy()
    env.setdefault("RUN_QVL", "1")
    env.setdefault("RUN_QWEN", "0")
    env.setdefault("RUN_DSV41", "0")
    env.setdefault("RUN_KIMI", "0")
    env.setdefault("LOSS_TOL", "0.0005")
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_SCRIPT.parents[2]),
    )
    tail = (result.stdout + result.stderr)[-4000:]
    print(tail)
    assert result.returncode == 0 and "RESULT: PASSED" in result.stdout, (
        f"codegen dual-arm flow failed (exit {result.returncode}); "
        f"last output:\n{tail}"
    )
