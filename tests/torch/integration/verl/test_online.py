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
"""Launch the real online VeRL stack; collection must not import Torch."""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level2", card_mark="allcards", essential_mark="unessential")
def test_online_first_step() -> None:
    """Verify online rollout and native actor execution on 16 devices."""
    required = ("MODEL_PATH", "TRAIN_DATA_PATH", "VAL_DATA_PATH", "VERL_ST_OUTPUT_DIR",
                "MCP_WINDOW_RUNNER", "MCP_WINDOW_DIR")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip(f"Real-device prerequisites are unset: {missing}")
    for name in ("MODEL_PATH", "TRAIN_DATA_PATH", "VAL_DATA_PATH"):
        assert Path(os.environ[name]).exists(), f"{name}={os.environ[name]}, expected an existing input asset"
    state = json.loads(subprocess.check_output([
        sys.executable, os.environ["MCP_WINDOW_RUNNER"], "status", "--work-dir", os.environ["MCP_WINDOW_DIR"],
    ], text=True))
    assert state["phase"] == "running" and state.get("current"), f"window={state}, expected a running task"
    assert state["remaining_seconds"] > 0, f"remaining={state['remaining_seconds']}, expected>0"
    os.kill(state["pid"], 0)
    root = Path(__file__).resolve().parents[4]
    output = Path(os.environ["VERL_ST_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=False)
    layers = os.environ.get("DEBUG_LAYERS", "3")
    env = dict(
        os.environ,
        DEBUG="1",
        DEBUG_LAYERS=layers,
        DEBUG_STEPS="1",
        DETERMINISTIC="1",
        ROLLOUT_SKIP="0",
        ROLLOUT_MODE="off",
        HP_SOURCE_ROOT=str(root),
    )
    observer = "tests.torch.integration.verl._online_observer"
    env["VERL_USE_EXTERNAL_MODULES"] = observer
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        "bash", str(root / "examples/verl/run.sh"),
        "skip.rollout_tq.enable=false", "trainer.total_training_steps=1",
        "++actor_rollout_ref.actor.engine.trainer_config.debug.check_fsdp_runtime=true",
        f"++actor_rollout_ref.actor.engine.trainer_config.model.config_overrides.num_hidden_layers={layers}",
        f"++actor_rollout_ref.ref.engine.trainer_config.model.config_overrides.num_hidden_layers={layers}",
        f'trainer.default_local_dir="{output / "checkpoints"}"',
    ]
    log = output / "train.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    assert result.returncode == 0, f"returncode={result.returncode}, expected=0; log={log}"
    assert "step:1 - " in log.read_text(encoding="utf-8"), f"Missing completed first step; log={log}"
    paths = sorted(output.glob("rank_*.jsonl"))
    expected = {f"rank_{rank}.jsonl" for rank in range(16)}
    assert {path.name for path in paths} == expected, f"rank files={[p.name for p in paths]}, expected={expected}"
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        counts = Counter(row["event"] for row in rows)
        assert counts == Counter(initialized=1, infer=1, train=1, scheduler=1), (
            f"{path}: counts={counts}, expected=1 each"
        )
        assert all(row["step"] == 1 for row in rows if row["event"] in ("train", "scheduler")), (
            f"{path}: steps={rows}, expected one completed optimizer step"
        )
